import os
import uuid
import json
import subprocess
import time

from typing import List
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from loguru import logger

from autobench.config import K6_BIN
from autobench.data import BenchmarkDataset
from autobench.deployment import Deployment
from autobench.executor import K6Executor


@dataclass
class ScenarioResult:
    scenario_id: str
    deployment_id: str
    executor_type: str
    executor_variables: Dict[str, Any]
    k6_script: str
    metrics: Dict[str, Any]
    scenario_status: Optional[Dict[str, Any]] = None


@dataclass
class ScenarioGroupResult:
    deployment_id: str
    scenario_results: List[ScenarioResult]
    deployment_details: Optional[Dict[str, Any]] = None
    deployment_status: Optional[Dict[str, Any]] = None


class Scenario:
    """
    Represents a single benchmark scenario, which is a single executor run against a deployment and dataset.

    Attributes:
        deployment: The Deployment object to be benchmarked.
        benchmark_dataset: The BenchmarkDataset object containing the data for the benchmark.
        executor: The K6Executor object used to run the benchmark.
        data_file: The file path of the benchmark dataset.
        scenario_id: A unique identifier for the scenario.
        scenario_name: A name for the scenario based on the scenario_id.
    """

    def __init__(
        self,
        deployment: Deployment,
        executor: K6Executor,
        benchmark_dataset: BenchmarkDataset,
    ):
        """
        Initializes a new Scenario instance.

        Args:
            deployment: The Deployment object to be benchmarked.
            executor: The K6Executor object used to run the benchmark.
            benchmark_dataset: The BenchmarkDataset object containing the data for the benchmark.
        """
        self.deployment = deployment
        self.benchmark_dataset = benchmark_dataset
        self.executor = executor
        self.data_file = benchmark_dataset.file_path
        self.scenario_id = str(uuid.uuid4())
        self.scenario_name = "scenario_" + self.scenario_id

    def _prepare_benchmark(self):
        """
        Prepares the benchmark by updating executor variables and rendering the script.
        """
        self.executor.update_variables(
            host=self.deployment.endpoint.url,
            data_file=self.data_file,
        )
        self.executor.render_script()
        logger.debug(f"Prepared benchmark for scenario: {self.scenario_id}")

    def _run(self):
        """
        Runs the benchmark scenario and returns the result.

        Returns:
            ScenarioResult: The result of the benchmark scenario.

        Raises:
            Exception: If the deployment is not running.
        """
        logger.info(f"Running scenario: {self.scenario_id}")

        if self.deployment.endpoint.status != "running":
            raise Exception(
                f"Deployment {self.deployment.deployment_id} is not running, will not run benchmark."
            )

        logger.info(f"Starting scenario: {self.scenario_id}")
        self._prepare_benchmark()

        # start a k6 subprocess
        logger.info(f"Running K6 for scenario: {self.scenario_id}")
        args = f"{K6_BIN} run --quiet {self.executor.rendered_file}"
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
        )

        stdout, stderr = self.process.communicate()

        scenario_status = {
            "status": None,
            "error": None,
        }
        if self.process.returncode != 0:
            logger.error(
                f"k6 process failed with return code {self.process.returncode}"
            )
            logger.error(f"stderr: {stderr}")
            scenario_status["status"] = "failed"
            scenario_status["error"] = stderr

        try:
            result_summary = json.loads(stdout.strip())
            scenario_status["status"] = "success"
        except json.JSONDecodeError:
            scenario_status["status"] = "failed"
            scenario_status["error"] = "Failed to parse k6 output as JSON"
            result_summary = None

        logger.info(f"Scenario {self.scenario_id} completed")

        return ScenarioResult(
            scenario_id=self.scenario_id,
            deployment_id=self.deployment.deployment_id,
            executor_type=self.executor.name,
            executor_variables=self.executor.variables,
            k6_script=self._get_scenario_script(),
            metrics=result_summary,
            scenario_status=scenario_status,
        )

    def _get_scenario_script(self):
        """
        Retrieves the rendered K6 script for the scenario.

        Returns:
            str: The contents of the rendered K6 script.
        """
        with open(self.executor.rendered_file, "r") as f:
            script = f.read()
        return script


class ScenarioGroup:
    """
    Represents a group of benchmark scenarios that share the same deployment and dataset.

    Attributes:
        deployment: The Deployment object to be benchmarked.
        benchmark_dataset: The BenchmarkDataset object containing the data for the benchmarks.
        executors: A list of K6Executor objects or a single K6Executor object.
        scenarios: A list of Scenario objects in the group.
        scenario_results: A list to store the results of each scenario run.
    """

    def __init__(
        self,
        deployment: Deployment,
        benchmark_dataset: BenchmarkDataset,
        executors: Union[K6Executor, List[K6Executor]],
    ):
        """
        Initializes a new ScenarioGroup instance.

        Args:
            deployment: The Deployment object to be benchmarked.
            benchmark_dataset: The BenchmarkDataset object containing the data for the benchmarks.
            executors: A list of K6Executor objects or a single K6Executor object.
        """
        self.deployment = deployment
        self.benchmark_dataset = benchmark_dataset
        self.executors = executors if isinstance(executors, list) else [executors]
        self.scenarios = self._build_scenarios()
        self._validate_scenarios()
        self.scenario_results = []

    def _build_scenarios(self):
        """
        Builds a list of Scenario objects based on the executors.

        Returns:
            List[Scenario]: A list of Scenario objects.
        """
        scenarios = []
        for executor in self.executors:
            scenarios.append(
                Scenario(
                    deployment=self.deployment,
                    benchmark_dataset=self.benchmark_dataset,
                    executor=executor,
                )
            )
        return scenarios

    def _validate_scenarios(self):
        """
        Validates that all scenarios in the group have the same deployment_id.

        Raises:
            ValueError: If any scenario has a different deployment_id than the group.
        """
        for scenario in self.scenarios:
            if scenario.deployment.deployment_id != self.deployment.deployment_id:
                raise ValueError(
                    "All scenarios must have the same deployment_id as the scenario group."
                )

    def _run(self):
        """
        Runs all scenarios in the group and collects their results.

        Returns:
            ScenarioGroupResult: The result of running all scenarios in the group.
        """
        for scenario in self.scenarios:
            scenario_result = scenario._run()
            self.scenario_results.append(scenario_result)
            time.sleep(10)

        return ScenarioGroupResult(
            deployment_id=self.deployment.deployment_id,
            scenario_results=self.scenario_results,
            deployment_details={
                "tgi_config": asdict(self.deployment.tgi_config),
                "instance_config": asdict(self.deployment.instance_config),
                "endpoint_details": {
                    **self.deployment.endpoint.raw,
                },
            },
        )
