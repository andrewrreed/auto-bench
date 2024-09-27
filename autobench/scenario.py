import os
import uuid
import json
import subprocess
import time

from typing import List
from dataclasses import dataclass
from typing import Dict, Any, Optional
from loguru import logger

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
    deployment_status: Optional[Dict[str, Any]] = None


class Scenario:
    """
    A single benchmark scenario, which is a single executor run against a deployment and dataset.

    """

    def __init__(
        self,
        deployment: Deployment,
        executor: K6Executor,
        benchmark_dataset: BenchmarkDataset,
    ):
        self.deployment = deployment
        self.benchmark_dataset = benchmark_dataset
        self.executor = executor
        self.data_file = benchmark_dataset.file_path
        self.scenario_id = str(uuid.uuid4())
        self.scenario_name = "scenario_" + self.scenario_id

    def _prepare_benchmark(self):
        self.executor.update_variables(
            host=self.deployment.endpoint.url,
            data_file=self.data_file,
        )
        self.executor.render_script()
        logger.debug(f"Prepared benchmark for scenario: {self.scenario_id}")

    def _run(self):

        logger.info(f"Running scenario: {self.scenario_id}")

        if self.deployment.endpoint.status != "running":
            raise Exception(
                f"Deployment {self.deployment.deployment_id} is not running, will not run benchmark."
            )

        logger.info(f"Starting scenario: {self.scenario_id}")
        self._prepare_benchmark()

        # start a k6 subprocess
        logger.info(f"Running K6 for scenario: {self.scenario_id}")
        args = f"~/.local/bin/k6-sse run --quiet {self.executor.rendered_file}"
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
        with open(self.executor.rendered_file, "r") as f:
            script = f.read()
        return script


class ScenarioGroup:
    def __init__(self, deployment: Deployment, scenarios: List[Scenario]):
        self.deployment = deployment
        self.scenarios = scenarios
        self.scenario_results = []
        self._validate_scenarios()

    def _validate_scenarios(self):
        for scenario in self.scenarios:
            if scenario.deployment.deployment_id != self.deployment.deployment_id:
                raise ValueError(
                    "All scenarios must have the same deployment_id as the scenario group."
                )

    def _run(self):
        for scenario in self.scenarios:
            scenario_result = scenario._run()
            self.scenario_results.append(scenario_result)
            time.sleep(10)

        return ScenarioGroupResult(
            deployment_id=self.deployment.deployment_id,
            scenario_results=self.scenario_results,
        )
