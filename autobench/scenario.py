import os
import uuid
import json
import subprocess
import shutil
import time

from typing import List
from dataclasses import asdict
from loguru import logger

from autobench.data import BenchmarkDataset
from autobench.deployment import Deployment
from autobench.executor import K6Executor
from autobench.benchmark import ScenarioResult, ScenarioGroupResult

BENCHMARK_DATA_DIR = os.path.join(os.path.dirname(__file__), "benchmark_data")


class Scenario:
    """
    A single benchmark scenario, which is a single executor run against a deployment and dataset.

    """

    def __init__(
        self,
        deployment: Deployment,
        executor: K6Executor,
        benchmark_dataset: BenchmarkDataset,
        output_dir: str,
    ):
        self.deployment = deployment
        self.benchmark_dataset = benchmark_dataset
        self.executor = executor
        self.data_file = benchmark_dataset.file_path
        self.scenario_id = str(uuid.uuid4())
        self.scenario_name = "scenario_" + self.scenario_id
        self.output_dir = os.path.join(
            output_dir, deployment.deployment_name, self.scenario_name
        )

    def _prepare_benchmark(self):
        self.executor.update_variables(
            host=self.deployment.endpoint.url,
            data_file=self.data_file,
            out_dir=self.output_dir,
        )
        self.executor.render_script()
        logger.debug(f"Prepared benchmark for scenario: {self.scenario_id}")

    def _run(self):

        logger.info(f"Running scenario: {self.scenario_id}")
        logger.info(f"HAS ENDPOINT?: {hasattr(self.deployment, 'endpoint')}")

        if self.deployment.endpoint.status != "running":
            raise Exception(
                f"Deployment {self.deployment.deployment_id} is not running, will not run benchmark."
            )

        logger.info(f"Starting scenario: {self.scenario_id}")
        os.makedirs(self.output_dir)
        self._prepare_benchmark()

        # start a k6 subprocess
        logger.info(f"Running K6 for scenario: {self.scenario_id}")
        # args = f"~/.local/bin/k6-sse run --out json={self.output_dir}/results.json {self.executor.rendered_file}"
        args = f"~/.local/bin/k6-sse run --quiet {self.executor.rendered_file}"
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
        )

        stdout, stderr = self.process.communicate()

        if self.process.returncode != 0:
            logger.error(
                f"k6 process failed with return code {self.process.returncode}"
            )
            logger.error(f"stderr: {stderr}")
            return None

        # Log the entire output for debugging
        logger.debug(f"k6 stdout: {stdout}")

        # Try to parse the last line as JSON
        try:
            result_summary = json.loads(stdout.strip())
        except json.JSONDecodeError:
            logger.error("Failed to parse k6 output as JSON")
            result_summary = None

        return ScenarioResult(
            scenario_id=self.scenario_id,
            deployment_id=self.deployment.deployment_id,
            executor_type=self.executor.name,
            executor_variables=self.executor.variables,
            k6_script=self._get_scenario_script(),
            metrics=result_summary,
        )

    def _get_scenario_script(self):
        with open(self.executor.rendered_file, "r") as f:
            script = f.read()
        return script

        # logger.info(f"Saving results for scenario: {self.scenario_id}")
        # self.save_scenario_details()
        # self.save_scenario_script()
        # self.save_deployment_details()

        # logger.info(f"Scenario {self.scenario_id} completed")

    # def save_scenario_details(self):

    #     scenario_details_path = f"{self.output_dir}/scenario_details.json"

    #     try:
    #         scenario_details = {
    #             "scenario_id": self.scenario_id,
    #             "host": self.deployment.endpoint.url,
    #             "executor_type": self.executor.name,
    #             **self.executor.variables,
    #         }
    #         with open(scenario_details_path, "w") as f:
    #             json.dump(scenario_details, f)

    #     except Exception as e:
    #         logger.error(f"Error saving scenario details: {str(e)}")

    # def save_scenario_script(self):
    #     script_path = f"{self.output_dir}/{self.executor.rendered_file.split('/')[-1]}"
    #     shutil.copy(self.executor.rendered_file, script_path)
    #     logger.debug(f"Scenario script saved to: {script_path}")

    def save_deployment_details(self):
        # Save deployment details
        deployment_details = {
            "tgi_config": asdict(self.deployment.tgi_config),
            "instance_config": asdict(self.deployment.instance_config),
            "deployment": {
                "deployment_id": self.deployment.deployment_id,
                "deployment_name": self.deployment.deployment_name,
                **self.deployment.endpoint.raw,
            },
        }

        deployment_details_path = os.path.join(
            self.output_dir, "deployment_details.json"
        )

        with open(deployment_details_path, "w") as f:
            json.dump(deployment_details, f, indent=4)


class ScenarioGroup:
    def __init__(self, deployment: Deployment, scenarios: List[Scenario]):
        self.deployment = deployment
        self.scenarios = scenarios
        self.scenario_results = []

    def _run(self):
        for scenario in self.scenarios:
            scenario_result = scenario._run()
            self.scenario_results.append(scenario_result)
            time.sleep(10)

        return ScenarioGroupResult(
            deployment_id=self.deployment.deployment_id,
            scenario_results=self.scenario_results,
        )


# class BenchmarkRunner:
#     """
#     TO-DO:
#     - add a benchmark config that specifies the executor type(s) and executor params to test
#     - add a benchmark_id and save out the benchmark run's details to a file inside the deployment's results dir

#     """

#     def __init__(
#         self,
#         deployment: Deployment,
#         benchmark_dataset: BenchmarkDataset,
#         output_dir: str,
#     ):
#         self.deployment = deployment
#         self.benchmark_dataset = benchmark_dataset
#         self.output_dir = os.path.join(output_dir, self.deployment.deployment_name)
#         # self.arrival_rates = self._get_arrival_rates()
#         self.arrival_rates = [1, 10, 25, 50, 75, 100]
#         logger.info(
#             f"Initialized BenchmarkRunner for deployment: {deployment.deployment_id}"
#         )

#     def _get_arrival_rates(self):
#         arrival_rates = list(range(0, 200, 10))
#         arrival_rates[0] = 1
#         arrival_rates.append(200)
#         return arrival_rates

#     def run_benchmark(self):
#         """ """

#         logger.info(
#             f"Starting benchmark for deployment: {self.deployment.deployment_id}"
#         )
#         for arrival_rate in self.arrival_rates:
#             logger.info(f"Running benchmark for arrival rate: {arrival_rate}")
#             executor = K6ConstantArrivalRateExecutor(
#                 pre_allocated_vus=500,  # NOTE: should be 2k for full tests
#                 rate_per_second=arrival_rate,
#                 duration="15s",
#             )
#             scenario = Scenario(
#                 host=self.deployment.endpoint.url,
#                 executor=executor,
#                 data_file=self.benchmark_dataset.file_path,
#                 output_dir=self.output_dir,
#             )
#             scenario.run()
#             time.sleep(10)
#             logger.info(f"Benchmark for arrival rate {arrival_rate} completed")

#         # Save deployment details
#         deployment_details = {
#             "tgi_config": asdict(self.deployment.tgi_config),
#             "instance_config": asdict(self.deployment.instance_config),
#             "deployment": {
#                 "deployment_id": self.deployment.deployment_id,
#                 "deployment_name": self.deployment.deployment_name,
#                 **self.deployment.endpoint.raw,
#             },
#         }

#         deployment_details_path = os.path.join(
#             self.output_dir, "deployment_details.json"
#         )

#         with open(deployment_details_path, "w") as f:
#             json.dump(deployment_details, f, indent=4)

#         logger.info(f"Deployment details saved to: {deployment_details_path}")
#         logger.info(
#             f"Benchmark for deployment {self.deployment.deployment_id} completed"
#         )
