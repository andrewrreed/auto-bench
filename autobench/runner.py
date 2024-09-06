import os
import uuid
import json
import tempfile
import subprocess
import shutil
import time
from dataclasses import asdict
from jinja2 import Environment, select_autoescape, PackageLoader
from loguru import logger

from autobench.data import BenchmarkDataset
from autobench.deployment import Deployment

BENCHMARK_DATA_DIR = os.path.join(os.path.dirname(__file__), "benchmark_data")

env = Environment(loader=PackageLoader("autobench"), autoescape=select_autoescape())


class K6Executor:
    def __init__(self, name: str, template_name: str):
        self.name = name
        self.template_name = template_name
        self.variables = {}
        logger.info(f"Initialized K6Executor: {name} with template: {template_name}")

    def update_variables(self, **kwargs):
        self.variables.update(kwargs)
        logger.debug(f"Updated variables for {self.name}: {kwargs}")

    def render_script(self):
        template = env.get_template(self.template_name)
        _, path = tempfile.mkstemp(
            prefix="autobench_",
            suffix="_k6_script.js",
        )

        with open(path, "w") as f:
            rendered_script = template.render(**self.variables)
            f.write(rendered_script)

        self.rendered_file = path
        logger.info(f"Rendered K6 script to: {path}")


class K6ConstantArrivalRateExecutor(K6Executor):
    def __init__(self, pre_allocated_vus: int, rate_per_second: int, duration: str):
        super().__init__(
            name="constant_arrival_rate", template_name="k6_constant_arrival_rate.js.j2"
        )
        self.variables = {
            "pre_allocated_vus": pre_allocated_vus,
            "rate": rate_per_second,
            "duration": duration,
        }


class Scenario:

    def __init__(
        self,
        host: str,
        executor: K6Executor,
        data_file: str,
        output_dir: str,
    ):
        self.host = host
        self.executor = executor
        self.data_file = data_file
        self.scenario_id = str(uuid.uuid4())
        self.scenario_name = "scenario_" + self.scenario_id
        self.output_dir = os.path.join(output_dir, self.scenario_name)

        os.makedirs(self.output_dir)
        logger.info(f"Initialized Scenario: {self.scenario_id} for host: {host}")

    def _prepare_benchmark(self):
        self.executor.update_variables(
            host=self.host,
            data_file=self.data_file,
            out_dir=self.output_dir,
        )
        self.executor.render_script()
        logger.debug(f"Prepared benchmark for scenario: {self.scenario_id}")

    def run(self):
        logger.info(f"Starting scenario: {self.scenario_id}")
        self._prepare_benchmark()

        # start a k6 subprocess
        logger.info(f"Running K6 for scenario: {self.scenario_id}")
        args = f"~/.local/bin/k6-sse run --out json={self.output_dir}/results.json {self.executor.rendered_file}"
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        while buffer := os.read(
            self.process.stdout.fileno(),
            2048,
            # ):  # read the output of the process, don't buffer on new lines
            #     print(buffer.decode(), end="")
        ):
            logger.debug(buffer.decode().strip())
        self.process.wait()

        logger.info(f"Saving results for scenario: {self.scenario_id}")
        self.save_scenario_details()
        self.save_scenario_script()

        logger.info(f"Scenario {self.scenario_id} completed")

    def save_scenario_details(self):
        scenario_details_path = f"{self.output_dir}/scenario_details.json"
        logger.info(f"Saving scenario details to: {scenario_details_path}")
        try:
            scenario_details = {
                "scenario_id": self.scenario_id,
                "host": self.host,
                "executor_type": self.executor.name,
                **self.executor.variables,
            }
            with open(scenario_details_path, "w") as f:
                json.dump(scenario_details, f)
            logger.debug("Scenario details successfully saved")
        except Exception as e:
            logger.error(f"Error saving scenario details: {str(e)}")

    def save_scenario_script(self):
        script_path = f"{self.output_dir}/{self.executor.rendered_file.split('/')[-1]}"
        shutil.copy(self.executor.rendered_file, script_path)
        logger.debug(f"Scenario script saved to: {script_path}")


class BenchmarkRunner:
    """
    TO-DO:
    - add a benchmark config that specifies the executor type(s) and executor params to test
    - add a benchmark_id and save out the benchmark run's details to a file inside the deployment's results dir

    """

    def __init__(
        self,
        deployment: Deployment,
        benchmark_dataset: BenchmarkDataset,
        output_dir: str,
    ):
        self.deployment = deployment
        self.benchmark_dataset = benchmark_dataset
        self.output_dir = os.path.join(output_dir, self.deployment.deployment_name)
        # self.arrival_rates = self._get_arrival_rates()
        self.arrival_rates = [1, 10, 50]
        logger.info(
            f"Initialized BenchmarkRunner for deployment: {deployment.deployment_id}"
        )

    def _get_arrival_rates(self):
        arrival_rates = list(range(0, 200, 10))
        arrival_rates[0] = 1
        arrival_rates.append(200)
        return arrival_rates

    def run_benchmark(self):
        """ """

        logger.info(
            f"Starting benchmark for deployment: {self.deployment.deployment_id}"
        )
        for arrival_rate in self.arrival_rates:
            logger.info(f"Running benchmark for arrival rate: {arrival_rate}")
            executor = K6ConstantArrivalRateExecutor(
                pre_allocated_vus=50,  # NOTE: should be 2k for full tests
                rate_per_second=arrival_rate,
                duration="5s",
            )
            scenario = Scenario(
                host=self.deployment.endpoint.url,
                executor=executor,
                data_file=self.benchmark_dataset.file_path,
                output_dir=self.output_dir,
            )
            scenario.run()
            time.sleep(10)
            logger.info(f"Benchmark for arrival rate {arrival_rate} completed")

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

        logger.info(f"Deployment details saved to: {deployment_details_path}")
        logger.info(
            f"Benchmark for deployment {self.deployment.deployment_id} completed"
        )
