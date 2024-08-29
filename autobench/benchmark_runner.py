import os
import uuid
import json
import tempfile
import subprocess
import shutil
from jinja2 import Environment, select_autoescape, PackageLoader
from autobench.config import BenchmarkConfig

BENCHMARK_DATA_DIR = os.path.join(os.path.dirname(__file__), "benchmark_data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")

env = Environment(loader=PackageLoader("autobench"), autoescape=select_autoescape())


class K6Executor:
    def __init__(self, name: str, template_name: str):
        self.name = name
        self.template_name = template_name
        self.variables = {}

    def update_variables(self, **kwargs):
        self.variables.update(kwargs)

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
    ):
        self.host = host
        self.executor = executor
        self.data_file = data_file
        self.scenario_id = str(uuid.uuid4())
        self.output_dir = os.path.join(RESULTS_DIR, self.scenario_id)

        os.makedirs(self.output_dir)

    def _prepare_benchmark(self):
        self.executor.update_variables(
            host=self.host,
            data_file=self.data_file,
            out_dir=self.output_dir,
        )
        self.executor.render_script()

    def run(self):
        print(f"Preparing scenario {self.scenario_id}")
        self._prepare_benchmark()

        # start a k6 subprocess
        print(f"Running scenario {self.scenario_id}")
        args = f"~/.local/bin/k6-sse run --out json={self.output_dir}/results.json {self.executor.rendered_file}"
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        while buffer := os.read(
            self.process.stdout.fileno(), 2048
        ):  # read the output of the process, don't buffer on new lines
            print(buffer.decode(), end="")
        self.process.wait()

        print(f"Saving scenario {self.scenario_id} results")
        self.add_config_to_summary()
        self.add_config_to_results()
        self.save_scenario_details()
        self.save_scenario_script()

        print(f"Scenario {self.scenario_id} complete")

    def add_config_to_summary(self):
        summary_path = f"{self.output_dir}/summary.json"
        with open(summary_path, "r") as f:
            summary = json.load(f)

        summary["config"] = {
            "host": self.host,
            "executor_type": self.executor.name,
            **self.executor.variables,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f)

    def add_config_to_results(self):
        results_path = f"{self.output_dir}/results.json"
        with open(results_path, "r") as f:
            results = f.readlines()
            # append the k6 config to the results in jsonlines format
            results += "\n"
            results += json.dumps(
                {
                    "host": self.host,
                    "executor_type": self.executor.name,
                    **self.executor.variables,
                }
            )
        with open(results_path, "w") as f:
            f.writelines(results)

    def save_scenario_details(self):
        with open(f"{self.output_dir}/scenario_details.json", "w") as f:
            scenario_details = {
                "scenario_id": self.scenario_id,
                "host": self.host,
                "executor_type": self.executor.name,
                **self.executor.variables,
            }
            json.dump(scenario_details, f)

    def save_scenario_script(self):
        script_path = f"{self.output_dir}/{self.executor.rendered_file.split('/')[-1]}"
        shutil.copy(self.executor.rendered_file, script_path)


