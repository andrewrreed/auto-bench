import os
import json
import shutil
import tempfile
import subprocess
from jinja2 import Environment, select_autoescape, PackageLoader


BENCHMARK_DATA_DIR = os.path.join(os.path.dirname(__file__), "benchmark_data")

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
            suffix="k6_script",
        )

        with open(path, "w") as f:
            rendered_script = template.render(**self.variables)
            f.write(rendered_script)

        self.rendered_file = path


class K6ConstantArrivalRateExecutor(K6Executor):
    def __init__(self, pre_allocated_vus: int, rate_per_second: int, duration: str):
        super().__init__("constant_arrival_rate", "k6_constant_arrival_rate.js.j2")
        self.variables = {
            "pre_allocated_vus": pre_allocated_vus,
            "rate": rate_per_second,
            "duration": duration,
        }


class K6Config:
    def __init__(self, host: str, executor: K6Executor, data_file: str):
        self.host = host
        self.executor = executor
        self.data_file = data_file
        self._temp_dir = tempfile.mkdtemp(prefix="autobench_", suffix="_k6_results")

        self.executor.update_variables(
            host=host,
            data_file=data_file,
            data_path=BENCHMARK_DATA_DIR,
            temp_dir=self._temp_dir,
        )

    def __str__(self):
        return f"K6Config(url={self.host} executor={self.executor} data_file={self.data_file})"


class K6Benchmark:
    def __init__(self, config: K6Config, output_dir: str):
        self.config = config
        self.output_dir = output_dir

    def _prepare_data(self):
        pass

    def run(self):
        os.makedirs(self.config._temp_dir, exist_ok=True)
        self.config.executor.render_script()
        args = f"~/.local/bin/k6-sse run --out json={self.config._temp_dir}/results.json {self.config.executor.rendered_file}"

        # start a k6 subprocess
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        while buffer := os.read(
            self.process.stdout.fileno(), 2048
        ):  # read the output of the process, don't buffer on new lines
            print(buffer.decode(), end="")
        self.process.wait()
        os.makedirs(self._get_output_dir(), exist_ok=True)
        self.add_config_to_summary()
        self.add_config_to_results()
        shutil.rmtree(self.config._temp_dir)

    def add_config_to_summary(self):
        with open(f"{self.config._temp_dir}/summary.json", "r") as f:
            summary = json.load(f)
            summary["config"] = {
                "host": self.config.host,
                "executor_type": self.config.executor.name,
                **self.config.executor.variables,
            }
            with open(self.get_summary_path(), "w") as f2:
                json.dump(summary, f2)

    def add_config_to_results(self):
        with open(f"{self.config._temp_dir}/summary.json", "r") as f:
            results = f.readlines()
            # append the k6 config to the results in jsonlines format
            results += "\n"
            results += json.dumps(
                {
                    "host": self.config.host,
                    "executor_type": self.config.executor.name,
                    **self.config.executor.variables,
                }
            )
            with open(self.get_results_path(), "w") as f2:
                f2.writelines(results)

    def _get_output_dir(self):
        return self.output_dir

    def get_results_path(self):
        return f"{self._get_output_dir()}.results.json"

    def get_summary_path(self):
        return f"{self._get_output_dir()}.summary.json"
