import tempfile

from loguru import logger
from jinja2 import Environment, select_autoescape, PackageLoader

ENV = Environment(loader=PackageLoader("autobench"), autoescape=select_autoescape())


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
        template = ENV.get_template(self.template_name)
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
