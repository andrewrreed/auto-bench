import sys
import uuid
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Union
from collections import defaultdict
import asyncio
import nest_asyncio
from autobench.scheduler import Scheduler

from autobench.scenario import (
    Scenario,
    ScenarioGroup,
    ScenarioGroupResult,
)


@dataclass
class BenchmarkResult:
    benchmark_id: str
    scenario_group_results: List[ScenarioGroupResult]
    output_dir: str = None


class Benchmark:
    """

    The primary mechanism for running Scenario's.

    Can be initialized from:
    - A single or multiple scenarios
    - For each scenario, checks if deployment is active or not
    - If not, will add it to the scheduler for execution
    - Standardizes the output structure for all Benchmark runs:
        - benchmark_name(or id) / deployment_name / scenario_name

    """

    def __init__(
        self,
        scenarios: Union[Scenario, List[Scenario], ScenarioGroup, List[ScenarioGroup]],
        output_dir: str = None,
    ):
        self.output_dir = output_dir
        self.benchmark_id = str(uuid.uuid4())
        self.benchmark_name = f"benchmark_{self.benchmark_id}"
        self.output_dir = os.path.join(output_dir, self.benchmark_name)
        self.scenario_groups = self._get_scenario_groups(scenarios)
        self.namespace = self._get_namespace()

    def _get_scenario_groups(
        self,
        scenarios: Union[Scenario, List[Scenario], ScenarioGroup, List[ScenarioGroup]],
    ):
        if isinstance(scenarios, Scenario):
            return [self._parse_scenario_groups([scenarios])]
        elif isinstance(scenarios, ScenarioGroup):
            return [scenarios]
        elif isinstance(scenarios, list):
            if all(isinstance(s, Scenario) for s in scenarios):
                return self._parse_scenario_groups(scenarios)
            elif all(isinstance(s, ScenarioGroup) for s in scenarios):
                return scenarios
            else:
                raise ValueError(
                    "Invalid list of scenarios or scenario groups provided. Make sure the list is all of the same type."
                )

    @staticmethod
    def _parse_scenario_groups(scenarios: List[Scenario]):
        groups = defaultdict(list)
        for scenario in scenarios:
            groups[scenario.deployment].append(scenario)

        return [ScenarioGroup(deployment=k, scenarios=v) for k, v in groups.items()]

    def _assert_existing_deployments_running(self):
        """
        Assert that all _existing_ deployments in the scenario groups are running.

        """
        for scenario_group in self.scenario_groups:
            if (
                scenario_group.deployment._exists
                and scenario_group.deployment.endpoint_status() != "running"
            ):
                raise Exception(
                    f"You initialized Deployment: {scenario_group.deployment.deployment_name} from an existing endpoint, but it is not running. Please start all _existing_ deployments before running the benchmark."
                )

    def _get_namespace(self):
        namespaces = [
            sg.deployment.deployment_config.namespace for sg in self.scenario_groups
        ]
        if len(set(namespaces)) > 1:
            raise Exception(
                "Benchmarking deployments across multiple namespaces is not supported. Please ensure all deployments are in the same namespace."
            )
        else:
            return namespaces[0]

    async def _run_scheduler_async(self):
        self._assert_existing_deployments_running()
        scheduler = Scheduler(
            scenario_groups=self.scenario_groups,
            namespace=self.namespace,
        )
        await scheduler.run()
        return scheduler

    def run(self):
        """Run the benchmark, compatible with both CLI and Jupyter notebooks."""
        if "ipykernel" in sys.modules:
            # Running in Jupyter notebook
            nest_asyncio.apply()

        scheduler = asyncio.run(self._run_scheduler_async())

        benchmark_result = BenchmarkResult(
            benchmark_id=self.benchmark_id, scenario_group_results=scheduler.results
        )
        if self.output_dir:
            benchmark_result.output_dir = self.output_dir
            self.save_benchmark_results(benchmark_result, self.output_dir)

        return benchmark_result

    @staticmethod
    def save_benchmark_results(results: BenchmarkResult, output_dir: str):
        """
        Save the benchmark results to a directory.
        """

        os.makedirs(output_dir, exist_ok=False)

        results = asdict(results)

        for sg in results["scenario_group_results"]:
            for s in sg["scenario_results"]:
                k6_script = s.get("k6_script", None)
                if k6_script:
                    script_dir = os.path.join(output_dir, "scripts")
                    os.makedirs(script_dir, exist_ok=True)
                    file_path = os.path.join(script_dir, f"{s['scenario_id']}.js")
                    with open(file_path, "w") as f:
                        f.write(k6_script)
                    s["k6_script"] = file_path

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f)
