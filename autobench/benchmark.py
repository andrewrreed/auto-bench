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
    ScenarioResult,
)

from autobench.config import BENCHMARK_RESULTS_DIR


@dataclass
class BenchmarkResult:
    benchmark_id: str
    scenario_group_results: List[ScenarioGroupResult]
    output_dir: str = None

    def save(self, output_dir: str = None):
        """
        Save the benchmark results to a directory.

        Args:
            output_dir (str, optional): The directory to save the results in. If not provided, uses self.output_dir.
        """
        if output_dir:
            self.output_dir = output_dir
        if not self.output_dir:
            raise ValueError("No output directory specified.")

        os.makedirs(self.output_dir, exist_ok=False)

        results = asdict(self)

        for sg in results["scenario_group_results"]:
            for s in sg["scenario_results"]:
                k6_script = s.get("k6_script", None)
                if k6_script:
                    script_dir = os.path.join(self.output_dir, "scripts")
                    os.makedirs(script_dir, exist_ok=True)
                    file_path = os.path.join(script_dir, f"{s['scenario_id']}.js")
                    with open(file_path, "w") as f:
                        f.write(k6_script)
                    s["k6_script"] = file_path

        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f)

    @classmethod
    def from_directory(cls, directory: str):
        """
        Load a BenchmarkResult from a directory.

        Args:
            directory (str): The directory to load the results from.

        Returns:
            BenchmarkResult: The loaded benchmark result.
        """
        with open(os.path.join(directory, "results.json"), "r") as f:
            data = json.load(f)

        # Reconstruct ScenarioGroupResult objects
        scenario_group_results = []
        for sg_data in data["scenario_group_results"]:
            scenario_results = [
                ScenarioResult(**s) for s in sg_data["scenario_results"]
            ]
            sg_result = ScenarioGroupResult(
                deployment_id=sg_data["deployment_id"],
                scenario_results=scenario_results,
                deployment_details=sg_data.get("deployment_details"),
                deployment_status=sg_data.get("deployment_status"),
            )
            scenario_group_results.append(sg_result)

        # Create the BenchmarkResult object with properly typed scenario_group_results
        result = cls(
            benchmark_id=data["benchmark_id"],
            scenario_group_results=scenario_group_results,
        )
        result.output_dir = directory
        return result


class Benchmark:
    """
    The primary mechanism for running Scenarios.

    This class can be initialized with a single or multiple scenarios, checks if deployments
    are active, and standardizes the output structure for all Benchmark runs.

    Attributes:
        output_dir (str): Directory where benchmark results will be saved.
        benchmark_id (str): Unique identifier for this benchmark run.
        benchmark_name (str): Name of the benchmark, derived from the benchmark_id.
        scenario_groups (List[ScenarioGroup]): List of scenario groups to be run.
        namespace (str): Namespace for the deployments.
    """

    def __init__(
        self,
        scenarios: Union[ScenarioGroup, List[ScenarioGroup]],
        output_dir: str = None,
    ):
        """
        Initialize a Benchmark instance.

        Args:
            scenarios (Union[ScenarioGroup, List[ScenarioGroup]]): Scenario(s) to be run.
            output_dir (str, optional): Directory to save benchmark results. Defaults to BENCHMARK_RESULTS_DIR.
        """
        self.output_dir = BENCHMARK_RESULTS_DIR if not output_dir else output_dir
        self.benchmark_id = str(uuid.uuid4())
        self.benchmark_name = f"benchmark_{self.benchmark_id}"
        self.output_dir = os.path.join(self.output_dir, self.benchmark_name)
        self.scenario_groups = self._get_scenario_groups(scenarios)
        self.namespace = self._get_namespace()

    def _get_scenario_groups(
        self,
        scenarios: Union[ScenarioGroup, List[ScenarioGroup]],
    ) -> List[ScenarioGroup]:
        """
        Convert input scenarios to a list of ScenarioGroups.

        Args:
            scenarios (Union[ScenarioGroup, List[ScenarioGroup]]): Input scenarios.

        Returns:
            List[ScenarioGroup]: List of ScenarioGroups.

        Raises:
            ValueError: If the input is not a ScenarioGroup or a list of ScenarioGroups.
        """
        if isinstance(scenarios, ScenarioGroup):
            return [scenarios]
        elif isinstance(scenarios, list):
            if all(isinstance(s, ScenarioGroup) for s in scenarios):
                return scenarios
            else:
                raise ValueError(
                    "Invalid list of scenario groups provided. Make sure the list is all of the same type."
                )

    @staticmethod
    def _parse_scenario_groups(scenarios: List[Scenario]) -> List[ScenarioGroup]:
        """
        Parse a list of Scenarios into ScenarioGroups based on their deployments.

        Args:
            scenarios (List[Scenario]): List of Scenarios to be grouped.

        Returns:
            List[ScenarioGroup]: List of ScenarioGroups.
        """
        groups = defaultdict(list)
        for scenario in scenarios:
            groups[scenario.deployment].append(scenario)

        return [ScenarioGroup(deployment=k, scenarios=v) for k, v in groups.items()]

    def _assert_existing_deployments_running(self):
        """
        Assert that all existing deployments in the scenario groups are running.

        Raises:
            Exception: If any existing deployment is not running.
        """
        for scenario_group in self.scenario_groups:
            if (
                scenario_group.deployment._exists
                and scenario_group.deployment.endpoint_status() != "running"
            ):
                raise Exception(
                    f"You initialized Deployment: {scenario_group.deployment.deployment_name} from an existing endpoint, but it is not running. Please start all _existing_ deployments before running the benchmark."
                )

    def _get_namespace(self) -> str:
        """
        Get the namespace for the deployments in the scenario groups.

        Returns:
            str: The namespace.

        Raises:
            Exception: If deployments are across multiple namespaces.
        """
        namespaces = [
            sg.deployment.deployment_config.namespace for sg in self.scenario_groups
        ]
        if len(set(namespaces)) > 1:
            raise Exception(
                "Benchmarking deployments across multiple namespaces is not supported. Please ensure all deployments are in the same namespace."
            )
        else:
            return namespaces[0]

    async def _run_scheduler_async(self) -> Scheduler:
        """
        Run the scheduler asynchronously.

        Returns:
            Scheduler: The scheduler instance after running.
        """
        self._assert_existing_deployments_running()
        scheduler = Scheduler(
            scenario_groups=self.scenario_groups,
            namespace=self.namespace,
        )
        await scheduler.run()
        return scheduler

    def run(self) -> BenchmarkResult:
        """
        Run the benchmark, compatible with both CLI and Jupyter notebooks.

        Returns:
            BenchmarkResult: The result of the benchmark run.
        """
        if "ipykernel" in sys.modules:
            # Running in Jupyter notebook
            nest_asyncio.apply()

        scheduler = asyncio.run(self._run_scheduler_async())

        benchmark_result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            scenario_group_results=scheduler.results,
            output_dir=self.output_dir,
        )
        benchmark_result.save()
        return benchmark_result
