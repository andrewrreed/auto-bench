import sys
import uuid
import os
from collections import defaultdict
from typing import List, Union
import asyncio
import nest_asyncio
from autobench.scheduler import Scheduler

from autobench.runner import Scenario, ScenarioGroup


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

    def __init__(self, scenarios: Union[Scenario, List[Scenario]], output_dir: str):
        self.output_dir = output_dir
        self.namespace = "andrewrreed"
        self.benchmark_id = str(uuid.uuid4())
        self.benchmark_name = f"benchmark_{self.benchmark_id}"
        self.output_dir = os.path.join(output_dir, self.benchmark_name)

        self.scenarios = scenarios if isinstance(scenarios, list) else [scenarios]
        self.scenarios = [
            self._modify_output_dir(scenario) for scenario in self.scenarios
        ]
        self.scenario_groups = self._get_scenario_groups(self.scenarios)

    def _modify_output_dir(self, scenario: Scenario):
        scenario.output_dir = os.path.join(
            self.output_dir, scenario.deployment.deployment_name, scenario.scenario_name
        )
        return scenario

    @staticmethod
    def _get_scenario_groups(scenarios: List[Scenario]):
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

    async def _run_scheduler_async(self):
        self._assert_existing_deployments_running()
        scheduler = Scheduler(
            scenario_groups=self.scenario_groups,
            namespace=self.namespace,
            output_dir=self.output_dir,
        )
        await scheduler.run()
        return scheduler

    def run(self):
        """Run the benchmark, compatible with both CLI and Jupyter notebooks."""
        if "ipykernel" in sys.modules:
            # Running in Jupyter notebook
            nest_asyncio.apply()

        return asyncio.run(self._run_scheduler_async())
