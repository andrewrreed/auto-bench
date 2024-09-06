import sys
import os
import uuid
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from autobench.config import DataConfig, DeploymentConfig
from autobench.deployment import Deployment
from autobench.runner import BenchmarkRunner
from autobench.data import BenchmarkDataset

from huggingface_hub.constants import INFERENCE_ENDPOINTS_ENDPOINT
from huggingface_hub.utils import get_session, hf_raise_for_status, build_hf_headers

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")


class Scheduler:
    def __init__(self, viable_instances: List[Dict], namespace: str, output_dir: str):
        self.viable_instances = viable_instances
        self.namespace = namespace
        self.quota = None
        self.running_tasks = set()
        self.pending_tasks = asyncio.Queue()
        self.benchmark_dataset = BenchmarkDataset(data_config=DataConfig())
        self.scheduler_id = str(uuid.uuid4())
        self.scheduler_name = f"scheduler_{self.scheduler_id}"
        self.output_dir = os.path.join(output_dir, self.scheduler_name)
        print(
            f"Scheduler initialized with {len(viable_instances)} viable instances for namespace: {namespace}"
        )

    def fetch_quotas(self):
        """
        Fetch quotas for a given namespace.

        Args:
            namespace (str): The namespace to fetch quotas for.

        Returns:
            dict: The quotas for the given namespace.
        """
        print(f"Fetching quotas for namespace: {self.namespace}")
        session = get_session()
        response = session.get(
            f"{INFERENCE_ENDPOINTS_ENDPOINT}/provider/quotas/{self.namespace}",
            headers=build_hf_headers(),
        )
        hf_raise_for_status(response)
        print(f"Quotas fetched successfully")
        return response.json()

    async def run(self):
        print("Starting scheduler run")
        await self.update_quota()
        await self.initialize_tasks()
        await self.process_tasks()

    async def update_quota(self):
        print("Updating quota information")
        self.quota = await asyncio.to_thread(self.fetch_quotas)

    async def initialize_tasks(self):
        print(f"Initializing tasks for {len(self.viable_instances)} instances")
        for instance in self.viable_instances:
            await self.pending_tasks.put(instance)

    async def process_tasks(self):
        print("Starting to process tasks")
        while not self.pending_tasks.empty() or self.running_tasks:
            pending_instances = []
            while not self.pending_tasks.empty():
                instance = await self.pending_tasks.get()
                if self.can_deploy(instance):
                    task = asyncio.create_task(self.deploy_and_benchmark(instance))
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)
                else:
                    pending_instances.append(instance)

            # Put back the instances that couldn't be deployed
            for instance in pending_instances:
                await self.pending_tasks.put(instance)

            print(
                f"Current state: {self.pending_tasks.qsize()} pending tasks, {len(self.running_tasks)} running tasks"
            )
            await asyncio.sleep(10)  # Wait before checking again
            await self.update_quota()
            print(f"Sleeping for 10 seconds before next check")

    def can_deploy(self, instance):
        instance_type = instance["instance_config"].instance_type
        vendor = instance["instance_config"].vendor
        num_gpus_required = instance["instance_config"].num_gpus
        print(
            f"Checking if can deploy: {vendor} {instance_type} (requires {num_gpus_required} GPUs)"
        )

        for vendor_data in self.quota["vendors"]:
            if vendor_data["name"] == vendor:
                for quota in vendor_data["quotas"]:
                    if quota["instanceType"] == instance_type:
                        available_gpus = (
                            quota["maxAccelerators"] - quota["usedAccelerators"]
                        )
                        print(
                            f"Deployment possible: {available_gpus >= num_gpus_required}"
                        )
                        return available_gpus >= num_gpus_required
        print(f"Deployment possible: {False}")
        return False

    async def deploy_and_benchmark(self, instance):
        print(
            f"Deploying and benchmarking instance: {instance['instance_config'].instance_type}"
        )
        deployment = await asyncio.to_thread(
            Deployment,
            DeploymentConfig(
                tgi_config=instance["tgi_config"],
                instance_config=instance["instance_config"],
            ),
        )
        runner = BenchmarkRunner(
            deployment=deployment,
            benchmark_dataset=self.benchmark_dataset,
            output_dir=self.output_dir,
        )

        await asyncio.to_thread(runner.run_benchmark)
        await asyncio.to_thread(deployment.endpoint.delete)
        await self.update_quota()
        print(
            f"Benchmark completed for instance: {instance['instance_config'].instance_type}"
        )
        print(
            f"Deleting deployment for instance: {instance['instance_config'].instance_type}"
        )
        print(
            f"Deployment deleted for instance: {instance['instance_config'].instance_type}"
        )


async def run_scheduler_async(viable_instances: List[Dict], namespace: str):
    print(
        f"Starting async scheduler with {len(viable_instances)} instances for namespace: {namespace}"
    )
    scheduler = Scheduler(viable_instances, namespace, RESULTS_DIR)
    await scheduler.run()


def run_scheduler(viable_instances, namespace):
    print(
        f"Running scheduler for {len(viable_instances)} instances in namespace: {namespace}"
    )
    if "ipykernel" in sys.modules:
        # We're in a Jupyter notebook
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            future = asyncio.run_coroutine_threadsafe(
                run_scheduler_async(viable_instances, namespace), loop
            )
            return future.result()
    else:
        # We're in a regular Python environment
        return asyncio.run(run_scheduler_async(viable_instances, namespace))
