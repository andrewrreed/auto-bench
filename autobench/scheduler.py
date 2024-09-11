import os
import uuid
import asyncio
from typing import List, Dict
from loguru import logger
import json
from huggingface_hub.errors import InferenceEndpointError
from huggingface_hub import HfApi
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from autobench.config import DataConfig, DeploymentConfig
from autobench.deployment import Deployment
from autobench.runner import BenchmarkRunner
from autobench.data import BenchmarkDataset

from huggingface_hub.constants import INFERENCE_ENDPOINTS_ENDPOINT
from huggingface_hub.utils import get_session, hf_raise_for_status, build_hf_headers


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
        self.deployment_statuses = []
        logger.info(
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
        logger.info(f"Fetching quotas for namespace: {self.namespace}")
        session = get_session()
        response = session.get(
            f"{INFERENCE_ENDPOINTS_ENDPOINT}/provider/quotas/{self.namespace}",
            headers=build_hf_headers(),
        )
        hf_raise_for_status(response)
        logger.success(f"Quotas fetched successfully for namespace: {self.namespace}")
        return response.json()

    async def run(self):
        logger.info("Starting scheduler run")
        await self.update_quota()
        await self.initialize_tasks()
        await self.process_tasks()
        self.save_deployment_statuses()

    async def update_quota(self):
        logger.info("Updating quota information")
        self.quota = await asyncio.to_thread(self.fetch_quotas)
        logger.debug(f"Updated quota: {self.quota}")

    async def initialize_tasks(self):
        logger.info(f"Initializing tasks for {len(self.viable_instances)} instances")
        for instance in self.viable_instances:
            await self.pending_tasks.put(instance)
        logger.debug(f"Initialized {self.pending_tasks.qsize()} pending tasks")

    async def process_tasks(self):
        logger.info("Starting to process tasks")
        while not self.pending_tasks.empty() or self.running_tasks:
            pending_instances = []
            while not self.pending_tasks.empty():
                instance = await self.pending_tasks.get()
                if self.can_deploy(instance):
                    task = asyncio.create_task(self.deploy_and_benchmark(instance))
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)
                    logger.info(
                        f"Deployed task for instance: {instance['instance_config'].id}"
                    )
                else:
                    pending_instances.append(instance)
                    logger.warning(
                        f"Unable to deploy instance: {instance['instance_config'].id}"
                    )

            # Put back the instances that couldn't be deployed
            for instance in pending_instances:
                await self.pending_tasks.put(instance)

            logger.info(
                f"Current state: {self.pending_tasks.qsize()} pending tasks, {len(self.running_tasks)} running tasks"
            )
            await asyncio.sleep(10)  # Wait before checking again
            await self.update_quota()
            logger.debug("Sleeping for 10 seconds before next check")

    def can_deploy(self, instance):
        instance_id = instance["instance_config"].id
        instance_type = instance["instance_config"].instance_type
        vendor = instance["instance_config"].vendor
        num_gpus_required = instance["instance_config"].num_gpus
        logger.info(
            f"Checking if can deploy: {vendor} {instance_id} (requires {num_gpus_required} GPUs)"
        )

        for vendor_data in self.quota["vendors"]:
            if vendor_data["name"] == vendor:
                for quota in vendor_data["quotas"]:
                    if quota["instanceType"] == instance_type:
                        available_gpus = (
                            quota["maxAccelerators"] - quota["usedAccelerators"]
                        )
                        can_deploy = available_gpus >= num_gpus_required
                        logger.info(
                            f"Deployment possible for {instance_id}: {can_deploy} (Available GPUs: {available_gpus})"
                        )
                        return can_deploy
        logger.warning(f"No matching quota found for {vendor} {instance_id}")
        return False

    async def deploy_and_benchmark(self, instance):
        logger.info(
            f"Deploying and benchmarking instance: {instance['instance_config'].id}"
        )
        deployment_status = {
            "instance_id": instance["instance_config"].id,
            "instance_type": instance["instance_config"].instance_type,
            "status": "failed",
            "error": None,
        }
        deployment = None

        try:
            # Create Deployment object (this only sets the ID, doesn't deploy)
            deployment = await asyncio.to_thread(
                Deployment,
                DeploymentConfig(
                    tgi_config=instance["tgi_config"],
                    instance_config=instance["instance_config"],
                ),
            )
            deployment_status["deployment_id"] = deployment.deployment_id
            deployment_status["deployment_name"] = deployment.deployment_name

            # Now deploy the endpoint
            await asyncio.to_thread(deployment.deploy_endpoint)

            runner = BenchmarkRunner(
                deployment=deployment,
                benchmark_dataset=self.benchmark_dataset,
                output_dir=self.output_dir,
            )

            benchmark_results = await asyncio.to_thread(runner.run_benchmark)
            deployment_status["status"] = "success"
            logger.success(
                f"Benchmark completed for instance: {instance['instance_config'].id}"
            )

        except InferenceEndpointError as e:
            logger.error(
                f"Deployment failed for instance {instance['instance_config'].id}: {str(e)}"
            )
            deployment_status["error"] = str(e)

        except Exception as e:
            logger.error(
                f"Error during benchmark for instance {instance['instance_config'].id}: {str(e)}"
            )
            deployment_status["error"] = str(e)

        finally:
            if deployment:
                logger.info(
                    f"Attempting to delete deployment with ID: {deployment.deployment_id}"
                )
                try:
                    # TODO: Ideally, if endpoint has failed, retrieve container logs somehow and save them to the deployment_status before deleting
                    await asyncio.sleep(5)
                    await asyncio.to_thread(
                        delete_inference_endpoint,
                        deployment.deployment_id,
                        self.namespace,
                    )
                    logger.success(
                        f"Deployment deleted for instance: {instance['instance_config'].id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error deleting deployment for instance {instance['instance_config'].id}: {str(e)}"
                    )
                    deployment_status["error"] = (
                        f"{deployment_status['error']}; Error deleting: {str(e)}"
                        if deployment_status["error"]
                        else f"Error deleting: {str(e)}"
                    )
            else:
                logger.warning(
                    f"No deployment object created for instance: {instance['instance_config'].id}"
                )

            await self.update_quota()
            self.deployment_statuses.append(deployment_status)

    def save_deployment_statuses(self):
        results_file = os.path.join(self.output_dir, "deployment_statuses.json")
        with open(results_file, "w") as f:
            json.dump(self.deployment_statuses, f, indent=2)
        logger.info(f"Deployment results saved to {results_file}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def delete_inference_endpoint(endpoint_id: str, namespace: str):
    api = HfApi()
    try:
        api.delete_inference_endpoint(endpoint_id, namespace=namespace)
        logger.info(f"Successfully deleted endpoint {endpoint_id}")
    except Exception as e:
        logger.error(f"Failed to delete endpoint {endpoint_id}: {str(e)}")
        raise


async def run_scheduler_async(
    viable_instances: List[Dict], namespace: str, output_dir: str
):
    logger.info(
        f"Starting async scheduler with {len(viable_instances)} instances for namespace: {namespace}"
    )
    scheduler = Scheduler(viable_instances, namespace, output_dir)
    await scheduler.run()
    return scheduler


def run_scheduler(viable_instances, namespace, output_dir):
    logger.info(
        f"Running scheduler for {len(viable_instances)} instances in namespace: {namespace}"
    )
    return asyncio.run(run_scheduler_async(viable_instances, namespace, output_dir))
