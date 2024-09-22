import os
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

from autobench.runner import Scenario
from autobench.deployment import Deployment

from huggingface_hub.constants import INFERENCE_ENDPOINTS_ENDPOINT
from huggingface_hub.utils import get_session, hf_raise_for_status, build_hf_headers


class Scheduler:
    """
    Instead of taking in `viable_instances`, we take in mapping of {deployment: scenarios}.
    The scheduler then:
        - schedules each deployment
        - deploys if not already running
        - executes all the scenarios tied to it

    """

    def __init__(
        self,
        scenario_groups: Dict[Deployment, List[Scenario]],
        namespace: str,
        output_dir: str,
    ):
        self.scenario_groups = scenario_groups
        self.namespace = namespace
        self.quota = None
        self.running_tasks = set()
        self.pending_tasks = asyncio.Queue()
        self.scenerio_group_statuses = []
        self.output_dir = output_dir

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
        logger.info(f"Initializing tasks for {len(self.scenario_groups)} deployments")
        for scenario_group in self.scenario_groups:
            await self.pending_tasks.put(scenario_group)
        logger.debug(f"Initialized {self.pending_tasks.qsize()} pending tasks")

    async def process_tasks(self):
        logger.info("Starting to process tasks")

        while not self.pending_tasks.empty() or self.running_tasks:

            pending_scenario_groups = []

            while not self.pending_tasks.empty():
                scenario_group = await self.pending_tasks.get()

                if self._endpoint_exists(
                    scenario_group.deployment
                ) and self._is_running(scenario_group.deployment):
                    logger.info(
                        f"Endpoint exists and is running for deployment: {scenario_group.deployment.deployment_id}"
                    )
                    task = asyncio.create_task(
                        self.deploy_and_benchmark(scenario_group)
                    )
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)

                elif self._can_deploy(scenario_group.deployment):
                    logger.info(
                        f"Quota available to to run deployment: {scenario_group.deployment.deployment_id}"
                    )
                    task = asyncio.create_task(
                        self.deploy_and_benchmark(scenario_group)
                    )
                    self.running_tasks.add(task)
                    task.add_done_callback(self.running_tasks.discard)
                else:
                    # If endpoint doesn't exist and can't be deployed, add to pending
                    pending_scenario_groups.append(scenario_group)

            # Put back the instances that couldn't be deployed
            for scenario_group in pending_scenario_groups:
                await self.pending_tasks.put(scenario_group)

            logger.info(
                f"Current state: {self.pending_tasks.qsize()} pending tasks, {len(self.running_tasks)} running tasks"
            )
            await asyncio.sleep(10)  # Wait before checking again
            await self.update_quota()
            logger.debug("Sleeping for 10 seconds before next check")

    @staticmethod
    def _endpoint_exists(deployment):
        return deployment._exists

    @staticmethod
    def _is_running(deployment):
        return deployment.endpoint_status() == "running"

    def _can_deploy(self, deployment):
        instance_id = deployment.instance_config.id
        instance_type = deployment.instance_config.instance_type
        vendor = deployment.instance_config.vendor
        num_gpus_required = deployment.instance_config.num_gpus
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

    async def deploy_and_benchmark(self, scenario_group):

        scenerio_group_status = {
            "instance_id": scenario_group.deployment.instance_config.id,
            "instance_type": scenario_group.deployment.instance_config.instance_type,
            "status": "failed",
            "error": None,
        }

        try:
            # deploy if needed
            if not self._is_running(scenario_group.deployment):
                logger.info(
                    f"Deploying endpoint for instance: {scenario_group.deployment.deployment_id}"
                )
                await asyncio.to_thread(scenario_group.deployment.deploy_endpoint)
            else:
                logger.info(
                    f"Endpoint exists for instance: {scenario_group.deployment.deployment_id}"
                )

            # run scenario group
            benchmark_results = await asyncio.to_thread(scenario_group._run)
            scenerio_group_status["status"] = "success"
            logger.success(
                f"Benchmark completed for scenerio group on instance: {scenario_group.deployment.instance_config.id}"
            )

        except InferenceEndpointError as e:
            logger.error(
                f"Deployment failed for scenerio group instance {scenario_group.deployment.instance_config.id}: {str(e)}"
            )
            scenerio_group_status["error"] = str(e)

        except Exception as e:
            logger.error(
                f"Error during scenerio group benchmark for instance {scenario_group.deployment.instance_config.id}: {str(e)}"
            )
            scenerio_group_status["error"] = str(e)

        finally:
            if self._is_running(scenario_group.deployment):
                logger.info(
                    f"Attempting to delete deployment with ID: {scenario_group.deployment.deployment_id}"
                )
                try:
                    # TODO: Ideally, if endpoint has failed, retrieve container logs somehow and save them to the deployment_status before deleting
                    # await asyncio.sleep(5)
                    # await asyncio.to_thread(
                    #     delete_inference_endpoint,
                    #     scenario_group.deployment.deployment_id,
                    #     self.namespace,
                    # )
                    logger.success(
                        f"Deployment deleted for instance: {scenario_group.deployment.instance_config.id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error deleting deployment for instance {scenario_group.deployment.instance_config.id}: {str(e)}"
                    )
                    scenerio_group_status["error"] = (
                        f"{scenerio_group_status['error']}; Error deleting: {str(e)}"
                        if scenerio_group_status["error"]
                        else f"Error deleting: {str(e)}"
                    )
            else:
                logger.warning(
                    f"No deployment object created for instance: {scenario_group.deployment.instance_config.id}"
                )

            await self.update_quota()
            self.scenerio_group_statuses.append(scenerio_group_status)

    def save_deployment_statuses(self):
        results_file = os.path.join(self.output_dir, "deployment_statuses.json")
        with open(results_file, "w") as f:
            json.dump(self.scenerio_group_statuses, f, indent=2)
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
