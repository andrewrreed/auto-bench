import json
import asyncio
from typing import List
from loguru import logger
from dataclasses import asdict
from huggingface_hub.errors import InferenceEndpointError
from huggingface_hub import HfApi
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from autobench.scenario import ScenarioGroup, ScenarioGroupResult
from huggingface_hub.constants import INFERENCE_ENDPOINTS_ENDPOINT
from huggingface_hub.utils import get_session, build_hf_headers


class Scheduler:
    """
    A scheduler for managing and running scenario groups on Hugging Face inference endpoints.

    This class handles the deployment, benchmarking, and cleanup of scenario groups
    across multiple inference endpoints, managing quotas and concurrent tasks.

    Attributes:
        scenario_groups (List[ScenarioGroup]): List of scenario groups to be scheduled and run.
        namespace (str): The namespace for the Hugging Face inference endpoints.
        quota (dict): The current quota information for the namespace.
        running_tasks (set): Set of currently running asyncio tasks.
        pending_tasks (asyncio.Queue): Queue of pending scenario groups to be processed.
        results (list): List to store the results of completed scenario group runs.
    """

    def __init__(
        self,
        scenario_groups: List[ScenarioGroup],
        namespace: str,
    ):
        """
        Initialize the Scheduler.

        Args:
            scenario_groups (List[ScenarioGroup]): List of scenario groups to be scheduled and run.
            namespace (str): The namespace for the Hugging Face inference endpoints.
        """
        self.scenario_groups = scenario_groups
        self.namespace = namespace
        self.quota = None
        self.running_tasks = set()
        self.pending_tasks = asyncio.Queue()
        self.results = []

    def fetch_quotas(self):
        """
        Fetch quotas for the current namespace.

        Returns:
            dict: The quotas for the given namespace.
        """
        session = get_session()
        response = session.get(
            f"{INFERENCE_ENDPOINTS_ENDPOINT}/provider/quotas/{self.namespace}",
            headers=build_hf_headers(),
        )
        return response.json()

    async def run(self):
        """
        Run the scheduler, processing all scenario groups.

        This method initializes the quota, sets up tasks, and processes them until completion.
        """
        logger.success("Starting scheduler run")
        await self.update_quota()
        await self.initialize_tasks()
        await self.process_tasks()

        logger.success(
            "Benchmark run completed successfully! Remember to check your Hugging Face Inference Endpoints UI to ensure all resources have been paused or torn down as expected."
        )

    async def update_quota(self):
        """Update the current quota information."""
        self.quota = await asyncio.to_thread(self.fetch_quotas)

    async def initialize_tasks(self):
        """Initialize tasks by adding all scenario groups to the pending tasks queue."""
        logger.info(f"Initializing tasks for {len(self.scenario_groups)} deployments")
        for scenario_group in self.scenario_groups:
            await self.pending_tasks.put(scenario_group)

    async def process_tasks(self):
        """
        Process pending tasks, managing deployments and benchmarks.

        This method continuously checks for available resources and deploys scenario groups
        when possible, updating quotas and managing running tasks.
        """
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

    @staticmethod
    def _endpoint_exists(deployment):
        """
        Check if an endpoint exists for the given deployment.

        Args:
            deployment: The deployment object to check.

        Returns:
            bool: True if the endpoint exists, False otherwise.
        """
        return deployment._exists

    @staticmethod
    def _is_running(deployment):
        """
        Check if the endpoint for the given deployment is running.

        Args:
            deployment: The deployment object to check.

        Returns:
            bool: True if the endpoint is running, False otherwise.
        """
        return deployment.endpoint_status() == "running"

    def _can_deploy(self, deployment):
        """
        Check if there are enough resources available to deploy the given deployment.

        Args:
            deployment: The deployment object to check.

        Returns:
            bool: True if the deployment can be made, False otherwise.
        """
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
        """
        Deploy an endpoint for the given scenario group and run the benchmark.

        This method handles the deployment, benchmarking, and cleanup process for a single scenario group.

        Args:
            scenario_group (ScenarioGroup): The scenario group to deploy and benchmark.
        """

        scenerio_group_status = {"status": "failed", "error": None, "oom": False}

        try:
            # deploy if needed
            if not self._endpoint_exists(scenario_group.deployment):
                logger.info(
                    f"Creating endpoint for instance: {scenario_group.deployment.deployment_id}"
                )
                await asyncio.to_thread(scenario_group.deployment.deploy_endpoint)

            elif not self._is_running(scenario_group.deployment):
                logger.info(
                    f"Resuming endpoint for instance: {scenario_group.deployment.deployment_id}"
                )
                await asyncio.to_thread(scenario_group.deployment.resume_endpoint)
            else:
                logger.info(
                    f"Endpoint exists and is already running for instance: {scenario_group.deployment.deployment_id}"
                )

            # run scenario group
            scenario_group_result = await asyncio.to_thread(scenario_group._run)
            scenerio_group_status["status"] = "success"
            logger.success(
                f"Benchmark completed for scenerio group on instance: {scenario_group.deployment.instance_config.id}"
            )

        except InferenceEndpointError as e:
            logger.error(
                f"Deployment failed for scenerio group instance {scenario_group.deployment.instance_config.id}: {str(e)}"
            )
            try:
                logger.info("Attempting to gather logs from failed endpoint.")
                await asyncio.sleep(60)
                logs = await asyncio.to_thread(
                    get_endpoint_logs,
                    self.namespace,
                    scenario_group.deployment.deployment_id,
                )
                scenerio_group_status["oom"] = "OutOfMemoryError" in logs

            except Exception as e:
                logger.error(
                    f"Error fetching logs for instance {scenario_group.deployment.instance_config.id}: {str(e)}"
                )

            scenerio_group_status["error"] = str(e)

        except Exception as e:
            logger.error(
                f"Error during scenerio group benchmark for instance {scenario_group.deployment.instance_config.id}: {str(e)}"
            )
            scenerio_group_status["error"] = str(e)

        finally:
            if self._is_running(scenario_group.deployment):
                try:
                    if scenario_group.deployment.teardown_on_exit:
                        logger.info(
                            f"Attempting to delete deployment with ID: {scenario_group.deployment.deployment_id}"
                        )
                        await asyncio.sleep(5)
                        await asyncio.to_thread(
                            delete_inference_endpoint,
                            scenario_group.deployment.deployment_id,
                            self.namespace,
                        )
                        logger.success(
                            f"Endpoint deleted for deployment: {scenario_group.deployment.deployment_id}"
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
            # save results
            if "scenario_group_result" not in locals():
                scenario_group_result = ScenarioGroupResult(
                    deployment_id=scenario_group.deployment.deployment_id,
                    scenario_results=[],
                    deployment_details={
                        "tgi_config": asdict(scenario_group.deployment.tgi_config),
                        "instance_config": asdict(
                            scenario_group.deployment.instance_config
                        ),
                        "endpoint_details": None,
                    },
                )

            scenario_group_result.deployment_status = scenerio_group_status
            self.results.append(scenario_group_result)

            await self.update_quota()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def delete_inference_endpoint(endpoint_id: str, namespace: str):
    """
    Delete an inference endpoint with retry logic.

    This function attempts to delete the specified inference endpoint up to 3 times,
    with exponential backoff between attempts.

    Args:
        endpoint_id (str): The ID of the endpoint to delete.
        namespace (str): The namespace of the endpoint.

    Raises:
        Exception: If the deletion fails after all retry attempts.
    """
    api = HfApi()
    try:
        api.delete_inference_endpoint(endpoint_id, namespace=namespace)
        logger.info(f"Successfully deleted endpoint {endpoint_id}")
    except Exception as e:
        logger.error(f"Failed to delete endpoint {endpoint_id}: {str(e)}")
        raise


def get_endpoint_logs(namespace: str, endpoint_name: str):
    """
    Fetch logs for a given endpoint.

    This function retrieves the logs for the specified endpoint and attempts to parse them as JSON if possible.

    Args:
        namespace (str): The namespace of the endpoint.
        endpoint_name (str): The name of the endpoint.

    Returns:
        str or dict: The logs as plain text or parsed JSON if available.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request to fetch logs fails.
    """
    session = get_session()
    response = session.get(
        f"{INFERENCE_ENDPOINTS_ENDPOINT}/endpoint/{namespace}/{endpoint_name}/logs",
        headers=build_hf_headers(),
    )
    response.raise_for_status()  # Raise an exception for HTTP errors

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except json.JSONDecodeError:
            print("Warning: Content-Type is JSON but couldn't decode JSON content")
            return response.text
    else:
        return response.text
