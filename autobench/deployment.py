import uuid
from typing import Optional
from huggingface_hub import create_inference_endpoint, whoami, get_inference_endpoint
from loguru import logger

from autobench.config import DeploymentConfig


class Deployment:

    def __init__(
        self,
        deployment_config: DeploymentConfig,
        existing_endpoint_name: Optional[str] = None,
    ):
        logger.info(f"Initializing Deployment with config: {deployment_config}")
        self.deployment_config = deployment_config
        self.tgi_config = deployment_config.tgi_config
        self.instance_config = deployment_config.instance_config

        if existing_endpoint_name:
            logger.info(
                f"Attempting to set existing endpoint: {existing_endpoint_name}"
            )
            try:
                self.set_existing_endpoint(existing_endpoint_name)
            except Exception as e:
                logger.error(f"Failed to set existing endpoint: {e}")
                raise Exception(f"Endpoint {existing_endpoint_name} not found")
        else:
            self.deployment_id = str(uuid.uuid4())[
                :-4
            ]  # truncated due to IE endpoint naming restrictions
            logger.info(f"Generated new deployment ID: {self.deployment_id}")
            self.deploy_endpoint()

        self.deployment_name = "deployment_" + self.deployment_id
        logger.info(f"Deployment initialized with name: {self.deployment_name}")

    def deploy_endpoint(self):
        logger.info("Starting endpoint deployment process")
        try:
            logger.info("Creating inference endpoint...")
            endpoint = create_inference_endpoint(
                self.deployment_id,
                repository=self.tgi_config.model_id,
                namespace=whoami()["name"],
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor=self.instance_config.vendor,
                region=self.instance_config.region,
                instance_size=self.instance_config.instance_size,
                instance_type=self.instance_config.instance_type,
                min_replica=0,
                max_replica=1,
                type="protected",
                custom_image={
                    "health_route": "/health",
                    "url": "ghcr.io/huggingface/text-generation-inference:latest",
                    "env": self.tgi_config.env_vars,
                },
            )

            logger.info("Waiting for endpoint to be ready...")
            endpoint.wait()
            self.endpoint = endpoint
            logger.success(f"Endpoint created successfully: {endpoint.url}")

        except Exception as e:
            logger.error(f"Failed to create inference endpoint: {e}")
            raise

    def set_existing_endpoint(self, endpoint_id: str):
        """
        Get an existing endpoint by id.
        If the endpoint is not found, an exception is raised.
        If endpoint is not running, attempt to start it.
        """
        logger.info(f"Attempting to set existing endpoint: {endpoint_id}")
        try:
            self.endpoint = get_inference_endpoint(endpoint_id)
            logger.info(f"Endpoint found. Status: {self.endpoint.status}")

            # TO-DO: Ensure endpoint config matches deployment config
            # self.validate_existing_endpoint()

            if self.endpoint.status == "initializing":
                logger.info(f"Endpoint {endpoint_id} is initializing, waiting...")
                self.endpoint.wait()
                logger.success(f"Endpoint {endpoint_id} is now running")
            elif self.endpoint.status != "running":
                logger.warning(
                    f"Endpoint {endpoint_id} is not running, attempting to start"
                )
                self.endpoint.resume().wait()
                logger.success(f"Endpoint {endpoint_id} is now running")

            self.deployment_id = endpoint_id
            logger.info(f"Successfully set existing endpoint: {endpoint_id}")

        except Exception as e:
            logger.error(f"Failed to set existing endpoint: {e}")
            raise Exception(f"Endpoint {endpoint_id} not found")
