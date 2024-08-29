from huggingface_hub import create_inference_endpoint, whoami, get_inference_endpoint
from typing import Optional
from autobench.config import DeploymentConfig
import time


class Deployment:

    def __init__(
        self,
        deployment_config: DeploymentConfig,
        existing_endpoint_name: Optional[str] = None,
    ):
        self.deployment_config = deployment_config
        self.tgi_config = deployment_config.tgi_config
        self.instance_config = deployment_config.instance_config

        if existing_endpoint_name:
            try:
                self.set_existing_endpoint(existing_endpoint_name)
            except Exception as e:
                print(e)
                raise Exception(f"Endpoint {existing_endpoint_name} not found")
        else:
            self.endpoint_name = deployment_config.deployment_id
            self.deploy_endpoint()

    def deploy_endpoint(self):

        try:
            print("Creating inference endpoint...")
            endpoint = create_inference_endpoint(
                self.endpoint_name,
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

            endpoint.wait()
            self.endpoint = endpoint
            print(f"Endpoint created successfully: {endpoint.url}")

        except Exception as e:
            print(e)

    def set_existing_endpoint(self, endpoint_id: str):
        """
        Get an existing endpoint by id.
        If the endpoint is not found, an exception is raised.
        If endpoint is not running, attempt to start it.
        """
        try:
            print(f"Getting existing endpoint {endpoint_id}")
            self.endpoint = get_inference_endpoint(endpoint_id)

            print(f"Endpoint found.\nEndpoint status: {self.endpoint.status}")

            # TO-DO: Ensure endpoint config matches deployment config
            # self.validate_existing_endpoint()

            if self.endpoint.status == "initializing":
                print(f"Endpoint {endpoint_id} is initializing, waiting...")
                self.endpoint.wait()
                print(f"Endpoint {endpoint_id} is now running")

            elif self.endpoint.status != "running":
                print(f"Endpoint {endpoint_id} is not running, attempting to start")
                self.endpoint.resume().wait()
                print(f"Endpoint {endpoint_id} is now running")

            self.endpoint_name = endpoint_id
            self.deployment_config.deployment_id = endpoint_id

        except Exception as e:
            print(e)
            raise Exception(f"Endpoint {endpoint_id} not found")
