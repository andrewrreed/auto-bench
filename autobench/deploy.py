import pandas as pd
from huggingface_hub import create_inference_endpoint, whoami

from autobench.utils import TGIConfig, ComputeInstanceConfig


class IEDeployment:

    def __init__(
        self,
        tgi_config: TGIConfig,
        instance_config: ComputeInstanceConfig,
    ):
        self.tgi_config = tgi_config
        self.instance_config = instance_config
        # self.endpoint_name = (
        #     f"{self.tgi_config.model_id.split('/')[1].lower()}-autobench"
        # )
        self.endpoint_name = "autobench"

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

    def pause_endpoint(self):
        self.endpoint.pause()

    def resume_endpoint(self):
        self.endpoint.resume()

    def delete_endpoint(self):
        self.endpoint.delete()

    def status(self):
        return self.endpoint.status()
