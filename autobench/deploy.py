import requests
from typing import Dict
from dataclasses import dataclass, field

import pandas as pd
from huggingface_hub import create_inference_endpoint


@dataclass
class TGIConfig:
    model_id: str
    max_batch_prefill_tokens: int
    max_input_length: int
    max_total_tokens: int
    num_shard: int
    quantize: str
    estimated_memory_in_gigabytes: float


@dataclass
class DeploymentConfig:
    tgi_config: TGIConfig
    namespace: str
    vendor: str
    region: str
    instance_type: str
    instance_size: str
    env_vars: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.env_vars = {
            "MAX_BATCH_PREFILL_TOKENS": str(self.tgi_config.max_batch_prefill_tokens),
            "MAX_INPUT_LENGTH": str(self.tgi_config.max_input_length),
            "MAX_TOTAL_TOKENS": str(self.tgi_config.max_total_tokens),
            "NUM_SHARD": str(self.tgi_config.num_shard),
            "QUANTIZE": self.tgi_config.quantize,
        }

    @property
    def model_id(self):
        return self.tgi_config.model_id


class IEDeployment:

    def __init__(
        self,
        config: DeploymentConfig,
    ):
        self.config = config
        self.endpoint_name = f"{self.config.model_id}-autobench"

    def deploy_endpoint(self):

        try:
            endpoint = create_inference_endpoint(
                self.endpoint_name,
                repository=self.config.model_id,
                namespace=self.config.namespace,
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor=self.config.vendor,
                region=self.config.region,
                instance_size=self.config.instance_size,
                instance_type=self.config.instance_type,
                min_replica=0,
                max_replica=1,
                type="protected",
                custom_image={
                    "health_route": "/health",
                    "url": "ghcr.io/huggingface/text-generation-inference:latest",
                    "env": self.config.env_vars,
                },
            )

            endpoint.wait()
            self.endpoint = endpoint
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
