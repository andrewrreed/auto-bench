import uuid
from typing import Optional

from huggingface_hub import (
    create_inference_endpoint,
    get_inference_endpoint,
    HfApi,
)
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from autobench.config import DeploymentConfig, TGIConfig, ComputeInstanceConfig


class Deployment:

    def __init__(
        self,
        deployment_config: DeploymentConfig,
        teardown_on_exit: Optional[bool] = True,
    ):
        self.deployment_config = deployment_config
        self.tgi_config = deployment_config.tgi_config
        self.instance_config = deployment_config.instance_config
        self._exists = False

        if not getattr(self, "_from_factory", False):
            self.deployment_id = str(uuid.uuid4())[
                :-4
            ]  # truncated due to IE endpoint naming restrictions
            self.teardown_on_exit = teardown_on_exit
        else:
            self._exists = True

        # self.deployment_name = "deployment_" + self.deployment_id
        logger.info(f"Deployment initialized with name: {self.deployment_id}")

    @classmethod
    def from_existing_endpoint(
        cls,
        endpoint_name: str,
        namespace: str = None,  # if None, assume user namespace
        teardown_on_exit: Optional[bool] = False,
    ):
        logger.info(f"Creating Deployment from existing endpoint: {endpoint_name}")

        try:
            user_info = HfApi().whoami()
            namespace = user_info["name"] if namespace is None else namespace

            endpoint = get_inference_endpoint(endpoint_name, namespace=namespace)

            if endpoint.status == "initializing":
                logger.info(f"Endpoint {endpoint_name} is initializing, waiting...")
                endpoint.wait()
                logger.success(f"Endpoint {endpoint_name} is now running")
            elif endpoint.status != "running":
                logger.warning(
                    f"Endpoint {endpoint_name} is not running, attempting to start"
                )
                endpoint.resume().wait()
                logger.success(f"Endpoint {endpoint_name} is now running")

        except HfHubHTTPError as e:
            logger.error(
                f"Make sure you initialize the deployment with the correct endpoint name and associated namespace. Error: {e}"
            )
            raise Exception(f"Endpoint {endpoint_name} not found")
        except Exception as e:
            logger.error(f"Failed to get existing endpoint: {e}")
            raise Exception(f"Endpoint {endpoint_name} not found")

        deployment_config = cls._create_config_from_endpoint(endpoint)
        instance = cls.__new__(cls)
        instance._from_factory = True
        instance.endpoint = endpoint
        instance.deployment_id = endpoint_name
        instance.teardown_on_exit = teardown_on_exit
        instance.__init__(deployment_config)

        return instance

    @staticmethod
    def _create_config_from_endpoint(endpoint) -> DeploymentConfig:

        endpoint_info = endpoint.raw
        namespace = endpoint.namespace

        tgi_config = TGIConfig(
            model_id=endpoint_info["model"].get("repository", None),
            max_batch_prefill_tokens=endpoint_info["model"]["image"]["custom"][
                "env"
            ].get("MAX_BATCH_PREFILL_TOKENS", None),
            max_input_length=endpoint_info["model"]["image"]["custom"]["env"].get(
                "MAX_INPUT_LENGTH", None
            ),
            max_total_tokens=endpoint_info["model"]["image"]["custom"]["env"].get(
                "MAX_TOTAL_TOKENS", None
            ),
            num_shard=endpoint_info["model"]["image"]["custom"]["env"].get(
                "NUM_SHARD", 1
            ),
            quantize=endpoint_info["model"]["image"]["custom"]["env"].get(
                "QUANTIZE", None
            ),
        )

        compute_instance_config = ComputeInstanceConfig(
            id=endpoint_info["compute"].get("id", None),
            vendor=endpoint_info["provider"].get("vendor", None),
            region=endpoint_info["provider"].get("region", None),
            accelerator=endpoint_info["compute"].get("accelerator", None),
            instance_type=endpoint_info["compute"].get("instanceType", None),
            instance_size=endpoint_info["compute"].get("instanceSize", None),
            num_gpus=int(
                endpoint_info["compute"].get("instanceSize", None)[-1]
            ),  # need to fix this to handle None case
        )

        return DeploymentConfig(tgi_config, compute_instance_config, namespace)

    def deploy_endpoint(self):
        logger.info("Starting endpoint deployment process")
        try:
            logger.info("Creating inference endpoint...")
            endpoint = create_inference_endpoint(
                self.deployment_id,
                repository=self.tgi_config.model_id,
                namespace=self.deployment_config.namespace,
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor=self.instance_config.vendor,
                region=self.instance_config.region,
                instance_size=self.instance_config.instance_size,
                instance_type=self.instance_config.instance_type,
                min_replica=0,
                max_replica=1,
                scale_to_zero_timeout=30,
                type="protected",
                custom_image={
                    "health_route": "/health",
                    "url": "ghcr.io/huggingface/text-generation-inference:2.3.0",
                    "env": self.tgi_config.env_vars,
                },
            )

            logger.info("Waiting for endpoint to be ready...")
            endpoint.wait()
            self.endpoint = endpoint
            self._exists = True
            logger.success(f"Endpoint created successfully: {endpoint.url}")

        except Exception as e:
            logger.error(f"Failed to create inference endpoint: {e}")
            raise

    def resume_endpoint(self):
        self.endpoint.resume().wait()

    def endpoint_status(self):
        if hasattr(self, "endpoint"):
            deployment = self.endpoint.fetch()
            return deployment.status
        else:
            logger.error("Endpoint doesn't exist for this deployment.")
            return None
