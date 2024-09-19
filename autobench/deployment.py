import uuid
from huggingface_hub import create_inference_endpoint, whoami, get_inference_endpoint
from loguru import logger

from autobench.config import DeploymentConfig, TGIConfig, ComputeInstanceConfig


class Deployment:

    def __init__(
        self,
        deployment_config: DeploymentConfig,
    ):
        self.deployment_config = deployment_config
        self.tgi_config = deployment_config.tgi_config
        self.instance_config = deployment_config.instance_config

        if not getattr(self, "_from_factory", False):
            self.deployment_id = str(uuid.uuid4())[
                :-4
            ]  # truncated due to IE endpoint naming restrictions

        self.deployment_name = "deployment_" + self.deployment_id
        logger.info(f"Deployment initialized with name: {self.deployment_name}")

    @classmethod
    def from_existing_endpoint(cls, endpoint_name: str):
        logger.info(f"Creating Deployment from existing endpoint: {endpoint_name}")

        try:
            endpoint = get_inference_endpoint(endpoint_name)

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

        except Exception as e:
            logger.error(f"Failed to get existing endpoint: {e}")
            raise Exception(f"Endpoint {endpoint_name} not found")

        deployment_config = cls._create_config_from_endpoint(endpoint)
        instance = cls.__new__(cls)
        instance._from_factory = True
        instance.endpoint = endpoint
        instance.deployment_id = endpoint_name
        instance.__init__(deployment_config)

        return instance

    @staticmethod
    def _create_config_from_endpoint(endpoint) -> DeploymentConfig:

        endpoint_info = endpoint.raw

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
        )

        return DeploymentConfig(tgi_config, compute_instance_config)

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
                scale_to_zero_timeout=30,
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
