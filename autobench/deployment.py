import uuid
from typing import Optional

from huggingface_hub import (
    create_inference_endpoint,
    get_inference_endpoint,
    HfApi,
)
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from autobench.config import DeploymentConfig, DockerConfig, ComputeInstanceConfig


class Deployment:
    """
    A class to manage deployment of inference endpoints.

    This class handles the creation, management, and teardown of inference endpoints
    using the Hugging Face Inference API.

    Attributes:
        deployment_config (DeploymentConfig): Configuration for the deployment.
        docker_config (DockerConfig): Configuration for the Inference Endpoint Docker Image
        instance_config (ComputeInstanceConfig): Configuration for the compute instance.
        deployment_id (str): Unique identifier for the deployment.
        teardown_on_exit (bool): Whether to tear down the endpoint on exit.
        _exists (bool): Whether the deployment already exists.
    """

    def __init__(
        self,
        deployment_config: DeploymentConfig,
        teardown_on_exit: Optional[bool] = True,
    ):
        """
        Initialize a new Deployment instance.

        Args:
            deployment_config (DeploymentConfig): Configuration for the deployment.
            teardown_on_exit (Optional[bool]): Whether to tear down the endpoint on exit.
                Defaults to True.
        """
        self.deployment_config = deployment_config
        self.docker_config = deployment_config.docker_config
        self.instance_config = deployment_config.instance_config
        self._exists = False

        if not getattr(self, "_from_factory", False):
            self.deployment_id = 'ab-'+str(uuid.uuid4())[:-7]  # truncated due to IE endpoint naming restrictions
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
        """
        Create a Deployment instance from an existing endpoint.

        Args:
            endpoint_name (str): Name of the existing endpoint.
            namespace (str, optional): Namespace of the endpoint. If None, assumes the user's namespace.
            teardown_on_exit (Optional[bool]): Whether to tear down the endpoint on exit.
                Defaults to False.

        Returns:
            Deployment: A new Deployment instance connected to the existing endpoint.

        Raises:
            Exception: If the endpoint is not found or cannot be accessed.
        """
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
        """
        Create a DeploymentConfig from an existing endpoint.

        Args:
            endpoint: The existing endpoint object.

        Returns:
            DeploymentConfig: A configuration object based on the existing endpoint.
        """
        endpoint_info = endpoint.raw
        namespace = endpoint.namespace

        def get_nested(d, keys, default=None):
            """Helper function to get nested dictionary values."""
            for key in keys:
                if isinstance(d, dict):
                    d = d.get(key, {})
                else:
                    return default
            return d if d != {} else default

        compute = get_nested(endpoint_info, ["compute"], {})
        provider = get_nested(endpoint_info, ["provider"], {})

        model_id = get_nested(endpoint_info, ["model", "repository"], '')
        task = get_nested(endpoint_info, ["model", "task"], '')
        docker_path = get_nested(endpoint_info, ['model', 'image', 'custom', 'url'], '')
        env = get_nested(endpoint_info, ["model", "env"], {})

        # TODO update this to work for Docker Config
        docker_config = DockerConfig(model_id=model_id, task=task, docker_path=docker_path, env_vars=env)
        # tgi_config = TGIConfig(
        #     model_id=get_nested(endpoint_info, ["model", "repository"]),
        #     max_batch_prefill_tokens=env.get("MAX_BATCH_PREFILL_TOKENS"),
        #     max_input_tokens=env.get("MAX_INPUT_TOKENS"),
        #     max_total_tokens=env.get("MAX_TOTAL_TOKENS"),
        #     num_shard=env.get("NUM_SHARD", 1),
        #     quantize=env.get("QUANTIZE"),
        # )

        compute_instance_config = ComputeInstanceConfig(
            id=compute.get("id"),
            vendor=provider.get("vendor"),
            region=provider.get("region"),
            accelerator=compute.get("accelerator"),
            instance_type=compute.get("instanceType"),
            instance_size=compute.get("instanceSize"),
            num_gpus=(
                int(compute.get("instanceSize", "0")[-1])
                if compute.get("instanceSize")
                else None
            ),
        )

        return DeploymentConfig(docker_config, compute_instance_config, namespace)

    def deploy_endpoint(self):
        """
        Deploy a new inference endpoint.

        This method creates a new inference endpoint using the configured settings.

        Raises:
            Exception: If the endpoint creation fails.
        """
        logger.info("Starting endpoint deployment process")
        try:
            logger.info("Creating inference endpoint...")
            endpoint = create_inference_endpoint(
                self.deployment_id,
                repository=self.docker_config.model_id,
                namespace=self.deployment_config.namespace,
                framework="pytorch",
                task=self.docker_config.task,
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
                    "url": self.docker_config.docker_path,
                    "env": self.docker_config.env_vars,
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
        """
        Resume a paused endpoint.

        This method resumes the endpoint if it was previously paused.
        """
        self.endpoint.resume().wait()

    def endpoint_status(self):
        """
        Get the current status of the endpoint.

        Returns:
            str or None: The status of the endpoint if it exists, None otherwise.
        """
        if hasattr(self, "endpoint"):
            deployment = self.endpoint.fetch()
            return deployment.status
        else:
            logger.error("Endpoint doesn't exist for this deployment.")
            return None
