from dataclasses import dataclass
from loguru import logger


@dataclass
class TGIConfig:
    model_id: str
    max_batch_prefill_tokens: int
    max_input_length: int
    max_total_tokens: int
    num_shard: int
    quantize: str
    estimated_memory_in_gigabytes: float

    def __post_init__(self):
        logger.info(f"Initializing TGIConfig for model: {self.model_id}")
        self.env_vars = {
            "MAX_BATCH_PREFILL_TOKENS": str(self.max_batch_prefill_tokens),
            "MAX_INPUT_LENGTH": str(self.max_input_length),
            "MAX_TOTAL_TOKENS": str(self.max_total_tokens),
            "NUM_SHARD": str(self.num_shard),
            "MODEL_ID": "/repository",
        }
        if self.quantize:
            self.env_vars["QUANTIZE"] = self.quantize
        logger.debug(f"TGIConfig environment variables set: {self.env_vars}")


@dataclass
class ComputeInstanceConfig:
    id: str
    vendor: str
    vendor_status: str
    region: str
    region_label: str
    region_status: str
    accelerator: str
    num_gpus: int
    memory_in_gb: float
    gpu_memory_in_gb: float
    instance_type: str
    instance_size: str
    architecture: str
    status: str
    price_per_hour: float
    num_cpus: int

    def __post_init__(self):
        logger.info(f"Initializing ComputeInstanceConfig for instance: {self.id}")
        logger.debug(
            f"ComputeInstanceConfig details: vendor={self.vendor}, region={self.region}, accelerator={self.accelerator}"
        )


@dataclass
class DeploymentConfig:
    tgi_config: TGIConfig
    instance_config: ComputeInstanceConfig

    def __post_init__(self):
        logger.info("Initializing DeploymentConfig")
        logger.debug(
            f"DeploymentConfig: TGI model={self.tgi_config.model_id}, Instance={self.instance_config.id}"
        )


@dataclass
class DataConfig:
    dataset_name: str = "Open-Orca/slimorca-deduped-cleaned-corrected"
    dataset_split: str = "train"
    file_path: str = "benchmark_data/data.json"

    def __post_init__(self):
        logger.info(f"Initializing DataConfig with dataset: {self.dataset_name}")
        logger.debug(
            f"DataConfig details: split={self.dataset_split}, file_path={self.file_path}"
        )
