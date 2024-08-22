import os
import uuid
from typing import Any
from dataclasses import dataclass


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
        self.env_vars = {
            "MAX_BATCH_PREFILL_TOKENS": str(self.max_batch_prefill_tokens),
            "MAX_INPUT_LENGTH": str(self.max_input_length),
            "MAX_TOTAL_TOKENS": str(self.max_total_tokens),
            "NUM_SHARD": str(self.num_shard),
            "MODEL_ID": "/repository",
        }
        if self.quantize:
            self.env_vars["QUANTIZE"] = self.quantize


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


@dataclass
class DeploymentConfig:
    tgi_config: TGIConfig
    instance_config: ComputeInstanceConfig

    def __post_init__(self):
        self.deployment_id = str(uuid.uuid4())[:-4]


@dataclass
class DataConfig:
    dataset_name: str = "Open-Orca/slimorca-deduped-cleaned-corrected"
    dataset_split: str = "train"
    data_file_path: str = "benchmark_data/data.json"

    def __post_init__(self):
        # Ensure data_file_path is relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_file_path = os.path.join(project_root, self.data_file_path)


@dataclass
class K6Config:
    host: str
    executor: Any
    data_file: str

    def __str__(self):
        return f"K6Config(url={self.host} executor={self.executor} data_file={self.data_file})"


@dataclass
class BenchmarkConfig:
    deployment_config: DeploymentConfig
    data_config: DataConfig
    k6_config: K6Config
