import os
from typing import Optional
from huggingface_hub import HfApi
from dataclasses import dataclass, field
from autobench.compute_manager import ComputeManager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_RESULTS_DIR = os.path.join(ROOT_DIR, "benchmark_results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
K6_BIN = None


@dataclass
class TGIConfig:
    model_id: str
    max_batch_prefill_tokens: int
    max_input_tokens: int
    max_total_tokens: int
    num_shard: Optional[int] = 1
    quantize: Optional[str] = None
    estimated_memory_in_gigabytes: Optional[float] = None

    def __post_init__(self):
        self.env_vars = {
            "MAX_INPUT_TOKENS": str(self.max_input_tokens),
            "MAX_TOTAL_TOKENS": str(self.max_total_tokens),
            "MAX_BATCH_PREFILL_TOKENS": str(self.max_batch_prefill_tokens),
            "NUM_SHARD": str(self.num_shard),
            "MODEL_ID": "/repository",
        }
        if self.quantize:
            self.env_vars["QUANTIZE"] = self.quantize


@dataclass
class ComputeInstanceConfig:
    id: str
    vendor: str
    region: str
    accelerator: str
    instance_type: str
    instance_size: str
    num_gpus: Optional[int] = None
    memory_in_gb: Optional[float] = None
    gpu_memory_in_gb: Optional[float] = None
    vendor_status: Optional[str] = None
    region_label: Optional[str] = None
    region_status: Optional[str] = None
    architecture: Optional[str] = None
    status: Optional[str] = None
    price_per_hour: Optional[float] = None
    num_cpus: Optional[int] = None

    @classmethod
    def from_id(cls, id: str):
        cm = ComputeManager()
        try:
            option = cm.options[cm.options.id == id].to_dict(orient="records")[0]
        except IndexError:
            raise Exception(f"Compute instance with id {id} not found.")
        return cls(**option)


@dataclass
class DeploymentConfig:
    tgi_config: TGIConfig
    instance_config: ComputeInstanceConfig
    namespace: str = None  # user's namespace if not provided

    def __post_init__(self):

        user_info = HfApi().whoami()

        if self.namespace is None or self.namespace == user_info["name"]:
            if user_info["canPay"]:
                self.namespace = user_info["name"]
            else:
                raise Exception(
                    "You must add billing information to your HuggingFace account to deploy Inference Endpoints."
                )
        else:
            namespaces = {
                org["name"]: org.get("canPay", False) for org in user_info["orgs"]
            }
            if self.namespace not in namespaces.keys():
                raise Exception(
                    f"Your user account does not have access to namespace: {self.namespace}."
                )
            if not namespaces[self.namespace]:
                raise Exception(
                    f"The namespace: {self.namespace} does not have billing enabled. Please add billing information to the namespace."
                )


@dataclass
class DatasetConfig:

    min_prompt_length: int = 50
    max_prompt_length: int = 500
    file_path: str = field(init=False)

    # hardcoded for now
    split: str = field(default="train_sft", init=False)
    name: str = field(default="HuggingFaceH4/ultrachat_200k", init=False)
    tokenizer_name: str = field(
        default="meta-llama/Meta-Llama-3-8B", init=False
    )  # fixed for consistency

    def __post_init__(self):
        self.file_path = f'benchmark_data/{("__").join(self.name.split("/"))}-{self.split}-{self.min_prompt_length}_min-{self.max_prompt_length}_max.json'
