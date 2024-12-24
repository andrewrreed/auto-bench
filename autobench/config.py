import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from huggingface_hub import HfApi

from autobench.compute_manager import ComputeManager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "benchmark_data")
BENCHMARK_RESULTS_DIR = os.path.join(ROOT_DIR, "benchmark_results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
K6_BIN = "~/.local/bin/k6-sse"


@dataclass
class DockerConfig:
    docker_path: str
    model_id: str
    task: str
    env_vars: Dict[str, str] = field(default_factory=dict)


class TGIConfig(DockerConfig):
    docker_path = "ghcr.io/huggingface/text-generation-inference:2.3.0"
    task = "text-generation"
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


class InfinityConfig(DockerConfig):
    def __init__(
            self,
            docker_path: str,
            model_id: str,
            task: str,
            compile: Optional[Literal['true', 'false']],
            better_transformer: Optional[Literal['true', 'false']],
            engine: Literal["torch", "optimum", "ctranslate2", "cpu", "neuron"] = "torch",
            batch_size: int = 16,
            dtype: Literal["auto", "float16", "float32", "bfloat16", "int8", "fp8"] = "auto",
            embedding_dtype: Literal[
                "float32", "float16", "bfloat16", "int8", "uint8", "binary", "ubinary"] = "float32",
            pooling_method: Literal["auto", "mean", "cls"] = "auto"
            ):
        super().__init__(docker_path=docker_path, model_id=model_id, env_vars={}, task=task)

        # Set additional InfinityConfig-specific attributes
        self.engine = engine
        self.batch_size = batch_size
        self.dtype = dtype
        self.embedding_dtype = embedding_dtype
        self.pooling_method = pooling_method
        self.compile = compile
        self.better_transformer = better_transformer

        # Infer env_vars from the provided parameters
        self.env_vars = {
            "ENGINE": self.engine,
            "BATCH_SIZE": str(self.batch_size),
            "DTYPE": self.dtype,
            "EMBEDDING_DTYPE": self.embedding_dtype,
            "POOLING_METHOD": self.pooling_method,
            "COMPILE": self.compile,
            "BETTERTRANSFORMER": self.better_transformer,
            }


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
    quota: Dict[str, int] = field(default_factory=dict)

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
    docker_config: DockerConfig
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
    tokenizer_name: str
    min_prompt_length: int = 50
    max_prompt_length: int = 500
    file_path: str = field(init=False)

    # hardcoded for now
    split: str = field(default="all", init=False)
    name: str = field(default="tyqiangz/multilingual-sentiments", init=False)

    def __post_init__(self):
        self.file_path = os.path.join(
                DATA_DIR,
                f'{("__").join(self.name.split("/"))}-{self.split}-{self.min_prompt_length}_min-{self.max_prompt_length}_max.json',
                )
