import requests
from typing import Dict, List
from dataclasses import dataclass, field, asdict

import pandas as pd

from recommender.main import get_tgi_config


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


class ComputeOptionUtility:
    """A class representing the ComputeOptionUtility

    This class provides methods to retrieve and filter compute options from:
        https://api.endpoints.huggingface.cloud/#get-/v2/provider

    Attributes:
        options (DataFrame): A DataFrame containing the filtered compute options.

    Methods:
        get_ie_compute_options: Retrieves the compute options from the IECompute API.
        nested_json_to_df: Converts nested JSON data to a DataFrame.
        _filter_options: Filters the compute options based on specific criteria.
    """

    def __init__(self):
        self.options = self.get_ie_compute_options()

    def get_ie_compute_options(self):
        """
        Retrieves the compute options for the IE (Inference Engine) from a specified URL.

        Returns:
            pandas.DataFrame or None: A DataFrame containing the compute options for the IE if the request is successful,
            None otherwise.
        """
        url = "https://api.endpoints.huggingface.cloud/v2/provider"

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

        data = response.json()
        df = self._nested_json_to_df(data["vendors"])
        df = self._filter_options(df)
        return df

    @staticmethod
    def _nested_json_to_df(data):
        """
        Convert nested JSON data to a pandas DataFrame.

        Args:
            data (list): A list of dictionaries representing the nested JSON data.

        Returns:
            pandas.DataFrame: A DataFrame with flattened data.

        """
        flattened_data = []
        for vendor in data:
            for region in vendor["regions"]:
                for compute in region["computes"]:
                    flattened_data.append(
                        {
                            "vendor": vendor["name"],
                            "vendor_status": vendor["status"],
                            "region": region["name"],
                            "region_label": region["label"],
                            "region_status": region["status"],
                            **compute,
                        }
                    )

        df = pd.DataFrame(flattened_data)

        # Reorder the columns for better readability + rename some columns
        first_cols = [
            "vendor",
            "vendor_status",
            "region",
            "region_label",
            "region_status",
        ]
        column_order = first_cols + [col for col in df.columns if col not in first_cols]

        return df[column_order].rename(
            columns={
                "numAccelerators": "num_gpus",
                "memoryGb": "memory_in_gb",
                "gpuMemoryGb": "gpu_memory_in_gb",
                "instanceType": "instance_type",
                "instanceSize": "instance_size",
                "pricePerHour": "price_per_hour",
                "numCpus": "num_cpus",
            }
        )

    @staticmethod
    def _filter_options(df):
        return df[
            (df["vendor_status"] == "available")
            & (df["region_status"] == "available")
            & (df["accelerator"] == "gpu")
            & (df["status"] == "available")
        ].reset_index(drop=True)

    def get_instance_details(self, vendor: str, region: str, gpu_types: list):
        """
        Retrieve instance details based on the specified vendor, region, and GPU types.

        Args:
            vendor (str): The vendor of the instances.
            region (str): The region where the instances are located.
            gpu_types (list): A list of GPU types to filter the instances.

        Returns:
            list: A list of dictionaries containing the instance details that match the specified criteria.
        """
        return self.options[
            (self.options["vendor"] == vendor)
            & (self.options["region"] == region)
            & (self.options["instance_type"].isin(gpu_types))
        ].to_dict(orient="records")

    def get_viable_instance_configs(self, model_id: str, instances: List[Dict]):

        viable_instances = []
        for instance in instances:
            config = get_tgi_config(
                model_id,
                gpu_memory=(
                    instance["gpu_memory_in_gb"] * instance["num_gpus"]
                ),  # must be total VRAM on instance
                num_gpus=instance["num_gpus"],
            )
            if config:
                viable_instances.append(
                    {"tgi_config": asdict(config), "instance_config": instance}
                )
            else:
                print(
                    f"Instance {instance['name']} does not have enough memory to run the model: {self.model_id}\nExclude this instance from the benchmark."
                )

        return viable_instances
