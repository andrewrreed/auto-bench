import requests
from typing import Dict, List
from urllib.parse import urlencode
from dataclasses import dataclass, field, asdict

import pandas as pd


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


class ComputeOptionUtil:
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
        base_url = "https://api.endpoints.huggingface.cloud/v2/provider"

        try:
            response = requests.get(base_url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

        data = response.json()
        df = self._nested_json_to_df(data["vendors"])
        df = self._filter_options(df)
        df = self._clean_df(df)
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

        return pd.DataFrame(flattened_data)

    @staticmethod
    def _clean_df(df):

        # Reorder the columns for better readability + rename some columns
        first_cols = [
            "vendor",
            "vendor_status",
            "region",
            "region_label",
            "region_status",
        ]
        column_order = first_cols + [col for col in df.columns if col not in first_cols]

        df = df[column_order].rename(
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

        # Fix data types
        type_map = {"memory_in_gb": int, "gpu_memory_in_gb": int, "num_cpus": int}

        return df.astype(type_map)

    @staticmethod
    def _filter_options(df):
        return df[
            (df["vendor_status"] == "available")
            & (df["region_status"] == "available")
            & (df["accelerator"] == "gpu")
            & (df["status"] == "available")
        ].reset_index(drop=True)

    def get_instance_details(self, vendor: str, region: str, gpu_types: List[str]):
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

    @staticmethod
    def get_tgi_config(model_id: str, gpu_memory: int, num_gpus: int):
        """
        Retrieves a TGI (Text Genereration Inference) configuration for a given model.

        Args:
            model_id (str): The ID of the model.
            gpu_memory (int): The amount of GPU memory required for the model.
            num_gpus (int): The number of GPUs required for the model.

        Returns:
            dict: The TGI configuration as a dictionary.

        Raises:
            requests.exceptions.HTTPError: If an HTTP error occurs during the request.
            requests.exceptions.RequestException: If an error occurs during the request.

        """
        base_url = "https://huggingface.co/api/integrations/tgi/v1/config"

        params = {"model_id": model_id, "gpu_memory": gpu_memory, "num_gpus": num_gpus}

        encoded_params = urlencode(params)
        url = f"{base_url}?{encoded_params}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            error_detail = None
            if response.text:
                try:
                    error_detail = response.json().get("detail")
                except ValueError:
                    error_detail = response.textf
            print(f"HTTP error occurred: {http_err}. Detail: {error_detail}")
            return None
        except requests.exceptions.RequestException as err:
            print(f"An error occurred: {err}")
            return None

    def get_viable_instance_configs(self, model_id: str, instances: List[Dict]):
        """
        Get a list of viable instance configurations for running a specific model.

        Will return one TGI config per instance configuration assuming that instance has enough memory to run the model.

        Args:
            model_id (str): The ID of the model to be run.
            instances (List[Dict]): A list of dictionaries representing different instance configurations.

        Returns:
            List[Dict]: A list of dictionaries containing viable instance configurations along with their corresponding TGI configurations.

        """
        viable_instances = []
        for instance in instances:
            config = self.get_tgi_config(
                model_id,
                gpu_memory=(
                    instance["gpu_memory_in_gb"] * instance["num_gpus"]
                ),  # must be total VRAM on instance
                num_gpus=instance["num_gpus"],
            )
            if config:
                viable_instances.append(
                    {"tgi_config": config["config"], "instance_config": instance}
                )
            else:
                print(
                    f"Instance {instance['id']} does not have enough memory to run the model: {model_id}\nExclude this instance from the benchmark."
                )

        return viable_instances
