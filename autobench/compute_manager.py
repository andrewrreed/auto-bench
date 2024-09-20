from loguru import logger
import requests
import pandas as pd
from typing import Dict, List, Literal
from urllib.parse import urlencode

from autobench.config import TGIConfig, ComputeInstanceConfig


class ComputeManager:
    """A class representing the ComputeManager

    This class provides methods to retrieve, filter, and validate compute options from:
        https://api.endpoints.huggingface.cloud/#get-/v2/provider

    Attributes:
        options (DataFrame): A DataFrame containing the filtered compute options.

    Methods:
        get_ie_compute_options: Retrieves the compute options from the IECompute API.
        nested_json_to_df: Converts nested JSON data to a DataFrame.
        _filter_options: Filters the compute options based on specific criteria.
    """

    def __init__(self):
        logger.info("Initializing ComputeManager")
        self.options = self.get_ie_compute_options()

    def get_ie_compute_options(self):
        """
        Retrieves the all gpu-enabled compute instance options that are available on Inference Endpoints.

        This method fetches the available compute options, processes the data,
        and returns a filtered DataFrame of compute instances.

        Returns:
            pandas.DataFrame: A DataFrame containing filtered compute options.
                              Each row represents a compute instance with its
                              specifications and availability.

        Raises:
            requests.RequestException: If there's an error fetching data from the API.
        """
        base_url = "https://api.endpoints.huggingface.cloud/v2/provider"

        try:
            response = requests.get(base_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch compute options: {e}")
            return None

        data = response.json()
        df = self._nested_json_to_df(data["vendors"])
        df = self._filter_options(df)
        df = self._clean_df(df)
        logger.info(f"Gathered {len(df)} compute instance options")
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

        logger.debug(f"Flattened {len(flattened_data)} compute options")
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

        logger.debug(
            f"Cleaned DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        return df.astype(type_map)

    @staticmethod
    def _filter_options(df):
        filtered_df = df[
            (df["vendor_status"] == "available")
            & (df["region_status"] == "available")
            & (df["accelerator"] == "gpu")
            & (df["status"] == "available")
        ].reset_index(drop=True)
        logger.info(f"Filtered {len(filtered_df)} available GPU options")
        return filtered_df

    def get_instance_details(
        self,
        gpu_types: List[str],
        preferred_vendor: str = "aws",
        preferred_region_prefix: Literal["us", "eu"] = "us",
    ):
        """
        Retrieve instance details based on specified GPU types, with optional vendor and region preferences.

        This method filters the available compute options based on the provided GPU types and sorts them
        according to the specified preferences. It prioritizes instances from the preferred vendor and region,
        and then sorts by price per hour in ascending order.

        Args:
            gpu_types (List[str]): A list of GPU types to filter the instances.
            preferred_vendor (str, optional): The preferred vendor for instances. Defaults to "aws".
            preferred_region_prefix (Literal["us", "eu"], optional): The preferred region prefix. Defaults to "us".

        Returns:
            List[Dict]: A list of dictionaries containing the instance details that match the specified criteria.

        Note:
            The method first filters instances by GPU type, then sorts them based on the number of GPUs,
            instance type, vendor preference, region preference, and price per hour. It then removes
            duplicates, keeping the first occurrence (which will be the lowest priced option for each
            unique combination of number of GPUs and instance type).
        """
        logger.info(
            f"Getting instance details for gpu_types={gpu_types}, preferred_vendor={preferred_vendor}, preferred_region_prefix={preferred_region_prefix}"
        )
        df = self.options[self.options["instance_type"].isin(gpu_types)]

        df_sorted = df.sort_values(
            by=[
                "num_gpus",
                "instance_type",
                "vendor",
                "region",
                "price_per_hour",
            ],
            key=lambda col: (
                col
                if col.name not in ["vendor", "region"]
                else col.map(
                    lambda x: (
                        (0 if x == preferred_vendor else 1)
                        if col.name == "vendor"
                        else (0 if x.startswith(preferred_region_prefix) else 1)
                    )
                )
            ),
        )

        df_deduplicated = df_sorted.drop_duplicates(
            subset=["num_gpus", "instance_type"], keep="first"
        )

        return df_deduplicated.to_dict(orient="records")

    @staticmethod
    def get_tgi_config(model_id: str, gpu_memory: int, num_gpus: int):
        """
        Retrieves a TGI (Text Genereration Inference) configuration for a given model.

        Args:
            model_id (str): The ID of the model.
            gpu_memory (int): Total available GPU memory of the instance in GB.
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

        logger.info(
            f"Fetching TGI config for model_id={model_id}, gpu_memory={gpu_memory}, num_gpus={num_gpus}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug("Successfully retrieved TGI config")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            error_detail = None
            if response.text:
                try:
                    error_detail = response.json().get("detail")
                except ValueError:
                    error_detail = response.text
            logger.error(
                f"HTTP error occurred while fetching TGI config: {http_err}. Detail: {error_detail}"
            )
            return None
        except requests.exceptions.RequestException as err:
            logger.error(f"Error occurred while fetching TGI config: {err}")
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
        logger.info(f"Finding viable instance configs for model_id={model_id}")
        viable_instances = []
        for instance in instances:
            print(f"Num GPUs: {instance['num_gpus']}")
            print(f"Total GPU Memory: {instance['gpu_memory_in_gb']}")
            print()

            total_gpu_memory = instance["gpu_memory_in_gb"]
            num_gpus = instance["num_gpus"]

            config = self.get_tgi_config(
                model_id,
                gpu_memory=total_gpu_memory,
                num_gpus=num_gpus,
            )
            if config:
                tgi_config = TGIConfig(**config["config"])
                instance_config = ComputeInstanceConfig(**instance)
                viable_instances.append(
                    {"tgi_config": tgi_config, "instance_config": instance_config}
                )
                logger.debug(f"Found viable instance: {instance['id']}")
            else:
                logger.warning(
                    f"Instance {instance['id']} does not have enough memory to run the model: {model_id}. Excluding from benchmark."
                )

        logger.info(
            f"Found {len(viable_instances)} viable instances for model_id={model_id}"
        )
        return viable_instances
