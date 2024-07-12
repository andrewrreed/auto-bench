import requests
from typing import Dict
from dataclasses import dataclass, field

import pandas as pd
from huggingface_hub import create_inference_endpoint


@dataclass
class TGIConfig:
    model_id: str
    max_batch_prefill_tokens: int
    max_input_length: int
    max_total_tokens: int
    num_shard: int
    quantize: str
    estimated_memory_in_gigabytes: float


@dataclass
class DeploymentConfig:
    tgi_config: TGIConfig
    namespace: str
    vendor: str
    region: str
    instance_type: str
    instance_size: str
    env_vars: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.env_vars = {
            "MAX_BATCH_PREFILL_TOKENS": str(self.tgi_config.max_batch_prefill_tokens),
            "MAX_INPUT_LENGTH": str(self.tgi_config.max_input_length),
            "MAX_TOTAL_TOKENS": str(self.tgi_config.max_total_tokens),
            "NUM_SHARD": str(self.tgi_config.num_shard),
            "QUANTIZE": self.tgi_config.quantize,
        }

    @property
    def model_id(self):
        return self.tgi_config.model_id


class IEDeployment:

    def __init__(
        self,
        config: DeploymentConfig,
    ):
        self.config = config
        self.endpoint_name = f"{self.config.model_id}-autobench"

    def deploy_endpoint(self):

        try:
            endpoint = create_inference_endpoint(
                self.endpoint_name,
                repository=self.config.model_id,
                namespace=self.config.namespace,
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor=self.config.vendor,
                region=self.config.region,
                instance_size=self.config.instance_size,
                instance_type=self.config.instance_type,
                min_replica=0,
                max_replica=1,
                type="protected",
                custom_image={
                    "health_route": "/health",
                    "url": "ghcr.io/huggingface/text-generation-inference:latest",
                    "env": self.config.env_vars,
                },
            )

            endpoint.wait()
            self.endpoint = endpoint
        except Exception as e:
            print(e)

    def pause_endpoint(self):
        self.endpoint.pause()

    def resume_endpoint(self):
        self.endpoint.resume()

    def delete_endpoint(self):
        self.endpoint.delete()

    def status(self):
        return self.endpoint.status()


class IEComputeInstances:
    """A class representing the IEComputeInstances

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
        df = self.nested_json_to_df(data["vendors"])
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

        # Reorder the columns for better readability
        first_cols = [
            "vendor",
            "vendor_status",
            "region",
            "region_label",
            "region_status",
        ]
        column_order = first_cols + [col for col in df.columns if col not in first_cols]

        return df[column_order]

    @staticmethod
    def _filter_options(df):
        return df[
            (df["vendor_status"] == "available")
            & (df["region_status"] == "available")
            & (df["accelerator"] == "gpu")
            & (df["status"] == "available")
        ].reset_index(drop=True)
