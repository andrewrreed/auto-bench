import os
import json
import random
import datasets
from transformers import AutoTokenizer
from autobench.config import DatasetConfig
from loguru import logger


class BenchmarkDataset:
    """
    A class to build and manage datasets for benchmarking.

    This class handles the creation and storage of benchmark datasets based on
    the provided configuration. It loads data from a specified source, tokenizes
    it, and saves it to a local file for future use.

    Attributes:
        data_config (DatasetConfig): Configuration for the dataset.
        file_path (str): Path to save the benchmark dataset.
    """

    def __init__(self, data_config: DatasetConfig):
        """
        Initializes the BenchmarkDataset with the given configuration.

        Args:
            data_config (DatasetConfig): Configuration for the dataset.
        """
        self.data_config = data_config
        self.file_path = data_config.file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.build_benchmark_dataset()

    def build_benchmark_dataset(self):
        """
        Builds the benchmark dataset if it doesn't already exist.

        This method loads the dataset, tokenizes it, filters it based on token
        length, samples a subset, and saves it to a local file.
        """
        if not os.path.exists(self.file_path):
            dataset = datasets.load_dataset(
                self.data_config.name, split=self.data_config.split
            )
            tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)

            dataset = dataset.map(
                lambda example: {
                    "num_tokens": len(tokenizer.encode(example["prompt"]))
                },
                num_proc=8,
            )
            dataset = sample_dataset(
                dataset=dataset,
                n_samples=2500,
                min_tokens=self.data_config.min_prompt_length,
                max_tokens=self.data_config.max_prompt_length,
            )

            with open(self.file_path, "w") as f:
                json.dump(dataset["prompt"], f, indent=4)

            logger.success(f"Saved benchmark dataset to {self.file_path}")
        else:
            logger.info(f"Loaded benchmark dataset from {self.file_path}")


def sample_dataset(dataset, n_samples, min_tokens, max_tokens):
    """
    Samples a subset of the dataset based on token length constraints.

    This function filters the dataset to include only examples within the
    specified token length range, then randomly samples a subset of the
    desired size.

    Args:
        dataset (datasets.Dataset): The input dataset to sample from.
        n_samples (int): The number of samples to return.
        min_tokens (int): The minimum number of tokens allowed per example.
        max_tokens (int): The maximum number of tokens allowed per example.

    Returns:
        datasets.Dataset: A sampled subset of the input dataset.
    """
    filtered_dataset = dataset.filter(
        lambda x: min_tokens <= x["num_tokens"] <= max_tokens, num_proc=8
    )

    total_samples = len(filtered_dataset)

    # If we have fewer samples than requested, return all of them
    if total_samples <= n_samples:
        return filtered_dataset

    # Otherwise, randomly sample n_samples
    random.seed(42)
    random_indices = random.sample(range(total_samples), n_samples)
    sampled_dataset = filtered_dataset.select(random_indices)

    return sampled_dataset
