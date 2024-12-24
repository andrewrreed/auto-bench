import os
import json
import random
import datasets
import pandas as pd
from transformers import AutoTokenizer
from autobench.config import DatasetConfig
from loguru import logger


class BaseDataset:
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
        Initializes the BaseDataset with the given configuration.

        Args:
            data_config (DatasetConfig): Configuration for the dataset.
        """
        self.data_config = data_config
        self.file_path = data_config.file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.build_dataset()

    def build_dataset(self):
        if not os.path.exists(self.file_path):
            dataset = self.load_data()
            tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)

            dataset = self.tokenize_and_filter(dataset, tokenizer)
            dataset = self.sample_dataset(dataset)

            processed_data = self.process_data(dataset)

            with open(self.file_path, "w") as f:
                json.dump(processed_data, f, indent=4)

            logger.success(f"Saved dataset to {self.file_path}")
        else:
            logger.info(f"Loaded dataset from {self.file_path}")

    def load_data(self):
        return datasets.load_dataset(
                self.data_config.name, split=self.data_config.split
                )

    def tokenize_and_filter(self, dataset, tokenizer):
        return dataset.map(
                lambda example: {
                    "num_tokens": len(tokenizer.encode(example[self.data_config.input_key]))
                    },
                num_proc=8,
                )

    def sample_dataset(self, dataset):
        return sample_dataset(
                dataset=dataset,
                n_samples=self.data_config.n_samples,
                min_tokens=self.data_config.min_prompt_length,
                max_tokens=self.data_config.max_prompt_length,
                )

    def process_data(self, dataset):
        # Default processing, can be overridden
        return dataset[self.data_config.input_key]


class BenchmarkDataset(BaseDataset):
    def __init__(self, data_config):
        super().__init__(data_config)

    def process_data(self, dataset):
        # Specific to BenchmarkDataset
        return dataset["prompt"]


class ClassificationDataset(BaseDataset):
    def __init__(self, data_config):
        super().__init__(data_config)

    def load_data(self):
        return datasets.load_dataset(
                self.data_config.name, self.data_config.split, split='train'
                )

    def process_data(self, dataset):
        df = pd.DataFrame(dataset)
        label_mapping = {0: "positive", 1: "neutral", 2: "negative"}
        df['label'] = df['label'].map(label_mapping).fillna("unknown")
        df = df[['text', 'label']]
        return df.to_dict('records')


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
