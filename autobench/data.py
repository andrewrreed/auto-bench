import os
import json
import random
import datasets
from transformers import AutoTokenizer
from autobench.config import DatasetConfig
from loguru import logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BenchmarkDataset:
    def __init__(self, data_config: DatasetConfig):
        self.data_config = data_config
        self.file_path = self._adjust_file_path()

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.build_benchmark_dataset()

    def _adjust_file_path(self):
        """Ensure data_file_path is relative to the project root"""
        adjusted_path = os.path.join(ROOT_DIR, self.data_config.file_path)
        return adjusted_path

    def build_benchmark_dataset(self):

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

            logger.success(f"Saved conversations to {self.file_path}")


def sample_dataset(dataset, n_samples, min_tokens, max_tokens):

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
