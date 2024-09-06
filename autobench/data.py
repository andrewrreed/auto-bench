import os
import json
import tqdm
import datasets
from autobench.config import DataConfig
from loguru import logger


class BenchmarkDataset:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.file_path = self._adjust_file_path()

        logger.info(f"Initializing BenchmarkDataset with config: {data_config}")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        logger.debug(f"Ensured directory exists for file path: {self.file_path}")

    def _adjust_file_path(self):
        # Ensure data_file_path is relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adjusted_path = os.path.join(project_root, self.data_config.file_path)
        logger.debug(f"Adjusted file path: {adjusted_path}")
        return adjusted_path

    def build_data(self):
        # TODO:
        # Add a check to see if the data already exists
        # Generalize to other datasets
        # Add in "t-shirt" size options for dataset (e.g. multi-turn, long-propt RAG, etc.)
        dataset = datasets.load_dataset(
            self.data_config.dataset_name, split=self.data_config.dataset_split
        )
        logger.debug(f"Loaded dataset with {len(dataset)} items")

        # Select only the first 2k conversations
        max_conversations = min(2000, len(dataset))
        logger.info(f"Selecting up to {max_conversations} conversations")

        conversations = []

        for item in tqdm.tqdm(dataset, total=max_conversations):
            conv = item.get("conversations")
            if conv and conv[0]["from"] == "system":
                # Get only the initial user message
                conv = conv[1:2]
                conversations.append(conv)

                if len(conversations) >= max_conversations:
                    logger.debug("Reached maximum number of conversations")
                    break

        logger.info(f"Collected {len(conversations)} conversations")

        with open(self.file_path, "w") as f:
            json.dump(conversations, f, indent=4)
        logger.success(f"Saved conversations to {self.file_path}")

        # TODO: Implement dataset size options and generalize to other datasets
