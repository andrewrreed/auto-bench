import os
import json
import tqdm
import datasets
from autobench.config import DataConfig


class BenchmarkDataset:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.file_path = self._adjust_file_path()

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def _adjust_file_path(self):
        # Ensure data_file_path is relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, self.data_config.file_path)

    def build_data(self):
        # TODO:
        # Add a check to see if the data already exists
        # Generalize to other datasets
        # Add in "t-shirt" size options for dataset (e.g. multi-turn, long-propt RAG, etc.)
        dataset = datasets.load_dataset(
            self.data_config.dataset_name, split=self.data_config.dataset_split
        )

        # Select only the first 2k conversations
        max = min(2000, len(dataset))
        conversations = []

        for item in tqdm.tqdm(dataset, total=max):
            conv = item.get("conversations")
            if conv and conv[0]["from"] == "system":
                # Get only the initial user message
                conv = conv[1:2]
                conversations.append(conv)

                if len(conversations) >= max:
                    break

        with open(self.file_path, "w") as f:
            json.dump(conversations, f, indent=4)
