# torchcell/datasets/embedding.py
# [[torchcell.datasets.embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/embedding.py
# Test file: torchcell/datasets/test_embedding.py

import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class BaseEmbeddingDataset(InMemoryDataset, ABC):
    def __init__(
        self,
        root: str,
        # genome: SCerevisiaeGenome,  # If we include this we get ddp error
        model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        if model_name and model_name not in self.MODEL_TO_WINDOW:
            valid_model_names = ", ".join(self.MODEL_TO_WINDOW.keys())
            raise ValueError(
                f"Invalid model_name '{model_name}'."
                f"Valid options are: {valid_model_names}"
            )
        # BUG
        # self.genome = genome # If we include this we get ddp error
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        super().__init__(root, transform, pre_transform)
        if self.model_name:
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location=self.device
            )
        else:
            self.data, self.slices = None, None

    @abstractmethod
    def initialize_model(self):
        pass

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> str:
        # if not self.model_name:
        # return "dummy_data.pt"
        return f"{self.model_name}.pt"

    def download(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def get_data_list(self):
        return [data for data in self]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # Use _data instead of data to suppress warning. might be dangerous.
            if idx in self._data.id:
                index = self._data.id.index(idx)
                return Data(
                    id=self._data.id[index],
                    dna_windows={
                        key: value[index]
                        for key, value in self._data.dna_windows.items()
                    },
                    embeddings={
                        key: value[index : index + 1]  # Ensure 2D shape
                        for key, value in self._data.embeddings.items()
                    },
                )
            else:
                raise KeyError(f"Gene {idx} not found in the dataset.")
        else:
            return super().__getitem__(idx)

    def __radd__(self, other):
        # if 'other' is the default integer 0, return the current instance
        if isinstance(other, int) and other == 0:
            return self
        # Otherwise, just fall back to the normal add operation
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        # Ensure the other object is of the same type
        if not isinstance(other, BaseEmbeddingDataset):
            raise ValueError("Can only add datasets of the same type.")

        combined_data_list = []

        # Create a dictionary from the current dataset for efficient lookup
        current_data_dict = {data_item.id: data_item for data_item in self}

        # Lists to store duplicate keys
        duplicate_dna_windows_keys = []
        duplicate_embeddings_keys = []

        # Combine the data from the other dataset
        for data_item in other:
            if data_item.id in current_data_dict:
                # Check for duplicate keys in dna_windows
                for key in data_item.dna_windows:
                    if key in current_data_dict[data_item.id].dna_windows:
                        duplicate_dna_windows_keys.append(key)
                    else:
                        # Merge the dna_windows dictionaries
                        current_data_dict[data_item.id].dna_windows[
                            key
                        ] = data_item.dna_windows[key]
                # Check for duplicate keys in embeddings
                for key in data_item.embeddings:
                    if key in current_data_dict[data_item.id].embeddings:
                        duplicate_embeddings_keys.append(key)
                    else:
                        # Merge the embeddings dictionaries
                        current_data_dict[data_item.id].embeddings[
                            key
                        ] = data_item.embeddings[key]
            else:
                combined_data_list.append(data_item)

        # If there are duplicates, raise an error
        if duplicate_dna_windows_keys:
            raise ValueError(
                "Duplicate keys found in dna_windows:"
                f"{', '.join(duplicate_dna_windows_keys)}"
            )
        if duplicate_embeddings_keys:
            raise ValueError(
                "Duplicate keys found in embeddings:"
                f"{', '.join(duplicate_embeddings_keys)}"
            )

        # Add the modified data items from the current dataset to the combined list
        combined_data_list.extend(current_data_dict.values())

        # Use collate to convert the combined list into the format InMemoryDataset expects
        data, slices = self.collate(combined_data_list)

        # Create a new dataset instance with the combined data
        combined_dataset = CombinedEmbedding(root=self.root, model_name=None)

        combined_dataset.data, combined_dataset.slices = data, slices

        return combined_dataset


# TODO, not sure if necessary but it renames dataset which is nice and clear,
# might not need ot inherit from BaseEmbeddingDataset since we have to implement
# initialize_model and process with pass
class CombinedEmbedding(BaseEmbeddingDataset):
    """
    This class represents the combined dataset which can
    be the result of combining two other datasets.
    """

    # No need to override the `__init__` method if it's going
    # to be the same as the parent class. However, if you need
    # any specialized initialization, you can define the method here.

    def initialize_model(self):
        """
        Implementation of abstract method.
        """
        # Provide a proper implementation if needed or just pass.
        pass

    def process(self):
        """
        Implementation of abstract method.
        """
        # Provide a proper implementation if needed or just pass.
        pass


if __name__ == "__main__":
    pass
