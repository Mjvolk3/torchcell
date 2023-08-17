# src/torchcell/datasets/nucleotide_embedding.py
# [[src.torchcell.datasets.nucleotide_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/nucleotide_embedding.py
# Test file: tests/torchcell/datasets/test_nucleotide_embedding.py

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
        genome: SCerevisiaeGenome,
        transformer_model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        if (
            transformer_model_name
            and transformer_model_name not in self.MODEL_TO_WINDOW
        ):
            valid_model_names = ", ".join(self.MODEL_TO_WINDOW.keys())
            raise ValueError(
                f"Invalid transformer_model_name '{transformer_model_name}'."
                f"Valid options are: {valid_model_names}"
            )
        self.genome = genome
        self.transformer_model_name = transformer_model_name
        self.transformer = self.initialize_transformer()
        super().__init__(root, transform, pre_transform)
        if self.transformer_model_name:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

    @abstractmethod
    def initialize_transformer(self):
        pass

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> str:
        # if not self.transformer_model_name:
        # return "dummy_data.pt"
        return f"{self.transformer_model_name}.pt"

    def download(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def get_data_list(self):
        return [data for data in self]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # Use _data instead of data to suprress warning. might be dangerous.
            if idx in self._data.id:
                index = self._data.id.index(idx)
                return Data(
                    id=self._data.id[index],
                    dna_windows={
                        key: value[index]
                        for key, value in self._data.dna_windows.items()
                    },
                    embeddings={
                        key: value[index]
                        for key, value in self._data.embeddings.items()
                    },
                )
            else:
                raise KeyError(f"Gene {idx} not found in the dataset.")
        else:
            return super().__getitem__(idx)

    def __add__(self, other):
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
        combined_dataset = self.__class__(
            root=self.root, genome=None, transformer_model_name=None
        )
        combined_dataset.data, combined_dataset.slices = data, slices

        return combined_dataset


if __name__ == "__main__":
    pass
