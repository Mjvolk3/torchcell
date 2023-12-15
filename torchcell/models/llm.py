# torchcell/models/llm.py
# [[torchcell.models.llm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/llm.py
# Test file: torchcell/models/test_llm.py

from abc import ABC, abstractmethod

import torch
from attrs import define
from transformers import AutoModelForMaskedLM, AutoTokenizer


class NucleotideModel(ABC):
    def __init__(self, model_name: str):
        self.tokenizer = None
        self.model = None
        self.load_model(model_name)

    @property
    def max_sequence_size(self) -> int:
        """Returns the maximum sequence size for the transformer model."""
        if self._max_sequence_size is None:
            raise ValueError("Max size has not been set for this model.")
        return self._max_sequence_size

    @staticmethod
    @abstractmethod
    def _check_and_download_model():
        pass

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def embed(self, sequences: list, mean_embedding: bool = False) -> torch.Tensor:
        pass


class PeptideModel(ABC):
    def __init__(self, model_name: str):
        self.tokenizer = None
        self.model = None
        self.load_model(model_name)

    @property
    def max_sequence_size(self) -> int:
        """Returns the maximum sequence size for the transformer model."""
        if self._max_sequence_size is None:
            raise ValueError("Max size has not been set for this model.")
        return self._max_sequence_size

    @staticmethod
    @abstractmethod
    def _check_and_download_model(model_name: str):
        pass

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def embed(self, sequences: list, mean_embedding: bool = False) -> torch.Tensor:
        pass


@define
class pretrained_LLM:
    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM


if __name__ == "__main__":
    pass
