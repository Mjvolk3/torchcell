# torchcell/models/llm.py
# [[torchcell.models.llm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/llm.py
# Test file: torchcell/models/test_llm.py
"""Abstract base classes for nucleotide and peptide language models."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from attrs import define
from transformers import AutoModelForMaskedLM, AutoTokenizer


class NucleotideModel(ABC):
    """Abstract interface for nucleotide-sequence language models."""

    tokenizer: Any
    model: Any
    _max_sequence_size: int | None

    def __init__(self, model_name: str):
        """Initialize empty tokenizer/model and load the named model.

        Args:
            model_name: Identifier of the pretrained model to load.
        """
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
    def _check_and_download_model() -> None:
        """Fetch the model weights if absent; every subclass implements it."""
        raise NotImplementedError(
            "NucleotideModel._check_and_download_model is abstract; "
            "implement it in the subclass"
        )

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the tokenizer and model weights for the given model name."""
        raise NotImplementedError(
            "NucleotideModel.load_model is abstract; implement it in the subclass"
        )

    @abstractmethod
    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        """Return embeddings for the given sequences."""
        raise NotImplementedError(
            "NucleotideModel.embed is abstract; implement it in the subclass"
        )


class PeptideModel(ABC):
    """Abstract interface for peptide-sequence language models."""

    tokenizer: Any
    model: Any
    _max_sequence_size: int | None

    def __init__(self, model_name: str):
        """Initialize empty tokenizer/model and load the named model.

        Args:
            model_name: Identifier of the pretrained model to load.
        """
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
    def _check_and_download_model(model_name: str) -> None:
        """Fetch the model weights if absent; every subclass implements it."""
        raise NotImplementedError(
            "PeptideModel._check_and_download_model is abstract; "
            "implement it in the subclass"
        )

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load the tokenizer and model weights for the given model name."""
        raise NotImplementedError(
            "PeptideModel.load_model is abstract; implement it in the subclass"
        )

    @abstractmethod
    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        """Return embeddings for the given sequences."""
        raise NotImplementedError(
            "PeptideModel.embed is abstract; implement it in the subclass"
        )


@define
class pretrained_LLM:
    """Container pairing a tokenizer with a pretrained masked-LM model."""

    tokenizer: AutoTokenizer
    model: AutoModelForMaskedLM


if __name__ == "__main__":
    pass
