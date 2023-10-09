# src/torchcell/datasets/__init__.py
# [[src.torchcell.datasets.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/__init__.py
# Test file: tests/torchcell/datasets/test___init__.py
from .cell import CellDataset
from .codon_frequency import CodonFrequencyDataset
from .fungal_up_down_transformer import FungalUpDownTransformerDataset
from .nucleotide_transformer import NucleotideTransformerDataset

core_dataset = ["CellDataset"]

sequence_embedding_datasets = [
    "NucleotideTransformerDataset",
    "FungalUpDownTransformerDataset",
    "CodonFrequencyDataset",
]

__all__ = core_dataset + sequence_embedding_datasets
