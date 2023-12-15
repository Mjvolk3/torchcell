# torchcell/datasets/__init__.py
# [[torchcell.datasets.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/__init__.py
# Test file: tests/torchcell/datasets/test___init__.py
from .cell import CellDataset
from .codon_frequency import CodonFrequencyDataset
from .dcell import DCellDataset

# from .costanzo2016 import DmfCostanzo2016Dataset, SmfCostanzo2016Dataset # in scerevisiae
from .fungal_up_down_transformer import FungalUpDownTransformerDataset
from .nucleotide_transformer import NucleotideTransformerDataset
from .one_hot_gene import OneHotGeneDataset
from .protT5 import ProtT5Dataset

core_datasets = ["CellDataset", "DCellDataset"]

embedding_datasets = [
    "NucleotideTransformerDataset",
    "FungalUpDownTransformerDataset",
    "CodonFrequencyDataset",
    "OneHotGeneDataset",
    "ProtT5Dataset",
]


# experiment_datasets = ["SmfCostanzo2016Dataset", "DmfCostanzo2016Dataset"]


# + experiment_datasets

__all__ = core_datasets + embedding_datasets
