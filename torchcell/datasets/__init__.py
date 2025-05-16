# torchcell/datasets/__init__.py
# [[torchcell.datasets.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/__init__.py
# Test file: tests/torchcell/datasets/test___init__.py


# TODO when we import this we get all sorts of import error
# from .dcell import DCellDataset

# scerevisiae datasets
from .scerevisiae.costanzo2016 import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    DmiCostanzo2016Dataset,
)

from .scerevisiae.kuzmin2018 import (
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
    DmiKuzmin2018Dataset,
    TmiKuzmin2018Dataset,
)
from .scerevisiae.kuzmin2020 import (
    SmfKuzmin2020Dataset,
    DmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
    DmiKuzmin2020Dataset,
    TmiKuzmin2020Dataset,
)

from .scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
)

from .scerevisiae.sgd import GeneEssentialitySgdDataset

# other datasets
from .codon_frequency import CodonFrequencyDataset
from .fungal_up_down_transformer import FungalUpDownTransformerDataset
from .nucleotide_transformer import NucleotideTransformerDataset
from .one_hot_gene import OneHotGeneDataset
from .protT5 import ProtT5Dataset
from .sgd_gene_graph import GraphEmbeddingDataset
from .esm2 import Esm2Dataset
from .codon_language_model import CalmDataset
from .random_embedding import RandomEmbeddingDataset

from .dataset_registry import dataset_registry


embedding_datasets = [
    "NucleotideTransformerDataset",
    "FungalUpDownTransformerDataset",
    "CodonFrequencyDataset",
    "OneHotGeneDataset",
    "ProtT5Dataset",
    "GraphEmbeddingDataset",
    "Esm2Dataset",
    "CalmDataset",
    "RandomEmbeddingDataset",
]


# yeast
costanzo2016_datasets = [
    "SmfCostanzo2016Dataset",
    "DmfCostanzo2016Dataset",
    "DmiCostanzo2016Dataset",
]
kuzmin2018_datasets = [
    "SmfKuzmin2018Dataset",
    "DmfKuzmin2018Dataset",
    "TmfKuzmin2018Dataset",
    "DmiKuzmin2018Dataset",
    "TmiKuzmin2018Dataset",
]
kuzmin2020_datasets = [
    "SmfKuzmin2020Dataset",
    "DmfKuzmin2020Dataset",
    "TmfKuzmin2020Dataset",
    "DmiKuzmin2020Dataset",
    "TmiKuzmin2020Dataset",
]
synth_leth_db_datasets = [
    "SynthLethalityYeastSynthLethDbDataset",
    "SynthRescueYeastSynthLethDbDataset",
]
sgd_datasets = ["GeneEssentialitySgdDataset"]

organism_datasets = (
    costanzo2016_datasets
    + kuzmin2018_datasets
    + kuzmin2020_datasets
    + synth_leth_db_datasets
    + sgd_datasets
)

# + experiment_datasets
__all__ = embedding_datasets + organism_datasets
# __all__ = core_datasets + embedding_datasets + registries
