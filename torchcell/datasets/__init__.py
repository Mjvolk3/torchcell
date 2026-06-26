# torchcell/datasets/__init__.py
# [[torchcell.datasets.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/__init__.py
# Test file: tests/torchcell/datasets/test___init__.py

"""Dataset registry for embedding and S. cerevisiae organism datasets."""


# TODO when we import this we get all sorts of import error
# from .dcell import DCellDataset

# scerevisiae datasets
# other datasets
from .codon_frequency import CodonFrequencyDataset as CodonFrequencyDataset
from .codon_language_model import CalmDataset as CalmDataset
from .dataset_registry import dataset_registry as dataset_registry
from .esm2 import Esm2Dataset as Esm2Dataset
from .fungal_up_down_transformer import (
    FungalUpDownTransformerDataset as FungalUpDownTransformerDataset,
)
from .node_embedding_builder import NodeEmbeddingBuilder as NodeEmbeddingBuilder
from .nucleotide_transformer import (
    NucleotideTransformerDataset as NucleotideTransformerDataset,
)
from .one_hot_gene import OneHotGeneDataset as OneHotGeneDataset
from .protT5 import ProtT5Dataset as ProtT5Dataset
from .random_embedding import RandomEmbeddingDataset as RandomEmbeddingDataset
from .scerevisiae.costanzo2016 import DmfCostanzo2016Dataset as DmfCostanzo2016Dataset
from .scerevisiae.costanzo2016 import DmiCostanzo2016Dataset as DmiCostanzo2016Dataset
from .scerevisiae.costanzo2016 import SmfCostanzo2016Dataset as SmfCostanzo2016Dataset
from .scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset as DmfKuzmin2018Dataset
from .scerevisiae.kuzmin2018 import DmiKuzmin2018Dataset as DmiKuzmin2018Dataset
from .scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset as SmfKuzmin2018Dataset
from .scerevisiae.kuzmin2018 import TmfKuzmin2018Dataset as TmfKuzmin2018Dataset
from .scerevisiae.kuzmin2018 import TmiKuzmin2018Dataset as TmiKuzmin2018Dataset
from .scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset as DmfKuzmin2020Dataset
from .scerevisiae.kuzmin2020 import DmiKuzmin2020Dataset as DmiKuzmin2020Dataset
from .scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset as SmfKuzmin2020Dataset
from .scerevisiae.kuzmin2020 import TmfKuzmin2020Dataset as TmfKuzmin2020Dataset
from .scerevisiae.kuzmin2020 import TmiKuzmin2020Dataset as TmiKuzmin2020Dataset
from .scerevisiae.sgd import GeneEssentialitySgdDataset as GeneEssentialitySgdDataset
from .scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset as SynthLethalityYeastSynthLethDbDataset,
)
from .scerevisiae.synth_leth_db import (
    SynthRescueYeastSynthLethDbDataset as SynthRescueYeastSynthLethDbDataset,
)
from .sgd_gene_graph import GraphEmbeddingDataset as GraphEmbeddingDataset

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
    "NodeEmbeddingBuilder",
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
