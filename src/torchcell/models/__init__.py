from .act import act_register
from .constants import DNA_LLM_MAX_TOKEN_SIZE
from .dcell import DCell, DCellLinear, dcell_from_networkx
from .deep_set import DeepSet
from .deep_set_transformer import DeepSetTransformer
from .fungal_up_down_transformer import FungalUpDownTransformer
from .graph_attention import GraphAttention
from .graph_convolution import GraphConvolution
from .linear import SimpleLinearModel
from .mlp import Mlp
from .nucleotide_transformer import NucleotideTransformer

model_constants = ["DNA_LLM_MAX_TOKEN_SIZE"]

model_building_blocks = ["act_register"]

simple_models = ["Mlp"]

models = [
    "FungalUpDownTransformer",
    "NucleotideTransformer",
    "DeepSet",
    "DeepSetTransformer",
    "SimpleLinearModel",
    "GraphConvolution",
    "GraphAttention",
]

incidence_models = ["DCell", "DCellLinear"]

__all__ = (
    model_constants + simple_models + model_building_blocks + models + incidence_models
)
