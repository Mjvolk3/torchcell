"""Model registry exposing sequence, embedding, graph, and benchmark models."""

from .act import act_register as act_register
from .constants import DNA_LLM_MAX_TOKEN_SIZE as DNA_LLM_MAX_TOKEN_SIZE
from .dcell import DCell as DCell
from .deep_set import DeepSet as DeepSet
from .fungal_up_down_transformer import (
    FungalUpDownTransformer as FungalUpDownTransformer,
)
from .graph_attention import GraphAttention as GraphAttention
from .graph_convolution import GraphConvolution as GraphConvolution
from .linear import SimpleLinearModel as SimpleLinearModel
from .mlp import Mlp as Mlp
from .nucleotide_transformer import NucleotideTransformer as NucleotideTransformer
from .self_attention_deep_set import SelfAttentionDeepSet as SelfAttentionDeepSet

model_constants = ["DNA_LLM_MAX_TOKEN_SIZE"]

model_building_blocks = ["act_register"]

simple_models = ["Mlp"]

models = [
    "FungalUpDownTransformer",
    "NucleotideTransformer",
    "DeepSet",
    "SelfAttentionDeepSet",
    "SimpleLinearModel",
    "GraphConvolution",
    "GraphAttention",
]

benchmark_model = ["DCell"]

__all__ = (
    model_constants + simple_models + model_building_blocks + models + benchmark_model
)
