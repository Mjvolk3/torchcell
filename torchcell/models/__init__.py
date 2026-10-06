"""Model registry exposing sequence, embedding, graph, and benchmark models."""

from .act import act_register as act_register
from .dcell import DCell as DCell
from .deep_set import DeepSet as DeepSet
from .fungal_up_down_transformer import (
    FungalUpDownTransformer as FungalUpDownTransformer,
)
from .linear import SimpleLinearModel as SimpleLinearModel
from .mlp import Mlp as Mlp
from .nucleotide_transformer import NucleotideTransformer as NucleotideTransformer
from .self_attention_deep_set import SelfAttentionDeepSet as SelfAttentionDeepSet

model_building_blocks = ["act_register"]

simple_models = ["Mlp"]

models = [
    "FungalUpDownTransformer",
    "NucleotideTransformer",
    "DeepSet",
    "SelfAttentionDeepSet",
    "SimpleLinearModel",
]

benchmark_model = ["DCell"]

__all__ = simple_models + model_building_blocks + models + benchmark_model
