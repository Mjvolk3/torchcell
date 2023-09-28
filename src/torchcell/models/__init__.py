from .act import act_register
from .constants import DNA_LLM_MAX_TOKEN_SIZE
from .deep_set import DeepSet
from .fungal_up_down_transformer import FungalUpDownTransformer
from .mlp import Mlp
from .nucleotide_transformer import NucleotideTransformer

model_constants = ["DNA_LLM_MAX_TOKEN_SIZE"]

model_building_blocks = ["act_register"]

simple_models = ["Mlp"]

models = ["FungalUpDownTransformer", "NucleotideTransformer", "DeepSet"]

__all__ = model_constants + simple_models + model_building_blocks + models
