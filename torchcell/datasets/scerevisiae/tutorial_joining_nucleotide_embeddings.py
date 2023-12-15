# torchcell/datasets/scerevisiae/nucleotide_transformer.py
# [[torchcell.datasets.scerevisiae.nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/nucleotide_transformer.py
# Test file: tests/torchcell/datasets/scerevisiae/test_nucleotide_transformer.py

import os
from collections.abc import Callable
from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.fungal_utr_transformer import FungalUtrTransformerDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

if __name__ == "__main__":
    # genome
    genome = SCerevisiaeGenome()
    # nucleotide transformer
    nucleotide_transformer_name = "nt_window_3utr_300_undersize"
    nt_dataset = NucleotideTransformerDataset(
        root="data/scerevisiae/nucleotide_transformer_embed",
        genome=genome,
        transformer_model_name=nucleotide_transformer_name,
    )
    # fungal utr transformer
    fungal_utr_transformer_name = "fut_window_3utr_300_undersize"
    fut_dataset = FungalUtrTransformerDataset(
        root="data/scerevisiae/fungal_utr_embed",
        genome=genome,
        transformer_model_name=fungal_utr_transformer_name,
    )
    # Combine the datasets
    print()
