# src/torchcell/datasets/fungal_up_down_transformer.py
# [[src.torchcell.datasets.fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/fungal_up_down_transformer.py
# Test file: src/torchcell/datasets/test_fungal_up_down_transformer.py

import os
from collections.abc import Callable
from typing import Optional

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.models.fungal_up_down_transformer import (  # adjusted import
    FungalUpDownTransformer,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

os.makedirs("data/scerevisiae/fungal_utr_embed", exist_ok=True)


class CodonFrequencyDataset(BaseEmbeddingDataset):
    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(root, genome, transform, pre_transform)
        self.genome = genome

    def initialize_model(self) -> None:
        return None

    def process(self):
        if not self.transformer_model_name:
            return

        data_list = []
        (
            window_method,
            window_size,
            include_cds_codon,
            allow_undersize,
        ) = self.MODEL_TO_WINDOW[self.transformer_model_name]

        for gene_id in tqdm(self.genome.gene_set):
            sequence = self.genome[gene_id]
            dna_selection = getattr(sequence, window_method)(
                window_size, include_cds_codon, allow_undersize=allow_undersize
            )
            embeddings = self.transformer.embed(
                [dna_selection.seq], mean_embedding=True
            )

            dna_window_dict = {self.transformer_model_name: dna_selection}

            data = Data(id=gene_id, dna_windows=dna_window_dict)
            data.embeddings = {self.transformer_model_name: embeddings}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    genome = SCerevisiaeGenome()

    dataset = CodonFrequencyDataset(
        root="data/scerevisiae/codon_frequency", genome=genome
    )
    some_data = dataset[0][genome.gene_set[42]]
    print(some_data)
