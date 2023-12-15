# torchcell/datasets/fungal_up_down_transformer.py
# [[torchcell.datasets.fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/fungal_up_down_transformer.py
# Test file: torchcell/datasets/test_fungal_up_down_transformer.py

import os
from collections.abc import Callable
from typing import Optional

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.models.fungal_up_down_transformer import (  # adjusted import
    FungalUpDownTransformer,
)
from torchcell.sequence import GeneSet, ParsedGenome, compute_codon_frequency
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class CodonFrequencyDataset(BaseEmbeddingDataset):
    # Could add frequency for other parts of sequence
    # but this doesn't make much sense
    MODEL_TO_WINDOW = {"cds_codon_frequency": None}

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.genome = genome

        self.model_name = "cds_codon_frequency"
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

    # This is done to avoid pkl error when since genome uses sqlite
    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        # BUG we have to do this black magic because when you merge datasets with +
        # the genome is None
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self):
        return None

    def process(self):
        data_list = []

        for gene_id in tqdm(self.genome.gene_set):
            sequence = str(self.genome[gene_id].cds.seq)

            # Check if the sequence is valid for codon frequency computation
            try:
                codon_frequency = compute_codon_frequency(sequence)
            except ValueError:
                continue

            # Create a Data object
            data = Data(id=gene_id, dna_windows={})
            data.embeddings = {
                self.model_name: torch.tensor(codon_frequency.values()).unsqueeze(0)
            }
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    genome = SCerevisiaeGenome()
    dataset = CodonFrequencyDataset(
        root="data/scerevisiae/codon_frequency", genome=genome
    )

    some_data = dataset[0]  # Should give you the first dataset item
    print(some_data)
