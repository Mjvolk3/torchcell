# src/torchcell/datasets/fungal_up_down_transformer.py
# [[src.torchcell.datasets.fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/fungal_up_down_transformer.py
# Test file: src/torchcell/datasets/test_fungal_up_down_transformer.py

import os
from collections.abc import Callable
from typing import Optional

import torch
from pydantic import validator
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.models.fungal_up_down_transformer import (  # adjusted import
    FungalUpDownTransformer,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

os.makedirs("data/scerevisiae/fungal_utr_embed", exist_ok=True)


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @validator("gene_set")
    def validate_gene_set(cls, value):
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value


class FungalUpDownTransformerDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "species_downstream": ("window_three_prime", 300, True, True),
        "species_upstream": ("window_five_prime", 1003, True, True),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transformer_model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(root, transformer_model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        self.transformer_model_name = transformer_model_name

        if self.transformer_model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_transformer()
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

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

    def initialize_model(self) -> FungalUpDownTransformer | None:
        if self.transformer_model_name:
            split_name = self.transformer_model_name.split("_")
            if "downstream" in split_name and "species" in split_name:
                model_name = "downstream_species_lm"
            elif "upstream" in split_name and "species" in split_name:
                model_name = "upstream_species_lm"
            assert (
                model_name in FungalUpDownTransformer.VALID_MODEL_NAMES
            ), f"{model_name} not in valid model names."
            return FungalUpDownTransformer(model_name)
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

    # Adjust the model names accordingly.
    model_names = ["species_downstream", "species_upstream"]

    datasets = []
    for model_name in model_names:
        dataset = FungalUpDownTransformerDataset(
            root="data/scerevisiae/fungal_up_down_embed",
            genome=genome,
            transformer_model_name=model_name,
        )
        datasets.append(dataset)
        print(f"Dataset for {model_name}: {dataset}")

    some_data = datasets[0][genome.gene_set[42]]
    print(some_data)
    some_data = datasets[1][genome.gene_set[42]]
    print(some_data)
