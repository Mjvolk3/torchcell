# src/torchcell/datasets/scerevisiae/fungal_utr.py
# [[src.torchcell.datasets.scerevisiae.fungal_utr]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/scerevisiae/fungal_utr.py
# Test file: tests/torchcell/datasets/scerevisiae/test_fungal_utr.py

import os
from typing import Callable, Optional
from regex import P

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.models.fungal_utr_transformer import FungalUtrTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

os.makedirs("data/scerevisiae/fungal_utr_embed", exist_ok=True)


class FungalUtrTransformerDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "fut_window_3utr_300_undersize": ("window_3utr", 300, True),
        "fut_species_window_3utr_300_undersize": ("window_3utr", 300, True),
        "fut_species_window_5utr_1000_undersize": ("window_5utr", 1000, True),
        "fut_window_5utr_1000_undersize": ("window_5utr", 1000, True),
        "fut_window_3utr_300": ("window_3utr", 300, False),
        "fut_species_window_3utr_300": ("window_3utr", 300, False),
        "fut_species_window_5utr_1000": ("window_5utr", 1000, False),
        "fut_window_5utr_1000": ("window_5utr", 1000, False),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transformer_model_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.genome = genome
        self.transformer_model_name = transformer_model_name

        # Conditionally load the data
        if self.transformer_model_name:
            if not os.path.exists(self.processed_paths[0]):
                # Initialize the language model
                self.transformer = self.initialize_transformer()
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_transformer(self):
        """Initialize the transformer using the valid model name."""
        if self.transformer_model_name:
            split_name = self.transformer_model_name.split("_")
            if "3utr" in split_name and "species" not in split_name:
                model_name = "downstream_300"
            elif "3utr" in split_name and "species" in split_name:
                model_name = "species_downstream_300"
            elif "5utr" in split_name and "species" not in split_name:
                model_name = "upstream_1000"
            elif "5utr" in split_name and "species" in split_name:
                model_name = "species_upstream_1000"
            assert (
                model_name in FungalUtrTransformer.VALID_MODEL_NAMES
            ), f"{model_name} not in valid model names."
            return FungalUtrTransformer(model_name)
        return None

    def process(self):
        if not self.transformer_model_name:
            return

        data_list = []
        window_method, window_size, allow_undersize = self.MODEL_TO_WINDOW[
            self.transformer_model_name
        ]
        for gene_id in tqdm(self.genome.gene_set):
            sequence = self.genome[gene_id]
            dna_selection = getattr(sequence, window_method)(
                window_size, allow_undersize=allow_undersize
            )
            embeddings = self.transformer.embed(
                [dna_selection.seq],
                mean_embedding=True,
            )

            # Create or update the dna_window dictionary
            dna_window_dict = {self.transformer_model_name: dna_selection}

            data = Data(id=gene_id, dna_windows=dna_window_dict)
            data.embeddings = {self.transformer_model_name: embeddings}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    genome = SCerevisiaeGenome()

    model_names = [
        "fut_window_3utr_300_undersize",
        "fut_species_window_3utr_300_undersize",
        "fut_species_window_5utr_1000_undersize",
        "fut_window_5utr_1000_undersize",
        # "fut_window_3utr_300",
        # "fut_species_window_3utr_300",
        # "fut_species_window_5utr_1000",
        # "fut_window_5utr_1000",
    ]

    datasets = []
    for model_name in model_names:
        dataset = FungalUtrTransformerDataset(
            root="data/scerevisiae/fungal_utr_embed",
            genome=genome,
            transformer_model_name=model_name,
        )
        datasets.append(dataset)
        print(f"Dataset for {model_name}: {dataset}")

    dataset["YDR210W"]
    combined_dataset = datasets[0] + datasets[1] + datasets[2] + datasets[3]
    some_data = combined_dataset["YDR210W"]
    print(some_data)
