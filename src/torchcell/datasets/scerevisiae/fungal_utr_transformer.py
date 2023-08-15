# src/torchcell/datasets/scerevisiae/fungal_utr.py
# [[src.torchcell.datasets.scerevisiae.fungal_utr]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/scerevisiae/fungal_utr.py
# Test file: tests/torchcell/datasets/scerevisiae/test_fungal_utr.py

import os
from typing import Callable, Optional

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.models.fungal_utr_transformer import FungalUtrTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

os.makedirs("data/scerevisiae/fungal_utr_embed", exist_ok=True)


class FungalUtrTransformerDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "downstream_300": ("window_3utr", 300),
        "species_downstream_300": ("window_3utr", 300),
        "species_upstream_1000": ("window_5utr", 1000),
        "upstream_1000": ("window_5utr", 1000),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transformer_model_name: Optional[str] = None,
        allow_undersize: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        print(
            f"Initializing with transformer_model_name: {transformer_model_name}"
        )  # TODO  Added
        self.genome = genome
        self.transformer_model_name = transformer_model_name  # Not None
        self.allow_undersize = allow_undersize
        # Only initialize the transformer if a model name is provided
        if transformer_model_name:
            self.transformer = FungalUtrTransformer(transformer_model_name)
        else:
            self.transformer = None
        # Conditionally load the data
        if self.transformer_model_name is not None:
            if not os.path.exists(self.processed_paths[0]):
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_transformer(self):
        if self.transformer_model_name:
            return FungalUtrTransformer(self.transformer_model_name)
        return None

    @property
    def processed_file_names(self) -> str:
        if self.transformer_model_name is None:
            return "dummy_data.pt"  # Return a dummy file name
        window_type, window_size = self.MODEL_TO_WINDOW[self.transformer_model_name]
        return f"data_{self.transformer_model_name}_{window_type}_{window_size}_undersize{self.allow_undersize}.pt"

    def process(self):
        if self.transformer_model_name is None:
            return

        data_list = []
        window_method, window_size = self.MODEL_TO_WINDOW[self.transformer_model_name]
        for gene_id in tqdm(self.genome.gene_set):
            sequence = self.genome[gene_id]
            dna_selection = getattr(sequence, window_method)(
                window_size, allow_undersize=self.allow_undersize
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
        "downstream_300",
        "species_downstream_300",
        "species_upstream_1000",
        "upstream_1000",
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
    dataset = FungalUtrTransformerDataset(
        root="data/scerevisiae/fungal_utr_embed",
        genome=genome,
        transformer_model_name="downstream_window_3utr_300",
    )
