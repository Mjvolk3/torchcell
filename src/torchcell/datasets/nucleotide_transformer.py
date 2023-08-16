# src/torchcell/datasets/nucleotide_transformer.py
# [[src.torchcell.datasets.nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/nucleotide_transformer.py
# Test file: tests/torchcell/datasets/test_nucleotide_transformer.py
import os
from typing import Callable, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

os.makedirs("data/scerevisiae/nucleotide_transformer_embed", exist_ok=True)


class NucleotideTransformerDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "nt_window_5979_max": ("window", 5979, True),
        "nt_window_5979": ("window", 5979, False),
        "nt_window_3utr_5979": ("window_3utr", 5979, False),
        "nt_window_3utr_5979_undersize": ("window_3utr", 5979, True),
        "nt_window_5utr_5979": ("window_5utr", 5979, False),
        "nt_window_5utr_5979_undersize": ("window_5utr", 5979, True),
        "nt_window_3utr_300": ("window_3utr", 300, False),
        "nt_window_3utr_300_undersize": ("window_3utr", 300, True),
        "nt_window_5utr_1000": ("window_5utr", 1000, False),
        "nt_window_5utr_1000_undersize": ("window_5utr", 1000, True),
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
            print(self.processed_paths[0])
            if not os.path.exists(self.processed_paths[0]):
                # Initialize the language model
                self.transformer = self.initialize_transformer()
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_transformer(self):
        if self.transformer_model_name:
            return NucleotideTransformer()
        return None

    def process(self):
        if self.transformer_model_name is None:
            return

        data_list = []
        window_method, window_size, flag = self.MODEL_TO_WINDOW[
            self.transformer_model_name
        ]

        for gene_id in tqdm(self.genome.gene_set):
            sequence = self.genome[gene_id]

            if "utr" in window_method:
                dna_selection = getattr(sequence, window_method)(
                    window_size, allow_undersize=flag
                )
            else:
                dna_selection = getattr(sequence, window_method)(
                    window_size, is_max_size=flag
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
        "nt_window_5979",
        # "nt_window_5979_max",
        # "nt_window_3utr_5979",
        # "nt_window_3utr_5979_undersize",
        # "nt_window_5utr_5979",
        # "nt_window_5utr_5979_undersize",
        # "nt_window_3utr_300",
        # "nt_window_3utr_300_undersize",
        # "nt_window_5utr_1000",
        # "nt_window_5utr_1000_undersize"
    ]
    datasets = []
    for model_name in model_names:
        dataset = NucleotideTransformerDataset(
            root="data/scerevisiae/nucleotide_transformer_embed",
            genome=genome,
            transformer_model_name=model_name,
        )
        datasets.append(dataset)
        print(f"Dataset for {model_name}: {dataset}")
    print(dataset)
