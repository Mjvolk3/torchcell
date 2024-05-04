# torchcell/datasets/codon_language_model
# [[torchcell.datasets.codon_language_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/codon_language_model
# Test file: tests/torchcell/datasets/test_codon_language_model.py

from calm import CaLM
import os
from collections.abc import Callable
from typing import Optional
import torch
from pydantic import validator
from torch_geometric.data import Data
from tqdm import tqdm
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.sequence import GeneSet, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class CalmDataset(BaseEmbeddingDataset):
    # 3072 = 1024 * 3
    MODEL_TO_WINDOW = {"calm": ("window", 3072, False)}

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = "calm",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        batch_size: int = 100,
    ):
        self.genome = genome
        self.model_name = model_name
        self.model = self.initialize_model()
        self.batch_size = batch_size
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        if not os.path.exists(self.processed_paths[0]):
            self.model = self.initialize_model()
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_model(self) -> CaLM:
        return CaLM()

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def process(self):
        data_list = []
        (window_method, window_size, is_max_size) = self.MODEL_TO_WINDOW[
            self.model_name
        ]

        for i, gene_id in tqdm(enumerate(self.genome.gene_set)):
            sequence = self.genome[gene_id]
            if len(sequence) <= window_size:
                assert len(str(sequence.cds.seq)) % 3 == 0
                cds_sequence = sequence.cds.seq
                embeddings = self.model.embed_sequence(str(cds_sequence))
                dna_selection = getattr(sequence, window_method)(len(cds_sequence))
                dna_window_dict = {self.model_name: dna_selection}
            else:
                dna_selection = getattr(sequence, window_method)(window_size)
                assert len(dna_selection.seq) % 3 == 0
                embeddings = self.model.embed_sequence(dna_selection.seq)
                dna_window_dict = {self.model_name: dna_selection}

            data = Data(id=gene_id, dna_windows=dna_window_dict)
            data.embeddings = {self.model_name: embeddings}
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # Detach the tensors in the data object
            data = data.detach()
            
            data_list.append(data)

            if (i + 1) % self.batch_size == 0 or (i + 1) == len(self.genome.gene_set):
                # Load existing data from the file if it exists
                if os.path.exists(self.processed_paths[0]):
                    existing_data = torch.load(self.processed_paths[0])
                    existing_data_list = existing_data.get("data_list", [])
                    data_list = existing_data_list + data_list
                if (i + 1) == len(self.genome.gene_set):
                    torch.save(self.collate(data_list), self.processed_paths[0])

                else:
                    # Save the updated data back to the file
                    torch.save({"data_list": data_list}, self.processed_paths[0])
                data_list = []

if __name__ == "__main__":
    genome = SCerevisiaeGenome()
    dataset = CalmDataset(
        root="data/scerevisiae/calm_embedding", genome=genome, batch_size=100
    )
    print(f"Calm Dataset: {dataset}")
    some_data = dataset[genome.gene_set[42]]
    print(some_data)
