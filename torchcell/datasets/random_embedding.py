# torchcell/datasets/random_embedding
# [[torchcell.datasets.random_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/random_embedding
# Test file: tests/torchcell/datasets/test_random_embedding.py


import os
from collections.abc import Callable
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.sequence import ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class RandomEmbeddingDataset(BaseEmbeddingDataset):
    # 1000 = random embedding size
    MODEL_TO_WINDOW = {
        "random_1000": ("window", 1000, False),
        "random_100": ("window", 100, False),
        "random_10": ("window", 10, False),
        "random_1": ("window", 1, False),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = "random_1000",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        batch_size: int = 100,
    ):
        self.genome = genome
        self.model_name = model_name
        self.batch_size = batch_size
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)

        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self):
        raise NotImplementedError(
            "initialize_model is not needed for RandomEmbeddingDataset"
        )

    def process(self):
        data_list = []
        (window_method, window_size, is_max_size) = self.MODEL_TO_WINDOW[
            self.model_name
        ]

        torch.manual_seed(42)  # Set a fixed seed for reproducibility

        for i, gene_id in tqdm(enumerate(self.genome.gene_set)):
            sequence = self.genome[gene_id]
            if len(sequence) <= window_size:
                cds_sequence = sequence.cds.seq
                embeddings = torch.rand(1, window_size)  # Random values between 0 and 1
                dna_selection = getattr(sequence, window_method)(len(cds_sequence))
                dna_window_dict = {self.model_name: dna_selection}
            else:
                dna_selection = getattr(sequence, window_method)(window_size)
                embeddings = torch.rand(1, window_size)  # Random values between 0 and 1
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
    dataset = RandomEmbeddingDataset(
        root="data/scerevisiae/random_embedding",
        model_name="random_1",
        genome=genome,
        batch_size=100,
    )
    print(f"Random Embedding Dataset: {dataset}")
    some_data = dataset[genome.gene_set[42]]
    print(some_data)
