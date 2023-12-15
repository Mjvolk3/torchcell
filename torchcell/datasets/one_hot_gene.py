# torchcell/datasets/one_hot_gene.py
# [[torchcell.datasets.one_hot_gene]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/one_hot_gene.py
# Test file: torchcell/datasets/test_one_hot_gene.py

from collections.abc import Callable
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.sequence import GeneSet, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class OneHotGeneDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {"one_hot_gene": None}

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        # Create a mapping from gene to its index
        self.genome = genome
        self.gene_to_index = {
            gene: idx for idx, gene in enumerate(self.genome.gene_set)
        }

        self.model_name = "one_hot_gene"
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        # HACK
        if self.model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_transformer()
                self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

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

    def one_hot_encode_gene(self, gene: str) -> torch.Tensor:
        """One-hot encode a gene based on its position in the gene set."""
        encoded = torch.zeros(len(self.gene_to_index))
        encoded[self.gene_to_index[gene]] = 1
        return encoded

    def process(self):
        # HACK
        if not self.model_name:
            return

        data_list = []

        for gene_id in tqdm(self.genome.gene_set):
            encoded_gene = self.one_hot_encode_gene(gene_id)

            # Create a Data object
            data = Data(id=gene_id, dna_windows={})
            data.embeddings = {self.model_name: encoded_gene.unsqueeze(0)}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    import os
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))

    genome = SCerevisiaeGenome()
    dataset = OneHotGeneDataset(
        root="data/scerevisiae/gene_one_hot_encoded", genome=genome
    )

    some_data = dataset[0]  # Should give you the first dataset item
    print(some_data)


if __name__ == "__main__":
    pass
