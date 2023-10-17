import os
from collections.abc import Callable

import torch
from tqdm import tqdm
from torch_geometric.data import Data

from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.models.protT5 import ProtT5
from torchcell.sequence import GeneSet, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class ProtT5Dataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {"prot_t5_xl_uniref50": None}

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.genome = genome
        self.model_name = model_name
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        if self.model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_transformer()
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

    def initialize_model(self) -> ProtT5 | None:
        if self.model_name:
            assert (
                self.model_name in ProtT5.VALID_MODEL_NAMES
            ), f"{self.model_name} not in valid model names."
            return ProtT5(self.model_name)
        return None

    def process(self):
        if not self.model_name:
            return

        data_list = []
        for gene_id in tqdm(self.genome.gene_set):
            protein_sequence = str(self.genome[gene_id].protein.seq)
            embeddings = self.transformer.embed([protein_sequence], mean_embedding=True)

            protein_data_dict = {self.model_name: protein_sequence}

            data = Data(id=gene_id, protein_data=protein_data_dict)
            data.embeddings = {self.model_name: embeddings}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os.path as osp

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )

    model_name = "prot_t5_xl_uniref50"

    dataset = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
        genome=genome,
        model_name=model_name,
    )
    print(f"Dataset for {model_name}: {dataset}")
