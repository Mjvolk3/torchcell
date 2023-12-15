# torchcell/datasets/protT5.py
# [[torchcell.datasets.protT5]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/protT5.py
# Test file: torchcell/datasets/test_protT5.py

import os
from collections.abc import Callable

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.models.protT5 import ProtT5
from torchcell.sequence import GeneSet, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class ProtT5Dataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "prot_t5_xl_uniref50_all": None,
        "prot_t5_xl_uniref50_no_dubious_uncharacterized": [
            "dubious",
            "uncharacterized",
        ],
        "prot_t5_xl_uniref50_no_dubious": ["Dubious"],
        "prot_t5_xl_uniref50_no_uncharacterized": ["Uncharacterized"],
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.genome = genome
        self.model_name = model_name
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        # self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = torch.load(
        #     self.processed_paths[0], map_location=self.device
        # )
        if self.model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_transformer()
                self.process()
            # HACK we send cpu because all data needs to be on cpu for lightning
            # lightning automatically moves
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location="cpu"
            )

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self) -> ProtT5:
        return ProtT5("prot_t5_xl_uniref50")

    def process(self):
        # HACK
        # self.transformer = self.initialize_model()
        if not self.model_name:
            return

        data_list = []

        exclude_classifications = self.MODEL_TO_WINDOW.get(self.model_name, None)
        for gene_id in tqdm(self.genome.gene_set):
            orf_classification = self.genome[gene_id].orf_classification[0]

            protein_sequence = str(self.genome[gene_id].protein.seq)

            if (
                exclude_classifications
                and orf_classification in exclude_classifications
            ):
                print(f"zeros for {gene_id}")
                embeddings = torch.zeros(1, 1024, dtype=torch.float32).to(self.device)
            else:
                embeddings = self.transformer.embed(
                    [protein_sequence], mean_embedding=True
                ).to(torch.float32)

            protein_data_dict = {self.model_name: protein_sequence}

            # Using 'dna_windows' for compatibility, but this might need a more general solution in the future
            data = Data(id=gene_id, dna_windows=protein_data_dict)
            data.embeddings = {self.model_name: embeddings}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )

    dataset = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_all",
    )

    dataset = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_no_dubious",
    )
