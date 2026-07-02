# torchcell/datasets/esm2
# [[torchcell.datasets.esm2]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/esm2
# Test file: tests/torchcell/datasets/test_esm2.py
"""Dataset of ESM2 protein embeddings for S. cerevisiae genes."""

import os
from collections.abc import Callable
from typing import Any, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.models.esm2 import Esm2
from torchcell.sequence import ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import (
    SCerevisiaeGene,
    SCerevisiaeGenome,
)


class Esm2Dataset(BaseEmbeddingDataset):
    """Embedding dataset that computes ESM2 protein embeddings per gene."""

    MODEL_TO_WINDOW = {
        "esm2_t6_8M_UR50D_all": ("esm2_t6_8M_UR50D", None),
        "esm2_t6_8M_UR50D_no_dubious_uncharacterized": (
            "esm2_t6_8M_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t6_8M_UR50D_no_dubious": ("esm2_t6_8M_UR50D", ["Dubious"]),
        "esm2_t6_8M_UR50D_no_uncharacterized": (
            "esm2_t6_8M_UR50D",
            ["Uncharacterized"],
        ),
        "esm2_t12_35M_UR50D_all": ("esm2_t12_35M_UR50D", None),
        "esm2_t12_35M_UR50D_no_dubious_uncharacterized": (
            "esm2_t12_35M_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t12_35M_UR50D_no_dubious": ("esm2_t12_35M_UR50D", ["Dubious"]),
        "esm2_t12_35M_UR50D_no_uncharacterized": (
            "esm2_t12_35M_UR50D",
            ["Uncharacterized"],
        ),
        "esm2_t30_150M_UR50D_all": ("esm2_t30_150M_UR50D", None),
        "esm2_t30_150M_UR50D_no_dubious_uncharacterized": (
            "esm2_t30_150M_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t30_150M_UR50D_no_dubious": ("esm2_t30_150M_UR50D", ["Dubious"]),
        "esm2_t30_150M_UR50D_no_uncharacterized": (
            "esm2_t30_150M_UR50D",
            ["Uncharacterized"],
        ),
        "esm2_t33_650M_UR50D_all": ("esm2_t33_650M_UR50D", None),
        "esm2_t33_650M_UR50D_no_dubious_uncharacterized": (
            "esm2_t33_650M_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t33_650M_UR50D_no_dubious": ("esm2_t33_650M_UR50D", ["Dubious"]),
        "esm2_t33_650M_UR50D_no_uncharacterized": (
            "esm2_t33_650M_UR50D",
            ["Uncharacterized"],
        ),
        "esm2_t36_3B_UR50D_all": ("esm2_t36_3B_UR50D", None),
        "esm2_t36_3B_UR50D_no_dubious_uncharacterized": (
            "esm2_t36_3B_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t36_3B_UR50D_no_dubious": ("esm2_t36_3B_UR50D", ["Dubious"]),
        "esm2_t36_3B_UR50D_no_uncharacterized": (
            "esm2_t36_3B_UR50D",
            ["Uncharacterized"],
        ),
        "esm2_t48_15B_UR50D_all": ("esm2_t48_15B_UR50D", None),
        "esm2_t48_15B_UR50D_no_dubious_uncharacterized": (
            "esm2_t48_15B_UR50D",
            ["dubious", "uncharacterized"],
        ),
        "esm2_t48_15B_UR50D_no_dubious": ("esm2_t48_15B_UR50D", ["Dubious"]),
        "esm2_t48_15B_UR50D_no_uncharacterized": (
            "esm2_t48_15B_UR50D",
            ["Uncharacterized"],
        ),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ):
        """Configure the genome and model, then process embeddings if needed."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.genome: SCerevisiaeGenome | ParsedGenome | None = genome
        self.model_name = model_name
        self.exclude_classifications = self.MODEL_TO_WINDOW[cast(str, self.model_name)][
            1
        ]
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

        if self.model_name:
            if not os.path.exists(self.processed_paths[0]):
                self.transformer = self.initialize_model()
                self.process()
            self.data, self.slices = torch.load(
                self.processed_paths[0], map_location="cpu"
            )

    @staticmethod
    def parse_genome(genome: SCerevisiaeGenome | None) -> ParsedGenome | None:
        """Return a ParsedGenome holding the genome's gene set, or None."""
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self) -> Esm2:
        """Return the ESM2 model corresponding to the selected model name."""
        return Esm2(model_name=self.MODEL_TO_WINDOW[cast(str, self.model_name)][0])

    def process(self) -> None:
        """Compute ESM2 embeddings for each gene and save the processed dataset."""
        self.transformer = self.initialize_model()
        if not self.model_name:
            return

        data_list = []

        genome = cast(SCerevisiaeGenome, self.genome)
        for gene_id in tqdm(genome.gene_set):
            orf_classification = cast(
                SCerevisiaeGene, genome[gene_id]
            ).orf_classification[0]
            protein_sequence = str(cast(Any, genome[gene_id]).protein.seq)

            if (
                self.exclude_classifications
                and orf_classification in self.exclude_classifications
            ):
                print(f"zeros for {gene_id}")
                embeddings = (
                    torch.zeros(
                        cast(Any, self.transformer.model).config.hidden_size,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )  # Ensure embeddings has shape (1, hidden_size)
            else:
                embeddings = self.transformer.embed(
                    [protein_sequence], mean_embedding=True
                )

            embeddings = (
                embeddings.cpu().squeeze()
            )  # Remove extra dimensions if necessary

            protein_data_dict = {self.model_name: protein_sequence}

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
        genome_root=osp.join(cast(str, DATA_ROOT), "data/sgd/genome"), overwrite=True
    )

    model_names = [key for key in Esm2Dataset.MODEL_TO_WINDOW.keys()]

    for model_name in model_names:
        dataset = Esm2Dataset(
            root=osp.join(cast(str, DATA_ROOT), "data/scerevisiae/esm2_embedding_test"),
            genome=genome,
            model_name=model_name,
        )
