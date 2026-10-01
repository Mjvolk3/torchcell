"""Embedding dataset of random per-gene feature vectors for baselines/tests."""

# torchcell/datasets/random_embedding
# [[torchcell.datasets.random_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/random_embedding
# Test file: tests/torchcell/datasets/test_random_embedding.py

import logging
import os
import os.path as osp
from collections.abc import Callable
from typing import Any, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.sequence import ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import (
    SCerevisiaeGene,
    SCerevisiaeGenome,
)

log = logging.getLogger(__name__)


class RandomEmbeddingDataset(BaseEmbeddingDataset):
    """Embedding dataset assigning each gene a random feature vector."""

    # 1000 = random embedding size
    # We should remove this to make it more general
    MODEL_TO_WINDOW = {
        "random_6579": ("window", 6579, False),
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
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        batch_size: int = 100,
    ):
        """Set up the dataset, processing random embeddings if not yet cached.

        Args:
            root: Root directory for processed data.
            genome: Genome supplying the gene set to embed.
            model_name: Key into ``MODEL_TO_WINDOW`` selecting embedding size.
            transform: Optional runtime transform applied to each item.
            pre_transform: Optional transform applied before caching.
            batch_size: Number of genes processed per save chunk.
        """
        self.genome: SCerevisiaeGenome | ParsedGenome | None = genome
        self.model_name = model_name
        self.batch_size = batch_size
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)

        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @staticmethod
    def parse_genome(genome: SCerevisiaeGenome | None) -> ParsedGenome | None:
        """Return a ParsedGenome holding the genome's gene set, or None."""
        if genome is None:
            return None
        else:
            data = {}
            data["gene_set"] = genome.gene_set
            return ParsedGenome(**data)

    def initialize_model(self) -> Any:
        """Raise; random embeddings need no model initialization."""
        raise NotImplementedError(
            "initialize_model is not needed for RandomEmbeddingDataset"
        )

    @property
    def partial_path(self) -> str:
        """Path of the in-progress chunk file, beside (never at) the final store."""
        return osp.join(self.processed_dir, f"{self.model_name}.partial.pt")

    def process(self) -> None:
        """Generate random embeddings per gene and save them in batched chunks.

        Rows are drawn from a private ``torch.Generator`` seeded with 42, so the
        store is the seed-42 stream and the caller's global RNG is left untouched.
        Chunks are written to ``partial_path``; a chunk file left by an interrupted
        build is removed with a warning, since the build restarts at the first gene.
        """
        data_list = []
        (window_method, window_size, is_max_size) = self.MODEL_TO_WINDOW[
            cast(str, self.model_name)
        ]

        generator = torch.Generator().manual_seed(42)
        if os.path.exists(self.partial_path):
            log.warning(
                f"Removing partial chunk file {self.partial_path} left by an "
                "interrupted build; rebuilding from the first gene."
            )
            os.remove(self.partial_path)

        genome = cast(SCerevisiaeGenome, self.genome)
        for i, gene_id in tqdm(enumerate(genome.gene_set)):
            sequence = cast(SCerevisiaeGene, genome[gene_id])
            if len(sequence) <= window_size:
                cds_sequence = sequence.cds.seq
                embeddings = torch.rand(1, window_size, generator=generator)
                dna_selection = getattr(sequence, window_method)(len(cds_sequence))
                dna_window_dict = {self.model_name: dna_selection}
            else:
                dna_selection = getattr(sequence, window_method)(window_size)
                embeddings = torch.rand(1, window_size, generator=generator)
                dna_window_dict = {self.model_name: dna_selection}

            data = Data(id=gene_id, dna_windows=dna_window_dict)
            data.embeddings = {self.model_name: embeddings}
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Detach the tensors in the data object
            data = data.detach()

            data_list.append(data)

            if (i + 1) % self.batch_size == 0 or (i + 1) == len(genome.gene_set):
                # Merge the chunks this run already wrote, then consume the file
                if os.path.exists(self.partial_path):
                    existing_data = torch.load(self.partial_path, weights_only=False)
                    data_list = existing_data["data_list"] + data_list
                    os.remove(self.partial_path)
                if (i + 1) == len(genome.gene_set):
                    torch.save(self.collate(data_list), self.processed_paths[0])
                else:
                    torch.save({"data_list": data_list}, self.partial_path)
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
