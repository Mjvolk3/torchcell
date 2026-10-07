# torchcell/datasets/genslm.py
# [[torchcell.datasets.genslm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/genslm.py
# Test file: tests/torchcell/datasets/test_genslm.py

"""Dataset producing GenSLM codon language-model embeddings for the genes of any
genome whose genes expose a coding sequence (yeast S288C and the bacterial
assemblies alike).
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.models.genslm import GENSLM_MODELS, GENSLM_SEQ_LENGTH
from torchcell.sequence import ParsedGenome

if TYPE_CHECKING:
    from torchcell.models.genslm import GenSLM

log = logging.getLogger(__name__)


class CdsGene(Protocol):
    """A gene with a spliced coding sequence record, or ``None`` for an RNA gene."""

    @property
    def cds(self) -> Any:
        """The spliced CDS as a ``SeqRecord``, or ``None`` when the gene has none."""
        ...


class CdsGenome(Protocol):
    """``gene_set`` plus ``__getitem__`` to a :class:`CdsGene`."""

    @property
    def gene_set(self) -> Any:
        """The sorted gene ids of the genome."""
        ...

    def __getitem__(self, gene_id: str) -> CdsGene | None:
        """The gene for ``gene_id``, or ``None`` when the genome has no such gene."""
        ...


class GenSLMDataset(BaseEmbeddingDataset):
    """Embedding dataset that runs a GenSLM foundation model over gene CDS sequences.

    GenSLM tokenizes codons, so each gene embeds its spliced CDS (in frame from the
    start codon) truncated to the first 6,144 nt (2,048 codons, the model context)
    when longer. ``dna_windows`` stores exactly the string embedded and
    ``embeddings`` the masked mean of the last hidden layer.

    A gene whose ``cds`` is ``None`` (an rRNA, tRNA or other RNA gene) has no codon
    representation and is left out of the store; its id is written to
    ``processed/<model_name>.no_cds.json`` so the omission is on disk, not only in
    the log.
    """

    #: 6144 = 2048 codons * 3: the most CDS nucleotides embedded per gene.
    MODEL_TO_WINDOW = {
        model_id: ("cds", GENSLM_SEQ_LENGTH * 3) for model_id in GENSLM_MODELS
    }

    def __init__(
        self,
        root: str,
        genome: CdsGenome,
        model_name: str = "genslm_25M_patric",
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        batch_size: int = 16,
        weights_dir: str | None = None,
    ) -> None:
        """Parse the genome and build (or load) the embedding store.

        Args:
            root: Dataset root; the store is ``processed/<model_name>.pt``.
            genome: Any genome whose genes expose ``cds``.
            model_name: One of ``GENSLM_MODELS``.
            transform: Optional transform applied at access time.
            pre_transform: Optional transform applied before saving.
            batch_size: Sequences per forward pass (padded to the longest).
            weights_dir: Checkpoint directory; defaults to ``$DATA_ROOT/models/genslm``.
        """
        self.genome: CdsGenome | ParsedGenome | None = genome
        self.model_name = model_name
        # LAZY: the backbone is built inside process(), which PyG calls from
        # super().__init__() only when the store is absent; a store on disk loads
        # without touching the weights (offline nodes, no GPU).
        self.model: GenSLM | None = None
        self.batch_size = batch_size
        self.weights_dir = weights_dir
        super().__init__(root, self.model_name, transform, pre_transform)
        self.genome = self.parse_genome(genome)
        del genome

    def initialize_model(self) -> GenSLM:
        """Load the named GenSLM model from the verified checkpoint."""
        from torchcell.models.genslm import GenSLM

        return GenSLM(cast(str, self.model_name), weights_dir=self.weights_dir)

    @staticmethod
    def parse_genome(genome: CdsGenome | None) -> ParsedGenome | None:
        """Reduce a genome to its gene set (what a loaded dataset still needs)."""
        if genome is None:
            return None
        return ParsedGenome(gene_set=genome.gene_set)

    @property
    def no_cds_path(self) -> str:
        """Where the ids of genes without a CDS are recorded."""
        return osp.join(self.processed_dir, f"{self.model_name}.no_cds.json")

    def process(self) -> None:
        """Embed every gene with a CDS and save the processed data list."""
        if self.model is None:
            self.model = self.initialize_model()
        model = self.model
        model_name = cast(str, self.model_name)
        # 6,144 is a multiple of 3, so truncation keeps an in-frame CDS in frame; the
        # model's codon tokenizer refuses a CDS that is not in frame to begin with.
        _, window_size = self.MODEL_TO_WINDOW[model_name]
        genome = cast(CdsGenome, self.genome)

        windows: dict[str, str] = {}
        no_cds: list[str] = []
        for gene_id in genome.gene_set:
            gene = genome[gene_id]
            if gene is None:
                raise KeyError(f"{gene_id} is in gene_set but the genome has no gene")
            cds = gene.cds
            if cds is None:
                no_cds.append(gene_id)
                continue
            windows[gene_id] = str(cds.seq)[:window_size]
        if no_cds:
            log.warning(
                "%s: %d of %d genes have no CDS and are left out (ids in %s)",
                model_name,
                len(no_cds),
                len(genome.gene_set),
                self.no_cds_path,
            )
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.no_cds_path, "w") as handle:
            json.dump(no_cds, handle, indent=2)

        data_list: list[Data] = []
        gene_ids = list(windows)
        for start in tqdm(range(0, len(gene_ids), self.batch_size)):
            batch_ids = gene_ids[start : start + self.batch_size]
            embeddings = model.embed(
                [windows[g] for g in batch_ids], mean_embedding=True
            ).cpu()
            for row, gene_id in enumerate(batch_ids):
                data = Data(id=gene_id, dna_windows={model_name: windows[gene_id]})
                data.embeddings = {model_name: embeddings[row : row + 1]}
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data.detach())

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    from dotenv import load_dotenv

    from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = PPutidaKT2440Genome(
        genome_root=osp.join(data_root, "data/pputida/kt2440/genome")
    )
    dataset = GenSLMDataset(
        root=osp.join(data_root, "data/pputida/kt2440/genslm_embedding"),
        genome=genome,
        model_name="genslm_25M_patric",
    )
    print(dataset)
    print(dataset["PP_0001"])
