# tests/torchcell/conftest.py
# [[tests.torchcell.conftest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/conftest.py
"""Shared synthetic fixtures for the ``torchcell`` test tree.

Everything here is tiny and built in memory, so no fixture needs ``DATA_ROOT``:

* the Cell Graph Transformer fixtures (``cell_graph``, ``batch``), lifted from
  ``tests/torchcell/models/test_equivariant_cell_graph_transformer.py`` so the model and
  trainer tests share one graph instead of copying the builder;
* ``fake_txn``, a dict-backed stand-in for the interned LMDB write transaction the
  dataset loaders take (``tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py``
  keeps its own copy for its module-level helpers);
* ``dcell_graph`` / ``dcell_batch``, a three-term GO hierarchy over four genes in the
  layout ``torchcell.models.dcell.DCell`` reads.
"""

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

# Sizes of the synthetic CGT graph. Test modules cannot import a conftest (the test
# tree has no __init__.py, so a relative import has no parent package); they restate
# the sizes they assert on and take the graph itself from the fixtures below.
CGT_GENE_NUM = 8
CGT_NUM_REACTIONS = 4
CGT_NUM_METABOLITES = 3


def make_cell_graph() -> HeteroData:
    """Tiny cell_graph with gene-gene, gpr, and rmr edges."""
    cg = HeteroData()
    cg["gene"].num_nodes = CGT_GENE_NUM
    cg["reaction"].num_nodes = CGT_NUM_REACTIONS
    cg["metabolite"].num_nodes = CGT_NUM_METABOLITES

    # A gene-gene edge type (unused when graph_reg_lambda == 0).
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )

    # gene -> gpr -> reaction: genes {0,1}->r0, {2}->r1, {3,4}->r2, {5}->r3.
    cg["gene", "gpr", "reaction"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 3]], dtype=torch.long
    )

    # metabolite <- reaction (hyperedge): m0<-{r0,r1}, m1<-{r2}, m2<-{r3,r0}.
    cg["metabolite", "reaction", "metabolite"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 2], [0, 1, 2, 3, 0]], dtype=torch.long
    )
    return cg


def make_batch() -> HeteroData:
    """Tiny perturbation batch: 3 genotypes with varying perturbed gene counts."""
    batch = HeteroData()
    # sample 0 perturbs genes {1,2}; sample 1 perturbs {3}; sample 2 perturbs {0,4,5}
    batch["gene"].perturbation_indices = torch.tensor(
        [1, 2, 3, 0, 4, 5], dtype=torch.long
    )
    batch["gene"].perturbation_indices_batch = torch.tensor(
        [0, 0, 1, 2, 2, 2], dtype=torch.long
    )
    return batch


@pytest.fixture
def cell_graph() -> HeteroData:
    """Fresh synthetic cell graph per test (models mutate nothing, but stay isolated)."""
    return make_cell_graph()


@pytest.fixture
def batch() -> HeteroData:
    """Fresh synthetic perturbation batch per test."""
    return make_batch()


class FakeTxn:
    """A dict-backed stand-in for the interned LMDB write transaction."""

    def __init__(self) -> None:
        """Start empty."""
        self.store: dict[bytes, bytes] = {}

    def get(self, key: bytes) -> bytes | None:
        """The stored bytes for ``key``, or ``None``."""
        return self.store.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        """Store ``value`` under ``key``."""
        self.store[key] = value


@pytest.fixture
def fake_txn() -> FakeTxn:
    """An empty in-memory transaction."""
    return FakeTxn()


# DCell: root term 0 (stratum 0) with children 1 and 2 (stratum 1); term 1 annotates
# genes {0, 1}, term 2 annotates genes {2, 3}, the root annotates none directly. The
# ``go_gene_strata_state`` rows are [go_idx, gene_idx, stratum, state], one row per
# (term, gene) annotation, and the batch repeats the template once per sample with the
# state column flipped to 0 for perturbed genes.
DCELL_GENES = 4
DCELL_TERMS = 3
_DCELL_TEMPLATE = torch.tensor(
    [[1, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1], [2, 3, 1, 1]], dtype=torch.long
)


def make_dcell_graph() -> HeteroData:
    """The ontology template ``DCell.__init__`` reads."""
    data = HeteroData()
    data["gene"].num_nodes = DCELL_GENES
    go = data["gene_ontology"]
    go.num_nodes = DCELL_TERMS
    go.strata = torch.tensor([0, 1, 1], dtype=torch.long)
    go.stratum_to_terms = {0: torch.tensor([0]), 1: torch.tensor([1, 2])}
    go.term_gene_counts = torch.tensor([0, 2, 2], dtype=torch.long)
    go.go_gene_strata_state = _DCELL_TEMPLATE.clone()
    # child -> parent
    data["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[1, 2], [0, 0]], dtype=torch.long
    )
    return data


def make_dcell_batch(perturbed: list[list[int]]) -> HeteroData:
    """A batch of ``len(perturbed)`` samples; each inner list names the knocked-out genes."""
    batch = HeteroData()
    n = len(perturbed)
    batch["gene"].x = torch.zeros(n * DCELL_GENES, 1)
    batch["gene"].batch = torch.arange(n).repeat_interleave(DCELL_GENES)
    rows = []
    for genes in perturbed:
        sample = _DCELL_TEMPLATE.clone()
        for gene in genes:
            sample[sample[:, 1] == gene, 3] = 0
        rows.append(sample)
    go = batch["gene_ontology"]
    go.go_gene_strata_state = torch.cat(rows, dim=0)
    go.go_gene_strata_state_ptr = torch.arange(n + 1) * len(_DCELL_TEMPLATE)
    return batch


@pytest.fixture
def dcell_graph() -> HeteroData:
    """Three-term GO hierarchy over four genes."""
    return make_dcell_graph()


@pytest.fixture
def dcell_batch() -> HeteroData:
    """Two samples: gene 0 knocked out; genes 2 and 3 knocked out."""
    return make_dcell_batch([[0], [2, 3]])


# The legacy DCell trainers (``torchcell.trainers.dcell_regression`` and
# ``dcell_regression_slim``) take ``models={"dcell": ..., "dcell_linear": ...}`` where
# ``dcell(batch)`` returns one hidden tensor per GO term and ``dcell_linear`` maps each
# to a ``[B, 1]`` prediction. The pair below does that with no random weights, so every
# prediction is a closed form of the knockouts in a ``make_dcell_batch`` batch:
# ``GO:1`` = intact genes of term 1, ``GO:2`` = intact genes of term 2, and ``GO:ROOT``
# = ``GO:1 - GO:2`` (the root annotates no gene, so it gets a feature of its own that is
# not proportional to the subsystem mean). Each feature is scaled by a learnable
# per-term scalar (1.0) and passed through an identity ``Linear(1, 1)`` head.
DCELL_TERM_NAMES = ("GO:ROOT", "GO:1", "GO:2")


class DCellCountSubsystems(torch.nn.Module):
    """Per-term intact-gene counts from ``go_gene_strata_state``, times ``scale``."""

    def __init__(self) -> None:
        """One learnable scale per term, initialized to 1."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(DCELL_TERMS))

    def forward(self, batch: HeteroData) -> dict[str, torch.Tensor]:
        """Map each term name to its ``[B, 1]`` scaled count (root = GO:1 - GO:2)."""
        go = batch["gene_ontology"]
        rows = go.go_gene_strata_state
        ptr = go.go_gene_strata_state_ptr
        n = len(ptr) - 1
        sample = torch.arange(n).repeat_interleave(ptr.diff())
        counts = torch.zeros(n, DCELL_TERMS).index_put_(
            (sample, rows[:, 0]), rows[:, 3].float(), accumulate=True
        )
        counts[:, 0] = counts[:, 1] - counts[:, 2]
        scaled = counts * self.scale
        return {name: scaled[:, i : i + 1] for i, name in enumerate(DCELL_TERM_NAMES)}


class DCellIdentityHeads(torch.nn.Module):
    """One ``Linear(1, 1)`` per term, initialized to weight 1 and bias 0."""

    def __init__(self) -> None:
        """Three identity heads, one per entry of ``DCELL_TERM_NAMES``."""
        super().__init__()
        heads = [torch.nn.Linear(1, 1) for _ in DCELL_TERM_NAMES]
        for head in heads:
            torch.nn.init.ones_(head.weight)
            torch.nn.init.zeros_(head.bias)
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, hidden: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply each term's head to that term's hidden tensor."""
        return {
            name: self.heads[i](hidden[name]) for i, name in enumerate(DCELL_TERM_NAMES)
        }


def make_dcell_regression_batch() -> HeteroData:
    """Three samples for the DCell trainers, with the two top-level fields they read.

    Knockouts {0, 1, 2, 3}, {0} and {2} give intact counts GO:1 = [0, 1, 2] and
    GO:2 = [0, 2, 1], so GO:ROOT = [0, -1, 1]; ``fitness`` is [1.0, 0.0, 0.5] and
    ``batch`` is the gene-level batch vector (its last entry + 1 is the batch size, 3).
    """
    batch = make_dcell_batch([[0, 1, 2, 3], [0], [2]])
    batch.batch = batch["gene"].batch
    batch.fitness = torch.tensor([1.0, 0.0, 0.5])
    return batch


# Embedding datasets (``torchcell/datasets/{esm2,protT5,nucleotide_transformer,
# fungal_up_down_transformer,codon_language_model,random_embedding}.py``) read only
# ``genome.gene_set`` and ``genome[gene_id]``. ``embedding_genome`` builds REAL
# ``SCerevisiaeGene`` objects (so every window method is the production one) over one
# 6,100 nt chromosome I drawn from ``random.Random(1809)``, with a dict-backed stand-in
# for the gffutils database. GFF coordinates are 1-based inclusive:
#
# * ``YAL001W`` ``+`` 101..112, Verified, protein ``MKPG*``, CDS = ``chrI[100:112]``;
# * ``YAL002C`` ``-`` 21..32, Dubious, protein ``MSK*``, CDS ``ATGGCCTAA`` (9 nt, a
#   spliced CDS shorter than the 12 nt locus);
# * ``YAL003W`` ``+`` 2001..5100, Uncharacterized, protein ``MKKS*``,
#   CDS = ``chrI[2000:5100]`` (3,100 nt).
EMBEDDING_CHROMOSOME_LENGTH = 6100
EMBEDDING_GENES: tuple[tuple[str, str, int, int, str, str], ...] = (
    ("YAL001W", "+", 101, 112, "Verified", "MKPG*"),
    ("YAL002C", "-", 21, 32, "Dubious", "MSK*"),
    ("YAL003W", "+", 2001, 5100, "Uncharacterized", "MKKS*"),
)


class _EmbeddingStubDb:
    """The two gffutils ``FeatureDB`` calls ``SCerevisiaeGene`` makes."""

    def __init__(self, features: dict[str, Any]) -> None:
        """Index gffutils features by id."""
        self.features = features

    def __getitem__(self, key: str) -> Any:
        """The feature with id ``key``."""
        return self.features[key]

    def region(
        self, region: tuple[str, int, int], completely_within: bool
    ) -> list[Any]:
        """Every feature on ``region[0]`` lying inside ``[region[1], region[2]]``."""
        chrom, start, end = region
        return [
            f
            for f in self.features.values()
            if f.chrom == chrom and f.start >= start and f.end <= end
        ]


class EmbeddingStubGenome:
    """``gene_set`` plus ``__getitem__`` over real ``SCerevisiaeGene`` objects."""

    def __init__(self) -> None:
        """Build the chromosome, the three genes and their FASTA records."""
        import random

        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from gffutils import Feature

        from torchcell.sequence.data import GeneSet
        from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGene

        rng = random.Random(1809)
        self.chromosome = "".join(
            rng.choice("ACGT") for _ in range(EMBEDDING_CHROMOSOME_LENGTH)
        )
        features = {
            gid: Feature(
                seqid="chrI",
                source="SGD",
                featuretype="gene",
                start=start,
                end=end,
                strand=strand,
                attributes={"ID": [gid], "orf_classification": [orf]},
                id=gid,
            )
            for gid, strand, start, end, orf, _ in EMBEDDING_GENES
        }
        cds = {
            "YAL001W": self.chromosome[100:112],
            "YAL002C": "ATGGCCTAA",
            "YAL003W": self.chromosome[2000:5100],
        }
        fasta_dna = {"NC_001133": SeqRecord(Seq(self.chromosome), id="NC_001133")}
        db = _EmbeddingStubDb(features)
        self.genes = {
            gid: SCerevisiaeGene(
                id=gid,
                db=db,
                fasta_dna=fasta_dna,
                fasta_protein={gid: SeqRecord(Seq(protein), id=gid)},
                fasta_cds={gid: SeqRecord(Seq(cds[gid]), id=gid)},
                chr_to_nc={1: "NC_001133"},
                chromosome_lengths={1: EMBEDDING_CHROMOSOME_LENGTH},
            )
            for gid, _, _, _, _, protein in EMBEDDING_GENES
        }
        self.gene_set = GeneSet(self.genes)

    def __getitem__(self, gene_id: str) -> Any:
        """The ``SCerevisiaeGene`` for ``gene_id``."""
        return self.genes[gene_id]


@pytest.fixture
def embedding_genome() -> EmbeddingStubGenome:
    """Three real ``SCerevisiaeGene`` objects on a 6,100 nt synthetic chromosome I."""
    return EmbeddingStubGenome()
