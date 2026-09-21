# experiments/028-gene-essentiality/scripts/dcell_style_baseline.py
# [[experiments.028-gene-essentiality.scripts.dcell_style_baseline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/dcell_style_baseline

r"""DCell on the gene-held-out essentiality splits: the published input and a shared one.

DCell (Ma et al. 2018; ``torchcell/models/dcell.py``) is one small subsystem per GO term,
wired along the GO DAG from the leaves to the root. A term's input is the states of its
child terms concatenated with the perturbation states of the genes annotated to it, the
root is read by a linear head, and every other term carries an auxiliary linear head whose
loss enters at weight alpha = 0.3. The gene input is a learned weight per gene per term.
Here each instance deletes ONE gene and the split is BY GENE, so the columns of a held-out
gene receive no gradient; the published input can place an unseen gene only through its
random initial columns. This script measures that, beside an adaptation that can
generalize.

Arms, all scored by held-out AUROC:

- ``dcell_style_shared``: DCell's hierarchy, subsystems, root head and auxiliary heads,
  with each term's gene input ONE shared weight vector applied to "the deleted gene is
  annotated to this term". The input is the gene's propagated GO multi-hot, each column
  entering at its own term and routed up the DAG.
- ``dcell_published_input``: the same network with DCell's own input, a weight vector per
  (gene, term) annotation pair.
- ``dcell_style_shared_noaux``: the shared arm with the auxiliary losses off (alpha = 0).

Protocol, matched to ``graph_baselines.py --arms go400`` (the comparison partner is its
GCN on the GO multi-hot over the union of nine gene graphs): the same split files, the
same GO source (``$DATA_ROOT/data/scerevisiae/go_term_hot/processed/go.pt``, the file
``embedding("go", ...)`` reads: propagated through ``is_a`` and ``part_of``, all three
namespaces, ten or more genes per term, 2,362 columns), class-weighted binary cross
entropy, Adam at 1e-3, 400 epochs, full batch, the epoch selected by a five-epoch moving
average of the validation AUROC, the held-out genes scored once at that epoch, seeds from
2700, and the per-gene held-out logits written per seed.

The hierarchy is the 2,362 kept terms, the three namespace roots and one root above them
(``GO:ROOT``, as torchcell's GO graph joins the namespaces, so there is one root to read).
Each term is connected to its nearest kept ancestors through ``is_a`` and ``part_of``
(the propagation ``go_term_hot.py`` uses); the kept set is closed under ancestry, because a
propagated ancestor carries every gene its descendant does, so these are the direct GO
parents, with an edge dropped when its parent is an ancestor of another parent.

What differs from the published DCell, and why:

- The subsystem width is fixed (``--width``) instead of max(20, 0.3 x genes under the
  term), so a stratum is one grouped matrix product rather than 2,366 Python modules.
- Batch normalization statistics are taken over the TRAINING genes in every forward pass
  and applied to all genes, so the training and scoring passes compute the same function
  and no running average is read (full batch makes this exact).
- A term no training gene is annotated to has an untrained subsystem whose constant
  training state has zero variance; a held-out gene lighting it would be divided by
  sqrt(eps). Such columns are zeroed for every gene (``train-support mask``), a rule that
  reads the training genes' annotations and no label.
- The auxiliary loss is alpha times the MEAN over the non-root terms, as
  ``torchcell/losses/dcell.py`` has it (the paper sums).
- Weights start at PyTorch's uniform +-1/sqrt(fan_in) rather than DCell's +-0.001, which
  under a variance floor of 1e-5 leaves the first strata unnormalized.

READS, fixed before any run:

(a) ``--read released``: the released split, six seeds 2700 to 2705; seed mean and sd.
(b) ``--read folds``: the five folds, seed 2700 per fold, scored on the 195 released
    held-out genes. The ensemble is the per-gene MEAN LOGIT over the folds, the rule of
    ``cv_ensemble_readout.py`` and of ``graph_baselines.py --arms go_folds`` that produced
    the GCN's 0.9176, with the same 2,000-draw gene bootstrap (seed 0). The mean of the
    per-fold standardized logits (the ``family_ensembles.py`` rule) is written beside it.
(c) ``--read resplits``: the ten repeated outer splits, seeds 2700 to 2702; per-split seed
    mean, and the paired difference against the GCN on GO on the same split
    (``results/graph_baselines_go400_fcl_resplit{r}.json``, row ``GO (400 epochs)``):
    mean, sd over splits and sign count. No p-value, because the test blocks overlap.

``--read tune`` scores eight configurations of the shared arm (width 8 or 20, the two
subsystem orders, weight decay 0 or 1e-3) by VALIDATION AUROC on the released split
(seeds 2700 to 2702) and never reads a held-out gene. The configuration with the highest
validation mean is then passed to the three reads for every arm. The repeated outer splits
redraw their test genes from all labeled genes, so some of their held-out genes sat in the
released validation set that chose the configuration.

Every read trains its (arm, split, seed) cells over a process pool (``--workers`` x
``--threads``, at most 16 threads in all); the cells are independent and seeded, so the
pool changes wall time only.

    PYTHONPATH=$PWD python experiments/028-gene-essentiality/scripts/dcell_style_baseline.py \
        --read released
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from goatools.obo_parser import GODag, GOTerm
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score
from torch import nn

from torchcell.datasets.go_term_hot import GOTermEmbeddingDataset

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLITS = osp.join(RESULTS, "splits")
GO_DIR = osp.join(DATA_ROOT, "data/scerevisiae/go_term_hot/processed")
RELEASED = "ess_fcl_val0.2_s0.json.gz"
ROOT_ID = "GO:ROOT"
EPOCHS = 400
LR = 1e-3
ALPHA = 0.3
BN_EPS = 1e-5
N_BOOTSTRAP = 2000
ARMS = ("dcell_style_shared", "dcell_published_input", "dcell_style_shared_noaux")
GCN_ROW = "GO (400 epochs)"
TUNE_WIDTHS = (8, 20)
TUNE_ORDERS = ("tanh_bn", "bn_tanh")
TUNE_WEIGHT_DECAYS = (0.0, 1e-3)


class Config(BaseModel):
    """The knobs one configuration fixes; everything else is a module constant."""

    model_config = ConfigDict(extra="forbid")
    width: int = 8
    rule: str = "fixed"
    order: str = "tanh_bn"
    weight_decay: float = 0.0
    epochs: int = EPOCHS
    lr: float = LR


class HierarchyStats(BaseModel):
    """The GO hierarchy as built: sizes that fix the architecture."""

    model_config = ConfigDict(extra="forbid")
    n_terms: int
    n_kept_terms: int
    n_edges: int
    n_redundant_edges_dropped: int
    n_strata: int
    terms_per_stratum: list[int]
    max_children: int
    n_labeled_genes: int
    n_annotation_pairs: int


class ArmRow(BaseModel):
    """One arm on one split over its seeds, in the shape of ``graph_baselines.ArmRow``."""

    model_config = ConfigDict(extra="forbid")
    model: str
    graph: str
    inputs: str
    split_file: str
    dim_in: int
    n_parameters: int
    n_effective_parameters: int
    n_columns_with_train_support: int
    seeds: list[int]
    test_auroc: list[float]
    val_auroc: list[float]
    select_epoch: list[int]
    mean: float
    sd: float
    seconds: float
    train_loss_at: dict[str, list[float]]
    test_logit_sd: list[float]
    test_genes: list[str]
    test_logits: list[list[float]]


class FoldEnsemble(BaseModel):
    """One arm's five fold models on the released held-out genes."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    fold_aurocs: list[float]
    fold_mean: float
    fold_sd: float
    ensemble_auroc: float
    ensemble_ci: list[float]
    ensemble_standardized_auroc: float
    ensemble_standardized_ci: list[float]
    n_genes: int
    n_essential: int
    genes: list[str]
    fold_logits: list[list[float]]


class PairedSplit(BaseModel):
    """One repeated outer split: this arm's seed mean beside the GCN on GO's."""

    model_config = ConfigDict(extra="forbid")
    split_file: str
    arm_seed_aurocs: list[float]
    arm_mean: float
    gcn_seed_aurocs: list[float]
    gcn_mean: float
    difference: float


class PairedSummary(BaseModel):
    """The paired difference (arm minus GCN on GO) over the repeated outer splits."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    n_splits: int
    arm_mean: float
    arm_sd: float
    gcn_mean: float
    gcn_sd: float
    difference_mean: float
    difference_sd: float
    n_arm_above: int
    n_arm_below: int
    splits: list[PairedSplit]


class TuneRow(BaseModel):
    """One configuration of the shared arm, scored on the validation genes only."""

    model_config = ConfigDict(extra="forbid")
    config: Config
    n_parameters: int
    n_effective_parameters: int
    seeds: list[int]
    val_auroc: list[float]
    val_mean: float
    val_sd: float
    select_epoch: list[int]
    seconds: float


class Report(BaseModel):
    """One read's result file."""

    model_config = ConfigDict(extra="forbid")
    read: str
    protocol: str
    config: Config
    hierarchy: HierarchyStats
    rows: list[ArmRow] = []
    fold_ensembles: list[FoldEnsemble] = []
    paired: list[PairedSummary] = []
    tune: list[TuneRow] = []
    seconds: float = 0.0


class Hierarchy:
    """The kept GO terms as a DAG in stratum order, with the gene membership matrix.

    Terms are ordered by stratum (0 = a term with no kept child, the root last), so a
    stratum is a contiguous block. ``membership`` is genes x terms in that order: the
    2,362 multi-hot columns, a namespace root lit by any column of its namespace, and
    the root lit by any column.
    """

    def __init__(self) -> None:
        """Read the multi-hot, its sidecar and the OBO, and wire the kept terms."""
        with open(osp.join(GO_DIR, "go.terms.json")) as fh:
            meta = json.load(fh)
        data, _ = torch.load(osp.join(GO_DIR, "go.pt"), weights_only=False)
        self.gene_ids: list[str] = list(data.id)
        hot = data.embeddings["go"].numpy().astype(np.float32)
        kept = [t["term_id"] for t in meta["terms"]]
        assert hot.shape == (len(self.gene_ids), len(kept)) and len(kept) == 2362
        namespace = {t["term_id"]: t["namespace"] for t in meta["terms"]}
        go_root = osp.join(DATA_ROOT, "data/go")
        dag = GODag(
            osp.join(go_root, "go.obo"), optional_attrs={"relationship"}, prt=None
        )
        ns_root = {
            cast(GOTerm, dag[t]).namespace: t
            for t in ("GO:0008150", "GO:0003674", "GO:0005575")
        }
        nodes = set(kept) | set(ns_root.values())
        memo: dict[str, set[str]] = {}

        def parents(term_id: str) -> set[str]:
            term = cast(GOTerm, dag[term_id])
            out = {p.item_id for p in term.parents}
            part_of = getattr(term, "relationship", {}).get("part_of", set())
            return out | {p.item_id for p in part_of}

        def nearest(term_id: str) -> set[str]:
            out: set[str] = set()
            for p in parents(term_id):
                out |= {p} if p in nodes else nearest(p)
            return out

        edges: list[tuple[str, str]] = []
        n_redundant = 0
        for t in kept:
            near = nearest(t)
            assert near, t
            above = set().union(
                *(GOTermEmbeddingDataset.ancestors(dag, p, memo) for p in near)
            )
            n_redundant += len(near & above)
            edges += [(t, p) for p in sorted(near - above)]
        edges += [(r, ROOT_ID) for r in sorted(ns_root.values())]
        children: dict[str, list[str]] = {}
        for c, p in edges:
            children.setdefault(p, []).append(c)
        stratum: dict[str, int] = {}

        def depth(term_id: str) -> int:
            if term_id not in stratum:
                kids = children.get(term_id, [])
                stratum[term_id] = 1 + max(map(depth, kids)) if kids else 0
            return stratum[term_id]

        depth(ROOT_ID)
        assert len(stratum) == len(nodes) + 1, "a kept term does not reach the root"
        self.terms: list[str] = sorted(stratum, key=lambda t: (stratum[t], t))
        assert self.terms[-1] == ROOT_ID
        index = {t: i for i, t in enumerate(self.terms)}
        self.stratum = np.array([stratum[t] for t in self.terms])
        self.edge_child = np.array([index[c] for c, _ in edges])
        self.edge_parent = np.array([index[p] for _, p in edges])
        self.n_redundant = n_redundant
        membership = np.zeros((len(self.gene_ids), len(self.terms)), dtype=np.float32)
        membership[:, [index[t] for t in kept]] = hot
        for ns, r in ns_root.items():
            cols = [i for i, t in enumerate(kept) if namespace[t] == ns]
            membership[:, index[r]] = hot[:, cols].max(axis=1)
        membership[:, index[ROOT_ID]] = hot.max(axis=1)
        self.membership = membership
        # Genes under each term, over all annotated genes: the kept terms' sidecar
        # counts, and for the four roots the genes lighting any column below them.
        n_genes = {t["term_id"]: int(t["n_genes"]) for t in meta["terms"]}
        for r in (*ns_root.values(), ROOT_ID):
            n_genes[r] = int(membership[:, index[r]].sum())
        self.n_genes = np.array([n_genes[t] for t in self.terms])

    def widths(self, config: Config) -> np.ndarray:
        """Subsystem width per term: ``config.width`` everywhere under the fixed rule;
        DCell's max(20, ceil(0.3 x genes under the term)) capped at ``config.width``
        under ``published_capped``.
        """
        assert config.rule in ("fixed", "published_capped"), config.rule
        if config.rule == "fixed":
            return np.full(len(self.terms), config.width)
        published = np.maximum(20, np.ceil(0.3 * self.n_genes).astype(int))
        return np.minimum(published, config.width)

    def stats(self, n_labeled: int, n_pairs: int) -> HierarchyStats:
        """The sizes that fix the architecture."""
        n_strata = int(self.stratum.max()) + 1
        return HierarchyStats(
            n_terms=len(self.terms),
            n_kept_terms=len(self.terms) - 4,
            n_edges=len(self.edge_child),
            n_redundant_edges_dropped=self.n_redundant,
            n_strata=n_strata,
            terms_per_stratum=np.bincount(self.stratum, minlength=n_strata).tolist(),
            max_children=int(np.bincount(self.edge_parent).max()),
            n_labeled_genes=n_labeled,
            n_annotation_pairs=n_pairs,
        )


class DCellStyle(nn.Module):
    """DCell's hierarchy as grouped tensor products, one block per stratum.

    A term's pre-activation is ``sum_c W[c->t] s_c + g_t + b_t``, which is DCell's
    Linear over the concatenation of its children's states and its gene input. The
    gene input ``g_t`` is ``x_t * w_t`` with one shared vector per term
    (``shared=True``) or the vector of the instance's own (gene, term) pair
    (``shared=False``, DCell's per-gene column). The subsystem is Linear, Tanh,
    BatchNorm (``order="tanh_bn"``, the paper) or Linear, BatchNorm, Tanh
    (``order="bn_tanh"``, ``torchcell/models/dcell.py``). Every term has a linear
    head; the root's is the prediction.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        membership: torch.Tensor,
        widths: np.ndarray,
        order: str,
        shared: bool,
    ) -> None:
        """``membership`` is instances x terms, already masked to train-supported terms;
        ``widths`` is one width per term, padded to the largest with units masked to
        zero, so a term with a narrower subsystem keeps exactly that many units.
        """
        super().__init__()
        assert order in ("tanh_bn", "bn_tanh"), order
        n_terms = len(hierarchy.terms)
        assert widths.shape == (n_terms,)
        self.order = order
        self.shared = shared
        width = int(widths.max())
        self.width = width
        self.register_buffer("x", membership)
        unit_mask = torch.arange(width)[None, :] < torch.from_numpy(widths)[:, None]
        self.register_buffer("unit_mask", unit_mask.float())
        # Edges sorted by (parent stratum, child stratum), so a stratum's incoming
        # edges are one slice of the weight tensor and each source stratum a sub-slice.
        strata = hierarchy.stratum
        edge_order = np.lexsort(
            (strata[hierarchy.edge_child], strata[hierarchy.edge_parent])
        )
        edge_child = hierarchy.edge_child[edge_order]
        edge_parent = hierarchy.edge_parent[edge_order]
        child_units = np.bincount(
            edge_parent, weights=widths[edge_child], minlength=n_terms
        )
        fan_in = torch.from_numpy(child_units + 1).float()
        n_pairs = int(membership.sum()) if not shared else 0
        pair_units = (
            0
            if shared
            else int(widths[torch.nonzero(membership, as_tuple=True)[1].numpy()].sum())
        )
        # The parameters a term of width w_t owns: its edge blocks (w_c x w_t per
        # child), bias, two batch-norm vectors, a head vector and its bias, and the
        # gene input (w_t shared, or w_t per annotation pair). Padded units are not
        # parameters of the network the padding stands in for.
        self.n_effective_parameters = int(
            (widths[edge_child] * widths[edge_parent]).sum()
            + 4 * widths.sum()
            + n_terms
            + (widths.sum() if shared else pair_units)
        )
        del n_pairs
        bound = (1.0 / fan_in.sqrt())[:, None]
        edge_bound = bound[edge_parent][:, :, None]
        n_edges = len(edge_child)
        self.edge_weight = nn.Parameter(
            (torch.rand(n_edges, width, width) * 2 - 1) * edge_bound
        )
        self.bias = nn.Parameter((torch.rand(n_terms, width) * 2 - 1) * bound)
        if shared:
            self.gene_weight = nn.Parameter(
                (torch.rand(n_terms, width) * 2 - 1) * bound
            )
        else:
            rows, cols = torch.nonzero(membership, as_tuple=True)
            self.register_buffer("pair_row", rows)
            self.register_buffer("pair_col", cols)
            self.gene_weight = nn.Parameter(
                (torch.rand(len(rows), width) * 2 - 1) * bound[cols]
            )
        self.bn_weight = nn.Parameter(torch.ones(n_terms, width))
        self.bn_bias = nn.Parameter(torch.zeros(n_terms, width))
        head_bound = 1.0 / float(np.sqrt(width))
        self.head_weight = nn.Parameter(
            (torch.rand(n_terms, width) * 2 - 1) * head_bound
        )
        self.head_bias = nn.Parameter((torch.rand(n_terms) * 2 - 1) * head_bound)
        # Per stratum: its term block, its slice of the edges, the parent's position
        # inside the block, and per source stratum the children's positions there.
        starts = np.searchsorted(strata, np.arange(int(strata.max()) + 2))
        self.blocks: list[
            tuple[int, int, int, int, torch.Tensor, list[tuple[int, torch.Tensor]]]
        ] = []
        for s in range(int(strata.max()) + 1):
            lo, hi = int(starts[s]), int(starts[s + 1])
            e = np.nonzero(strata[edge_parent] == s)[0]
            e_lo, e_hi = (int(e[0]), int(e[-1]) + 1) if len(e) else (0, 0)
            assert len(e) == e_hi - e_lo
            sources = [
                (
                    int(c),
                    torch.from_numpy(
                        edge_child[e][strata[edge_child[e]] == c] - starts[c]
                    ),
                )
                for c in np.unique(strata[edge_child[e]])
            ]
            self.blocks.append(
                (lo, hi, e_lo, e_hi, torch.from_numpy(edge_parent[e] - lo), sources)
            )

    def gene_input(self) -> torch.Tensor:
        """Instances x terms x width: each term's gene input for each deletion."""
        x = cast(torch.Tensor, self.x)
        if self.shared:
            return x[:, :, None] * self.gene_weight[None]
        out = torch.zeros(*x.shape, self.width, dtype=x.dtype)
        return out.index_put(
            (cast(torch.Tensor, self.pair_row), cast(torch.Tensor, self.pair_col)),
            self.gene_weight,
        )

    def normalize(
        self, h: torch.Tensor, stat_rows: torch.Tensor, lo: int, hi: int
    ) -> torch.Tensor:
        """Batch normalization with the training rows' statistics applied to every row."""
        mean = h[stat_rows].mean(dim=0, keepdim=True)
        var = h[stat_rows].var(dim=0, unbiased=False, keepdim=True)
        h = (h - mean) / (var + BN_EPS).sqrt()
        return h * self.bn_weight[lo:hi] + self.bn_bias[lo:hi]

    def forward(self, stat_rows: torch.Tensor) -> torch.Tensor:
        """Instances x terms logits, the last column the root's."""
        gene = self.gene_input()
        states: list[torch.Tensor] = []
        for lo, hi, e_lo, e_hi, parent_pos, sources in self.blocks:
            pre = gene[:, lo:hi] + self.bias[lo:hi]
            if e_hi > e_lo:
                below = torch.cat(
                    [states[c].index_select(1, pos) for c, pos in sources], dim=1
                )
                message = torch.einsum(
                    "beh,ehk->bek", below, self.edge_weight[e_lo:e_hi]
                )
                pre = pre.index_add(1, parent_pos, message)
            if self.order == "tanh_bn":
                state = self.normalize(torch.tanh(pre), stat_rows, lo, hi)
            else:
                state = torch.tanh(self.normalize(pre, stat_rows, lo, hi))
            states.append(state * cast(torch.Tensor, self.unit_mask)[lo:hi])
        state = torch.cat(states, dim=1)
        return (state * self.head_weight[None]).sum(dim=-1) + self.head_bias[None]


def smoothed_argmax(values: list[float]) -> int:
    """The selection rule of ``graph_baselines.py``: a five-epoch moving average."""
    s = np.convolve(np.array(values), np.ones(5) / 5, mode="same")
    return int(np.argmax(s))


def bootstrap(label: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    """The gene bootstrap of ``cv_ensemble_readout.py``: 2,000 draws, seed 0."""
    point = float(roc_auc_score(label, score))
    rng = np.random.default_rng(0)
    draws: list[float] = []
    n = label.shape[0]
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(label[idx])) < 2:
            continue
        draws.append(float(roc_auc_score(label[idx], score[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_split(name: str) -> dict:
    """One split file under ``results/splits``."""
    with gzip.open(osp.join(SPLITS, name), "rt") as fh:
        return cast(dict, json.load(fh))


class SeedRun(BaseModel):
    """One arm on one split at one seed: the unit of work a pool worker runs."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    split_file: str
    seed: int
    config: Config
    dim_in: int
    n_parameters: int
    n_effective_parameters: int
    n_columns_with_train_support: int
    select_epoch: int
    val_auroc: float
    test_auroc: float | None
    test_logit_sd: float | None
    train_loss_at: dict[str, float]
    test_genes: list[str]
    test_logits: list[float]
    seconds: float


class Task(BaseModel):
    """One (arm, split, seed, configuration) cell to train."""

    model_config = ConfigDict(extra="forbid")
    arm: str
    split_file: str
    seed: int
    config: Config
    score_test: bool = True


_HIERARCHY: list[Hierarchy] = []


def worker_hierarchy() -> Hierarchy:
    """The process's hierarchy, built once per pool worker."""
    if not _HIERARCHY:
        _HIERARCHY.append(Hierarchy())
    return _HIERARCHY[0]


def set_threads(n: int) -> None:
    """Pool initializer: the intra-op thread count of one worker."""
    torch.set_num_threads(n)


def run_seed(task: Task) -> SeedRun:
    """Train one cell; select the epoch on validation, score the held-out genes once."""
    assert task.arm in ARMS, task.arm
    hierarchy = worker_hierarchy()
    config = task.config
    split = load_split(task.split_file)
    labels = {k: int(v) for k, v in split["labels"].items()}
    gene_index = {g: i for i, g in enumerate(hierarchy.gene_ids)}
    genes = {
        k: [g for g in split["genes"][k] if g in gene_index]
        for k in ("train", "val", "test")
    }
    order = genes["train"] + genes["val"] + genes["test"]
    bounds = np.cumsum([0] + [len(genes[k]) for k in ("train", "val", "test")])
    rows = {
        k: torch.arange(int(bounds[i]), int(bounds[i + 1]))
        for i, k in enumerate(("train", "val", "test"))
    }
    ys = {
        k: torch.tensor([labels[g] for g in genes[k]], dtype=torch.float32)
        for k in genes
    }
    x = hierarchy.membership[[gene_index[g] for g in order]].copy()
    support = x[: len(genes["train"])].sum(axis=0) > 0
    x[:, ~support] = 0.0
    y_train = ys["train"]
    pos = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - pos) / pos)
    alpha = 0.0 if task.arm.endswith("noaux") else ALPHA
    marks = sorted({0, 9, 49, 99, 199, config.epochs - 1})
    loss_at: dict[str, float] = {}
    t0 = time.time()
    torch.manual_seed(task.seed)
    net = DCellStyle(
        hierarchy,
        torch.from_numpy(x),
        hierarchy.widths(config),
        config.order,
        shared=task.arm != "dcell_published_input",
    )
    opt = torch.optim.Adam(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    val_curve: list[float] = []
    test_curve: list[float] = []
    test_logits: list[np.ndarray] = []
    # One forward per epoch: batch normalization reads the training rows in every
    # pass and nothing is stochastic, so the forward that opens epoch e + 1 IS the
    # scoring pass after epoch e's step (as graph_baselines.py scores).
    for epoch in range(config.epochs + 1):
        out = net(rows["train"])
        if epoch > 0:
            logit = out[:, -1].detach()
            val_curve.append(
                float(roc_auc_score(ys["val"].numpy(), logit[rows["val"]].numpy()))
            )
            if task.score_test:
                held = logit[rows["test"]].numpy().copy()
                test_curve.append(float(roc_auc_score(ys["test"].numpy(), held)))
                test_logits.append(held)
        if epoch == config.epochs:
            break
        opt.zero_grad()
        out = out[rows["train"]]
        loss = F.binary_cross_entropy_with_logits(
            out[:, -1], y_train, pos_weight=pos_weight
        )
        if alpha > 0:
            loss = loss + alpha * F.binary_cross_entropy_with_logits(
                out[:, :-1],
                y_train[:, None].expand(-1, out.shape[1] - 1),
                pos_weight=pos_weight,
            )
        loss.backward()
        opt.step()
        if epoch in marks:
            loss_at[f"epoch_{epoch}"] = float(loss.detach())
    e = smoothed_argmax(val_curve)
    run = SeedRun(
        arm=task.arm,
        split_file=task.split_file,
        seed=task.seed,
        config=config,
        dim_in=int(x.shape[1]),
        n_parameters=sum(p.numel() for p in net.parameters()),
        n_effective_parameters=net.n_effective_parameters,
        n_columns_with_train_support=int(support.sum()),
        select_epoch=e,
        val_auroc=val_curve[e],
        test_auroc=test_curve[e] if task.score_test else None,
        test_logit_sd=float(test_logits[e].std()) if task.score_test else None,
        train_loss_at=loss_at,
        test_genes=genes["test"] if task.score_test else [],
        test_logits=test_logits[e].tolist() if task.score_test else [],
        seconds=time.time() - t0,
    )
    print(
        f"{run.arm:26s} {run.split_file:36s} s{run.seed} {config.rule} w{config.width} "
        f"{config.order} wd{config.weight_decay:g} val {run.val_auroc:.4f} test "
        + (f"{run.test_auroc:.4f}" if run.test_auroc is not None else "unread")
        + f" sel {e} loss {loss_at} ({run.seconds:.0f}s)",
        flush=True,
    )
    return run


def run_tasks(tasks: list[Task], workers: int, threads: int) -> list[SeedRun]:
    """Every task, in order, over a pool of ``workers`` processes of ``threads`` each."""
    assert workers * threads <= 16, "the node's cards are busy; at most 16 threads"
    if workers == 1:
        set_threads(threads)
        return [run_seed(t) for t in tasks]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=set_threads,
        initargs=(threads,),
    ) as pool:
        return list(pool.map(run_seed, tasks))


def arm_row(runs: list[SeedRun]) -> ArmRow:
    """The seeds of one arm on one split, in the shape of ``graph_baselines.ArmRow``."""
    assert len({(r.arm, r.split_file) for r in runs}) == 1
    tests = [r.test_auroc for r in runs if r.test_auroc is not None]
    assert len(tests) == len(runs), "an ArmRow holds scored seeds only"
    return ArmRow(
        model=runs[0].arm,
        graph="GO DAG",
        inputs="GO",
        split_file=runs[0].split_file,
        dim_in=runs[0].dim_in,
        n_parameters=runs[0].n_parameters,
        n_effective_parameters=runs[0].n_effective_parameters,
        n_columns_with_train_support=runs[0].n_columns_with_train_support,
        seeds=[r.seed for r in runs],
        test_auroc=tests,
        val_auroc=[r.val_auroc for r in runs],
        select_epoch=[r.select_epoch for r in runs],
        mean=float(np.mean(tests)),
        sd=float(np.std(tests)),
        seconds=float(sum(r.seconds for r in runs)),
        train_loss_at={
            k: [r.train_loss_at[k] for r in runs] for k in runs[0].train_loss_at
        },
        test_logit_sd=[cast(float, r.test_logit_sd) for r in runs],
        test_genes=runs[0].test_genes,
        test_logits=[r.test_logits for r in runs],
    )


def arm_rows(runs: list[SeedRun]) -> list[ArmRow]:
    """Group seed runs into one row per (arm, split), keeping the task order."""
    groups: dict[tuple[str, str], list[SeedRun]] = {}
    for r in runs:
        groups.setdefault((r.arm, r.split_file), []).append(r)
    rows = [arm_row(g) for g in groups.values()]
    for row in rows:
        print(
            f"{row.model:26s} {row.split_file:36s} test "
            + ", ".join(f"{t:.3f}" for t in row.test_auroc)
            + f"  mean {row.mean:.4f} sd {row.sd:.4f} sel {row.select_epoch}",
            flush=True,
        )
    return rows


def fold_ensemble(arm: str, rows: list[ArmRow], labels: dict[str, int]) -> FoldEnsemble:
    """The mean-logit ensemble of the fold models, and the standardized-logit one."""
    per_fold = []
    for row in rows:
        fold_logit = np.mean(np.array(row.test_logits, dtype=float), axis=0)
        per_fold.append(dict(zip(row.test_genes, fold_logit, strict=True)))
    genes = sorted(per_fold[0])
    assert all(sorted(f) == genes for f in per_fold), "folds score different genes"
    y = np.array([labels[g] for g in genes])
    table = np.array([[f[g] for g in genes] for f in per_fold])
    fold_aurocs = [float(roc_auc_score(y, r)) for r in table]
    point, lo, hi = bootstrap(y, table.mean(axis=0))
    z = (table - table.mean(axis=1, keepdims=True)) / table.std(axis=1, keepdims=True)
    z_point, z_lo, z_hi = bootstrap(y, z.mean(axis=0))
    return FoldEnsemble(
        arm=arm,
        fold_aurocs=fold_aurocs,
        fold_mean=float(np.mean(fold_aurocs)),
        fold_sd=float(np.std(fold_aurocs)),
        ensemble_auroc=point,
        ensemble_ci=[lo, hi],
        ensemble_standardized_auroc=z_point,
        ensemble_standardized_ci=[z_lo, z_hi],
        n_genes=len(genes),
        n_essential=int(y.sum()),
        genes=genes,
        fold_logits=table.tolist(),
    )


def paired_summary(arm: str, rows: list[ArmRow]) -> PairedSummary:
    """This arm against the GCN on GO, split by split, on the repeated outer splits."""
    splits: list[PairedSplit] = []
    for row in rows:
        tag = row.split_file.split("_val")[0].removeprefix("ess_")
        with open(osp.join(RESULTS, f"graph_baselines_go400_{tag}.json")) as fh:
            gcn = [r for r in json.load(fh)["rows"] if r["inputs"] == GCN_ROW]
        assert len(gcn) == 1, tag
        splits.append(
            PairedSplit(
                split_file=row.split_file,
                arm_seed_aurocs=row.test_auroc,
                arm_mean=row.mean,
                gcn_seed_aurocs=gcn[0]["test_auroc"],
                gcn_mean=float(np.mean(gcn[0]["test_auroc"])),
                difference=row.mean - float(np.mean(gcn[0]["test_auroc"])),
            )
        )
    diff = np.array([s.difference for s in splits])
    return PairedSummary(
        arm=arm,
        n_splits=len(splits),
        arm_mean=float(np.mean([s.arm_mean for s in splits])),
        arm_sd=float(np.std([s.arm_mean for s in splits], ddof=1)),
        gcn_mean=float(np.mean([s.gcn_mean for s in splits])),
        gcn_sd=float(np.std([s.gcn_mean for s in splits], ddof=1)),
        difference_mean=float(diff.mean()),
        difference_sd=float(diff.std(ddof=1)),
        n_arm_above=int((diff > 0).sum()),
        n_arm_below=int((diff < 0).sum()),
        splits=splits,
    )


def main() -> None:
    """Run one read and write ``results/dcell_style_<read>.json``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--read", required=True, choices=("released", "folds", "resplits", "tune")
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument(
        "--rule", default="fixed", choices=("fixed", "published_capped")
    )
    parser.add_argument("--tune-widths", type=int, nargs="+", default=list(TUNE_WIDTHS))
    parser.add_argument("--tune-rules", nargs="+", default=["fixed"])
    parser.add_argument("--tune-orders", nargs="+", default=list(TUNE_ORDERS))
    parser.add_argument(
        "--tune-weight-decays", type=float, nargs="+", default=list(TUNE_WEIGHT_DECAYS)
    )
    parser.add_argument("--order", default="tanh_bn", choices=("tanh_bn", "bn_tanh"))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2, help="Threads per worker.")
    parser.add_argument("--tag", default="", help="Suffix for the result file name.")
    args = parser.parse_args()
    config = Config(
        width=args.width,
        rule=args.rule,
        order=args.order,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )
    t0 = time.time()
    hierarchy = worker_hierarchy()
    released = load_split(RELEASED)
    labeled = [g for g in released["labels"] if g in set(hierarchy.gene_ids)]
    gene_index = {g: i for i, g in enumerate(hierarchy.gene_ids)}
    n_pairs = int(hierarchy.membership[[gene_index[g] for g in labeled]].sum())
    stats = hierarchy.stats(len(labeled), n_pairs)
    print(stats.model_dump_json(), flush=True)
    report = Report(
        read=args.read,
        protocol=(
            "one deletion per instance, split by gene; class-weighted BCE on the root "
            f"head plus {ALPHA} x the mean class-weighted BCE over the non-root term "
            f"heads; Adam {LR}, {config.epochs} epochs, full batch, batch normalization "
            "statistics from the training genes; epoch selected by a five-epoch moving "
            "average of the validation AUROC; held-out genes scored once at that epoch"
        ),
        config=config,
        hierarchy=stats,
    )

    def cells(split_files: list[str], seeds: tuple[int, ...]) -> list[ArmRow]:
        tasks = [
            Task(arm=a, split_file=f, seed=s, config=config)
            for a in args.arms
            for f in split_files
            for s in seeds
        ]
        return arm_rows(run_tasks(tasks, args.workers, args.threads))

    if args.read == "released":
        report.rows = cells([RELEASED], tuple(range(2700, 2706)))
    elif args.read == "folds":
        labels = {g: int(v) for g, v in released["labels"].items()}
        report.rows = cells(
            [f"ess_fcl_fold{k}of5_s0.json.gz" for k in range(5)], (2700,)
        )
        for arm in args.arms:
            ens = fold_ensemble(arm, [r for r in report.rows if r.model == arm], labels)
            report.fold_ensembles.append(ens)
            print(
                f"{arm}: folds {[round(a, 3) for a in ens.fold_aurocs]} mean "
                f"{ens.fold_mean:.4f}; mean-logit ensemble {ens.ensemble_auroc:.4f} "
                f"{[round(c, 3) for c in ens.ensemble_ci]}; standardized "
                f"{ens.ensemble_standardized_auroc:.4f}",
                flush=True,
            )
    elif args.read == "resplits":
        report.rows = cells(
            [f"ess_fcl_resplit{r}_val0.2_s0.json.gz" for r in range(10)],
            (2700, 2701, 2702),
        )
        for arm in args.arms:
            summary = paired_summary(arm, [r for r in report.rows if r.model == arm])
            report.paired.append(summary)
            print(
                f"{arm}: mean {summary.arm_mean:.4f} sd {summary.arm_sd:.4f}; GCN "
                f"{summary.gcn_mean:.4f}; difference {summary.difference_mean:+.4f} sd "
                f"{summary.difference_sd:.4f}; above {summary.n_arm_above} below "
                f"{summary.n_arm_below} of {summary.n_splits}",
                flush=True,
            )
    else:
        seeds = (2700, 2701, 2702)
        grid = [
            Config(width=w, rule=r, order=o, weight_decay=d, epochs=args.epochs)
            for r in args.tune_rules
            for w in args.tune_widths
            for o in args.tune_orders
            for d in args.tune_weight_decays
        ]
        tasks = [
            Task(
                arm="dcell_style_shared",
                split_file=RELEASED,
                seed=s,
                config=c,
                score_test=False,
            )
            for c in grid
            for s in seeds
        ]
        runs = run_tasks(tasks, args.workers, args.threads)
        for c in grid:
            mine = [r for r in runs if r.config == c]
            report.tune.append(
                TuneRow(
                    config=c,
                    n_parameters=mine[0].n_parameters,
                    n_effective_parameters=mine[0].n_effective_parameters,
                    seeds=[r.seed for r in mine],
                    val_auroc=[r.val_auroc for r in mine],
                    val_mean=float(np.mean([r.val_auroc for r in mine])),
                    val_sd=float(np.std([r.val_auroc for r in mine])),
                    select_epoch=[r.select_epoch for r in mine],
                    seconds=float(sum(r.seconds for r in mine)),
                )
            )
            print(report.tune[-1].model_dump_json(), flush=True)
    report.seconds = time.time() - t0
    out = osp.join(RESULTS, f"dcell_style_{args.read}{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
