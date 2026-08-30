# experiments/019-simb-multimodal/scripts/graph_prior_probe.py
# [[experiments.019-simb-multimodal.scripts.graph_prior_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/graph_prior_probe
"""Do the nine masked graphs carry the relationship the mask assumes, on EVERY phenotype?

THE ASSUMPTION UNDER TEST. The nine interaction graphs do not inject features into the
model. They enter as a hard additive attention mask, so all they do is constrain where each
token may look. Stacking L layers composes the attention matrices, which makes the
influence of gene X on gene Y's final token approximately the L-step random-walk
reachability of Y from X on that graph. The entire content of the graph channel is
therefore one falsifiable claim about the data, and it has to be checked on each phenotype
separately: a protein-protein graph could carry morphology and say nothing about
expression, and averaging the two would hide both.

TWO STATISTICS, and the reason there are two.

  REPORTER FORM (expression only).  "Are the reporter genes that respond to deleting X
  near X on graph k?"  Response magnitude is |y_{X,g}| from the measured log2 ratios; the
  predictor is the t-step row-normalized random-walk weight w^{(k,t)}_{X,g} = (A_k^t)_{gX}.
  AUC is the probability that a randomly chosen RESPONDING reporter is closer to X than a
  randomly chosen non-responder, averaged over deleted genes. Responders are the top
  RESPONDER_QUANTILE of reporters by |y| WITHIN each strain; a second quantile is reported
  so the choice is visible rather than load-bearing. This form needs the readout to be
  INDEXED BY GENE, so it exists only for expression (and proteome). It is undefined for
  morphology, the amino acids, the two pigments and fitness.

  PAIR FORM (every phenotype).  "Do deletions of two genes that are ADJACENT on graph k
  produce more similar phenotypes than deletions of two non-adjacent genes?"  Adjacent
  pairs are sampled as t-step random walks from a deleted gene, which is the same
  reachability the composed attention implies, rather than as uniform draws from the edge
  list. Background pairs are uniform draws from the same gene pool. AUC is the probability
  that an adjacent pair is more similar than a background pair. Similarity is the
  across-feature Pearson correlation for vector readouts (expression, morphology, the 19
  amino acids) and the negative absolute difference for scalar readouts (betaxanthin,
  beta-carotene, single-mutant fitness), each after per-feature z-scoring.

  Expression is run in BOTH forms. That is the calibration that licenses reading a
  pair-form number on a strand where the reporter form does not exist.

DIRECTION IS A REPORTER-FORM QUESTION, and this is a property of the statistic, not an
omission. Symmetrizing a mask adds (j,i) wherever (i,j) existed, so it does not change
which unordered PAIRS are adjacent, and phenotype similarity is symmetric in the pair. At
t = 1 the forward and transposed pair samples therefore cover the same unordered pairs and
differ only in how out-degree versus in-degree weights the draw. Both are computed and
reported anyway, at every t, so that invariance is visible in the numbers rather than
asserted. The orientation finding itself lives in the reporter form.

TWO CONTROLS, without which no number means anything:

  rewired    a configuration-model redraw from the same degree sequence. Destroys
             topology, keeps each node's degree, so anything above this is real structure
             rather than a hub artifact.
             This is the control that matters, and it matters MORE in the pair form: a
             random walk lands on high-degree nodes preferentially while the background
             draw does not, so adjacent pairs are degree-enriched by construction. The
             rewired walk carries exactly the same degree enrichment. Read real vs
             rewired, never real vs 0.5.
  random     a matched-density Erdos-Renyi graph. The floor.

COST. The graphs are sparse and only the deleted-gene COLUMNS of A^t are needed, so the
reporter-form walk propagates a dense block through a sparse matrix rather than densifying
A^t. The pair form samples walks through the CSR index and never forms A^t at all. The
three graph variants are built once per graph and shared by both forms and all phenotypes.

Run from repo root:
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/graph_prior_probe.py
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/graph_prior_probe.py --figure-only
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import rankdata

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS
from torchcell.graph.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

EXPERIMENT = "019-simb-multimodal"
DATASET_TAG = "019-simb-multimodal/fig3_core"

# The nine graphs the model masks attention with, one per head, exactly as the
# `head_graphs` block of the 019/023 configs lists them, so the probe tests the deployed
# prior rather than a plausible-looking selection.
#
# TWO NAMING CONVENTIONS, and they differ for the first two entries. The configs use the
# HeteroData RELATION names that `to_cell_data` emits, which are suffixed
# (`physical_interaction`); `build_gene_multigraph` takes the BUILDER names, which are not
# (`physical`). Mapping them here rather than renaming either side keeps the probe's graph
# list literally copy-pasteable from the config it is testing.
CONFIG_TO_BUILDER = {
    "physical_interaction": "physical",
    "regulatory_interaction": "regulatory",
}
GRAPH_NAMES = [
    "physical_interaction",
    "regulatory_interaction",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
BUILDER_NAMES = [CONFIG_TO_BUILDER.get(n, n) for n in GRAPH_NAMES]

WALK_LENGTHS = (1, 2, 3)
RESPONDER_QUANTILE = 0.01
RESPONDER_QUANTILE_ALT = 0.05
RANDOM_SEED = 0

# Pair-form sampling. WALK_DRAWS walks are launched per (graph, variant, walk length,
# PHENOTYPE), starting only from genes that carry that phenotype and keeping the walks
# whose endpoint carries it too. Starting from the union of all strands instead would waste
# most of the draw on expression, which measures 1,482 deleted genes against a union pool
# near 5,000. The background is drawn once per graph per phenotype and shared across
# variants and walk lengths, so the real-vs-rewired contrast that the figure reads carries
# no background sampling noise.
WALK_DRAWS = 400_000
BACKGROUND_DRAWS = 400_000
PAIR_CAP = 60_000  # ceiling on pairs actually scored, per side; AUC SE at this n is ~0.002

# Morphology: the 278 features the 019 delta config trains on (281 CalMorph labels minus
# the three the config drops).
MORPH_DROPPED = ["A113_A", "D203", "D205"]

PHENOTYPE_ORDER = [
    "expression",
    "morphology",
    "amino_acid",
    "betaxanthin",
    "beta_carotene",
    "fitness",
]
PHENOTYPE_TITLES = {
    "expression": "Expression (Kemmeren, 6,169 reporters)",
    "morphology": "Morphology (Ohya, 278 CalMorph features)",
    "amino_acid": "Amino acids (Mulleder, 19 metabolites)",
    "betaxanthin": "Betaxanthin (Cachera, scalar)",
    "beta_carotene": "Beta-carotene (Ozaydin, ordinal)",
    "fitness": "Single-mutant fitness (Costanzo, scalar)",
}


# --------------------------------------------------------------------------------------
# Phenotype loading. Each loader returns (genes, matrix) with matrix (n_genes x n_features)
# already z-scored per feature, so the similarity is comparable across readouts. Every path
# follows the loader the existing 019/023 scripts use for that dataset.
# --------------------------------------------------------------------------------------


def _zscore(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, ddof=1, keepdims=True)
    # A feature with no variance across strains carries no similarity information; it is
    # zeroed rather than divided by zero, which would put an inf into every pair.
    result: np.ndarray = np.divide(
        centered, sd, out=np.zeros_like(centered), where=sd > 0
    )
    return result


def load_expression(data_root: str) -> tuple[list[str], list[str], np.ndarray]:
    """(deleted genes, reporter genes, SIGNED log2 ratio) over single-deletion strains.

    The reporter form takes |.| of this; the pair form needs the sign, because two
    deletions that move the same reporters in OPPOSITE directions are not similar.
    """
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    rows: dict[str, np.ndarray] = {}
    reporters: list[str] | None = None
    with env.begin() as txn:
        cursor = txn.cursor()
        for _key, value in cursor:
            recs = json.loads(value.decode())
            if isinstance(recs, dict):
                recs = [recs]
            perts = recs[0]["experiment"]["genotype"]["perturbations"]
            if len(perts) != 1:
                continue  # single deletions only; the probe is indexed by deleted gene
            gene = perts[0]["systematic_gene_name"]
            for rec in recs:
                phen = rec["experiment"]["phenotype"]
                if phen.get("label_name") != "expression_log2_ratio":
                    continue
                values = phen["expression_log2_ratio"]
                if reporters is None:
                    reporters = sorted(values)
                rows[gene] = np.array(
                    [values[k] for k in reporters], dtype=np.float32
                )
    env.close()
    if reporters is None:
        raise RuntimeError("no expression records found in the fig3_core build")
    deleted = sorted(rows)
    matrix = np.vstack([rows[g] for g in deleted])
    return deleted, reporters, matrix


def load_morphology(data_root: str) -> tuple[list[str], np.ndarray]:
    """Ohya 2005 CalMorph, the 278 features the 019 config models.

    Same sha256-pinned SCMD mirror `morphology_noise_ceiling.py` reads.
    """
    path = osp.join(
        data_root,
        "torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data",
        "mt4718data.tsv",
    )
    frame = pd.read_csv(path, sep="\t")
    # THE MIRROR STORES ORFs IN LOWERCASE (`yal002w`), and every other table here plus
    # every graph uses uppercase. Without this the morphology gene set intersects the
    # graphs in zero genes and the panel comes back at chance for a reason that has
    # nothing to do with biology. `ohya2005.py` applies the same `.strip().upper()`.
    frame["ORF"] = frame["ORF"].str.strip().str.upper()
    frame = frame.set_index("ORF")
    feats = [f for f in CALMORPH_LABELS if f not in MORPH_DROPPED]
    missing = [f for f in feats if f not in frame.columns]
    if missing:
        raise RuntimeError(f"CalMorph features absent from the mirror: {missing[:5]}")
    sub = frame[feats].dropna()
    return list(sub.index), sub.to_numpy(dtype=np.float64)


def load_amino_acid(data_root: str) -> tuple[list[str], np.ndarray]:
    """Mulleder 2016 intracellular concentrations of 19 amino acids, log10 then z.

    Concentrations are positive and heavy-tailed, so the log is taken before z-scoring,
    matching what `betaxanthin_amino_acid_predictivity.py` does with the same table.
    """
    path = osp.join(
        data_root, "data/torchcell/amino_acid_mulleder2016/preprocess/data.csv"
    )
    frame = pd.read_csv(path)
    names = [c for c in frame.columns if c != "orf"]
    values = frame[names].to_numpy(dtype=np.float64)
    if (values <= 0).any():
        raise RuntimeError("non-positive amino-acid concentration; log10 is undefined")
    return list(frame["orf"]), np.log10(values)


def load_betaxanthin(data_root: str) -> tuple[list[str], np.ndarray]:
    """Cachera 2023 CRI-SPA corrected betaxanthin fluorescence, one value per ORF."""
    path = osp.join(
        data_root, "data/torchcell/betaxanthin_cachera2023/preprocess/data.csv"
    )
    frame = pd.read_csv(path)
    return list(frame["orf"]), frame[["level"]].to_numpy(dtype=np.float64)


def load_beta_carotene(data_root: str) -> tuple[list[str], np.ndarray]:
    """Ozaydin 2013 ordinal colony-color score, -5..+5."""
    path = osp.join(
        data_root, "data/torchcell/carotenoid_ozaydin2013/preprocess/data.csv"
    )
    frame = pd.read_csv(path)
    return list(frame["orf"]), frame[["visual_score"]].to_numpy(dtype=np.float64)


def load_fitness(data_root: str) -> tuple[list[str], np.ndarray]:
    """Costanzo 2016 single-mutant fitness, KanMX deletions at 30 C.

    The NatMX arm is the query-strain copy of the same deletions and the ts/damp alleles
    are not deletions at all, so pooling them would average different perturbation classes
    into one target. Same filter `betaxanthin_amino_acid_predictivity.py` applies.
    """
    path = osp.join(data_root, "data/torchcell/smf_costanzo2016/preprocess/data.csv")
    frame = pd.read_csv(path)
    frame = frame[
        (frame["perturbation_type"] == "KanMX_deletion") & (frame["Temperature"] == 30)
    ]
    grouped = (
        frame[["Systematic gene name", "Single mutant fitness"]]
        .groupby("Systematic gene name", as_index=False)
        .mean()
    )
    return (
        list(grouped["Systematic gene name"]),
        grouped[["Single mutant fitness"]].to_numpy(dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# Graph construction and controls (shared by both statistics).
# --------------------------------------------------------------------------------------


def row_normalized(
    graph: nx.Graph, index: dict[str, int], symmetrize: bool = True
) -> sp.csr_matrix:
    """Row-stochastic adjacency over the fixed node index, as the model's mask implies.

    Built through `to_scipy_sparse_array` rather than a Python loop over `graph.edges()`.
    Two of these graphs carry about a million edges and each is rebuilt three times (real,
    rewired, random), so the loop form was several million Python iterations per graph and
    was the slowest thing here by a wide margin.
    """
    n = len(index)
    node_list = [g for g in graph.nodes() if g in index]
    if not node_list:
        return sp.csr_matrix((n, n), dtype=np.float32)
    sub = graph.subgraph(node_list)
    local = nx.to_scipy_sparse_array(
        sub, nodelist=node_list, format="coo", weight=None, dtype=np.float32
    )
    # Vectorized local-to-global reindex; `to_scipy_sparse_array` already emits both
    # directions of an undirected edge, so no symmetrization is needed here.
    to_global = np.array([index[g] for g in node_list], dtype=np.int64)
    adjacency = sp.coo_matrix(
        (local.data, (to_global[local.row], to_global[local.col])),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    if symmetrize:
        # Exactly what `_build_attention_mask` does: set both (i,j) and (j,i).
        adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0  # collapse any parallel edges
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # An isolated node keeps an all-zero row rather than becoming a uniform distribution:
    # a gene with no edges on this graph should contribute no proximity signal, and
    # spreading mass uniformly would invent one.
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sp.diags(inverse).astype(np.float32) @ adjacency


def walk_columns(
    adjacency: sp.csr_matrix, columns: np.ndarray, walks: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """(A^t)[:, columns] for each t, by repeated sparse-times-dense propagation.

    Only the deleted-gene columns are ever needed, so this never densifies A^t.
    """
    n = adjacency.shape[0]
    block = np.zeros((n, len(columns)), dtype=np.float32)
    block[columns, np.arange(len(columns))] = 1.0
    out: dict[int, np.ndarray] = {}
    current = block
    for t in range(1, max(walks) + 1):
        current = adjacency @ current
        if t in walks:
            out[t] = current.copy()
    return out


def rewired(graph: nx.Graph, rng: np.random.Generator) -> tuple[nx.Graph, float]:
    """Degree-preserving null by the configuration model, and how well degree survived.

    WHY NOT `double_edge_swap`, which preserves degree exactly. Mixing needs roughly ten
    swaps per edge, and two of these graphs carry about a million edges (coexpression
    996,199; experimental 822,094), so that control alone is tens of millions of swaps in
    pure Python and runs for hours. The configuration model draws a fresh graph from the
    same degree sequence in O(m).

    The trade is that it produces a multigraph with self-loops, and collapsing those to a
    simple graph lowers the degree of high-degree nodes slightly. That matters here because
    degree is exactly what this control exists to hold fixed, so the Spearman correlation
    between the original and rewired degree sequences is returned and recorded rather than
    assumed to be 1.
    """
    nodes = list(graph.nodes())
    degrees = [d for _n, d in graph.degree()]
    multi = nx.configuration_model(
        degrees, seed=int(rng.integers(0, 2**31 - 1)), create_using=nx.MultiGraph
    )
    simple = nx.Graph(multi)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    simple = nx.relabel_nodes(simple, {i: nodes[i] for i in range(len(nodes))})
    new_degrees = np.array([simple.degree(n) for n in nodes], dtype=float)
    old_degrees = np.array(degrees, dtype=float)
    if old_degrees.std() == 0 or new_degrees.std() == 0:
        preserved = float("nan")
    else:
        preserved = float(
            np.corrcoef(rankdata(old_degrees), rankdata(new_degrees))[0, 1]
        )
    return simple, preserved


def random_matched(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Erdos-Renyi at matched node and edge count."""
    return nx.gnm_random_graph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


# --------------------------------------------------------------------------------------
# Reporter form (expression only). Unchanged from the published run.
# --------------------------------------------------------------------------------------


def score(
    weights: np.ndarray, response: np.ndarray, quantile: float
) -> tuple[float, float, float]:
    """Mean AUC, mean Spearman, and the fraction of deleted genes that were scorable.

    `weights` is (reporters x deleted); `response` is (deleted x reporters), magnitudes.

    Fully vectorized over deleted genes. The per-gene form called `rankdata` twice on a
    6,169-element vector for every gene, graph, walk length and control variant, which is
    about 27,000 sorts per graph; ranking each matrix once along the reporter axis does the
    same work in a handful of calls.
    """
    n_deleted = response.shape[0]
    resp_t = response.T  # (reporters x deleted), to match `weights`

    # A deleted gene with no edges on this graph gets no prediction from it. Excluded
    # rather than scored at chance, and the excluded fraction is reported.
    scorable = np.asarray((weights > 0).any(axis=0)).ravel()
    if not scorable.any():
        return float("nan"), float("nan"), 0.0
    w = weights[:, scorable]
    y = resp_t[:, scorable]

    cuts = np.quantile(y, 1.0 - quantile, axis=0)
    responder = y > cuts
    n_pos = responder.sum(axis=0).astype(float)
    n_neg = float(y.shape[0]) - n_pos

    w_ranks = rankdata(w, axis=0)
    rank_sum = (w_ranks * responder).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        auc[(n_pos == 0) | (n_neg == 0)] = np.nan

    # Spearman is Pearson on the ranks, computed column-wise.
    y_ranks = rankdata(y, axis=0)
    wc = w_ranks - w_ranks.mean(axis=0, keepdims=True)
    yc = y_ranks - y_ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt((wc**2).sum(axis=0) * (yc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denom > 0, (wc * yc).sum(axis=0) / denom, np.nan)

    return (
        float(np.nanmean(auc)),
        float(np.nanmean(rho)),
        float(scorable.sum() / n_deleted),
    )


# --------------------------------------------------------------------------------------
# Pair form (every phenotype).
# --------------------------------------------------------------------------------------


def sample_walk_pairs(
    adjacency: sp.csr_matrix,
    start_pool: np.ndarray,
    steps: int,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) global node ids of `draws` uniform `steps`-step random walks.

    Neighbors are drawn uniformly through the CSR index, so the endpoint distribution is
    the row-stochastic walk the composed attention mask implies. A walk that reaches a
    degree-zero node is dropped rather than restarted; the drop is invisible to the
    comparison because the rewired and random variants drop under the same rule.
    """
    indptr = adjacency.indptr
    indices = adjacency.indices
    degree = np.diff(indptr)
    live = start_pool[degree[start_pool] > 0]
    if live.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    start = rng.choice(live, size=draws, replace=True)
    current = start.copy()
    for _ in range(steps):
        deg = degree[current]
        keep = deg > 0
        start = start[keep]
        current = current[keep]
        deg = deg[keep]
        if current.size == 0:
            break
        offset = (rng.random(current.size) * deg).astype(np.int64)
        current = indices[indptr[current] + offset].astype(np.int64)
    return start, current


def pair_similarity(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Similarity of the phenotype rows at `left` and `right`, elementwise over pairs.

    Vector readouts get the across-feature Pearson correlation, which is the dot product of
    the row-centered, row-normalized vectors. Scalar readouts (one column) get the negative
    absolute difference; a correlation between two one-element vectors is undefined, and
    the difference is the statistic that actually orders scalar pairs by similarity.
    Chunked so the (pairs x features) intermediate never has to fit in memory: expression
    at 60,000 pairs and 6,169 reporters would be 3 GB in one block.
    """
    if matrix.shape[1] == 1:
        scalar_diff: np.ndarray = -np.abs(matrix[left, 0] - matrix[right, 0])
        return scalar_diff
    out = np.empty(left.size, dtype=np.float64)
    chunk = 2000
    for lo in range(0, left.size, chunk):
        hi = min(lo + chunk, left.size)
        a = matrix[left[lo:hi]]
        b = matrix[right[lo:hi]]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = na * nb
        num = (a * b).sum(axis=1)
        out[lo:hi] = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    return out


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    """P(a positive similarity exceeds a negative one), with ties at half credit, and SE.

    The SE is the Hanley-McNeil approximation, which is what says whether a 0.51 is a
    finding or a sampling wobble.
    """
    n_pos, n_neg = positive.size, negative.size
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    auc = float(
        (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (float(n_pos) * n_neg)
    )
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (float(n_pos) * n_neg)
    return auc, float(np.sqrt(max(var, 0.0)))


def score_pair_form(
    matrix: np.ndarray,
    row_lookup: np.ndarray,
    walk_start: np.ndarray,
    walk_end: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Pair-form AUC for one phenotype against one (graph, variant, t) walk sample.

    `row_lookup` maps a global node id to this phenotype's row, or -1 when the gene has no
    measurement on this strand; it is an array rather than a dict because the membership
    test runs over 600,000 walks for every graph, variant, walk length and phenotype.
    Walks whose endpoints are not both phenotyped, and self-pairs, are dropped. `baseline`
    is the precomputed similarity of the background pairs, which were drawn from the same
    phenotyped pool on the same graph and are shared across variants and walk lengths.
    """
    left = row_lookup[walk_start]
    right = row_lookup[walk_end]
    keep = (left >= 0) & (right >= 0) & (walk_start != walk_end)
    left, right = left[keep], right[keep]
    if left.size == 0 or baseline.size == 0:
        return {"auc": float("nan"), "auc_se": float("nan"), "n_pairs": int(left.size)}
    if left.size > PAIR_CAP:
        pick = rng.choice(left.size, size=PAIR_CAP, replace=False)
        left, right = left[pick], right[pick]
    adjacent = pair_similarity(matrix, left, right)
    auc, se = pair_auc(adjacent, baseline)
    return {
        "auc": auc,
        "auc_se": se,
        "n_pairs": int(left.size),
        "mean_similarity_adjacent": float(adjacent.mean()),
        "mean_similarity_background": float(baseline.mean()),
    }


# --------------------------------------------------------------------------------------
# Figure.
# --------------------------------------------------------------------------------------

SHORT_GRAPH = {
    "physical_interaction": "physical",
    "regulatory_interaction": "regulatory",
    "tflink": "tflink",
    "string12_0_neighborhood": "string neighborhood",
    "string12_0_fusion": "string fusion",
    "string12_0_cooccurence": "string cooccurence",
    "string12_0_coexpression": "string coexpression",
    "string12_0_experimental": "string experimental",
    "string12_0_database": "string database",
}


def _box(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    ax.tick_params(width=0.5, length=2, labelsize=5)
    ax.set_axisbelow(True)


def _phenotype_panel(
    ax: Axes, results: dict[str, Any], phenotype: str, names: list[str]
) -> None:
    """Nine graphs, real against both controls, in the pair form. Bars are FLUSH.

    Three bars share a unit slot at height 0.28 with centers 0.28 apart, so they touch
    without overlapping; the earlier 0.30-at-0.24 geometry drew them on top of each other.
    """
    block = results["pair_form"].get(phenotype)
    ys = np.arange(len(names))
    if block is None or block.get("status") != "ok":
        ax.text(
            0.5,
            0.5,
            f"{phenotype}\nnot run",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6,
        )
        ax.set_yticks([])
        return
    series = [
        ("graph", "auc", "auc_se", PLOT_PALETTE[0], 0.28),
        ("degree-preserving rewire", "auc_rewired", "auc_se_rewired",
         PLOT_PALETTE[2], 0.0),
        ("random, matched density", "auc_random", "auc_se_random",
         PLOT_PALETTE[5], -0.28),
    ]
    # BARS GROW FROM 0.5, NOT FROM ZERO. Every value sits within a few hundredths of
    # chance, so a bar anchored at zero encodes the interesting part in the last 5 % of its
    # length and the nine graphs look identical. Anchoring at the null puts the whole bar
    # on the quantity being read. The axis stays in AUC units.
    for label, key, se_key, color, offset in series:
        vals = [block["graphs"][g]["t1"][key] - 0.5 for g in names]
        errs = [block["graphs"][g]["t1"][se_key] for g in names]
        ax.barh(
            ys + offset,
            vals,
            left=0.5,
            height=0.28,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label=label,
            zorder=3,
            xerr=np.array(errs) * 1.96,
            error_kw={"elinewidth": 0.35, "ecolor": "black", "capsize": 0},
        )
    ax.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT_GRAPH[n] for n in names], fontsize=5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    # The x range is per panel because the effects differ by an order of magnitude across
    # strands, and a shared axis would flatten every panel to fit the one large one. It
    # always spans 0.5 and always contains every bar, so no bar is clipped.
    keys = ("auc", "auc_rewired", "auc_random")
    vals_all = [block["graphs"][g]["t1"][k] for g in names for k in keys]
    lo = min(min(vals_all), 0.5) - 0.01
    hi = max(max(vals_all), 0.5) + 0.01
    ax.set_xlim(lo, hi)
    step = 0.02 if hi - lo < 0.10 else (0.05 if hi - lo < 0.25 else 0.10)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlabel("pair-form AUC ($t = 1$)")
    ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
    _box(ax)


def make_figure(results: dict[str, Any], out_png: str, out_svg: str) -> None:
    # EVERY size is set explicitly rather than left to resolve from `font.size`. The
    # relative sizes (`medium`, `large`) are what a repo mplstyle or a stale rc can
    # override, and the failure is silent: the layout is right and the type is twice the
    # Nature minimum, which is only obvious once a label runs off its panel.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 4.5,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )
    names = [g for g in GRAPH_NAMES if g in results["graphs"]]
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(152.0)),
        constrained_layout=True,
    )
    flat = axes.ravel()

    # (a) to (f): one panel per phenotype strand, pair form.
    for i, phenotype in enumerate(PHENOTYPE_ORDER):
        ax = flat[i]
        _phenotype_panel(ax, results, phenotype, names)
        ax.set_title(PHENOTYPE_TITLES[phenotype], loc="left", fontsize=5.5)
        panel_label(ax, "abcdef"[i])
    flat[0].legend(fontsize=4.2, loc="lower right", frameon=False)

    # (g) calibration: the two statistics on the one strand where both are defined.
    ax = flat[6]
    expr = results["pair_form"]["expression"]
    reporter = [results["graphs"][g]["t1"]["auc"] for g in names]
    pair = [expr["graphs"][g]["t1"]["auc"] for g in names]
    ys = np.arange(len(names))
    ax.barh(ys + 0.21, np.array(reporter) - 0.5, left=0.5, height=0.42,
            color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.35,
            label="reporter form", zorder=3)
    ax.barh(ys - 0.21, np.array(pair) - 0.5, left=0.5, height=0.42,
            color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.35,
            label="pair form", zorder=3)
    ax.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT_GRAPH[n] for n in names], fontsize=5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    lo = min(min(reporter), min(pair), 0.5) - 0.01
    hi = max(max(reporter), max(pair), 0.5) + 0.01
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("AUC ($t = 1$), expression")
    ax.legend(fontsize=4.2, loc="lower right", frameon=False)
    ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
    _box(ax)
    ax.set_title("Two statistics, one strand (calibration)", loc="left", fontsize=5.5)
    panel_label(ax, "g")

    # (h) direction, reporter form, for the two graphs where orientation exists.
    ax = flat[7]
    directed = [g for g in names if results["graphs"][g].get("directed")]
    if directed:
        ys = np.arange(len(directed))
        series = [
            ("TF $\\to$ target", "auc_transposed", PLOT_PALETTE[1], 0.33),
            ("deployed (symmetric)", "auc", PLOT_PALETTE[0], 0.11),
            ("target $\\to$ TF", "auc_forward", PLOT_PALETTE[3], -0.11),
            ("rewired", "auc_rewired", PLOT_PALETTE[2], -0.33),
        ]
        vals_all = []
        for label, key, color, offset in series:
            vals = [results["graphs"][g]["t1"][key] for g in directed]
            vals_all += vals
            ax.barh(ys + offset, np.array(vals) - 0.5, left=0.5, height=0.22,
                    color=color, edgecolor="black", linewidth=0.35, label=label,
                    zorder=3)
        ax.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([SHORT_GRAPH[g] for g in directed], fontsize=5)
        ax.set_ylim(-0.55, len(directed) - 0.45)
        lo = min(min(vals_all), 0.5) - 0.01
        hi = max(max(vals_all), 0.5) + 0.01
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MultipleLocator(0.02))
        ax.legend(fontsize=4.2, loc="lower right", frameon=False)
    ax.set_xlabel("reporter-form AUC ($t = 1$), expression")
    ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
    _box(ax)
    ax.set_title("Direction, reporter form (the mask symmetrizes it)", loc="left",
                 fontsize=5.5)
    panel_label(ax, "h")

    # (i) walk length, pair form, every strand. Plotted as the excess over the
    # degree-preserving control, which is the quantity that is actually read.
    ax = flat[8]
    for i, phenotype in enumerate(PHENOTYPE_ORDER):
        block = results["pair_form"].get(phenotype)
        if block is None or block.get("status") != "ok":
            continue
        vals = []
        for t in WALK_LENGTHS:
            per = [
                block["graphs"][g][f"t{t}"]["auc"]
                - block["graphs"][g][f"t{t}"]["auc_rewired"]
                for g in names
            ]
            vals.append(float(np.nanmean(per)))
        ax.plot(list(WALK_LENGTHS), vals, marker="o", markersize=2.5, linewidth=0.8,
                color=PLOT_PALETTE[i % 18], label=phenotype.replace("_", " "))
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlabel("walk length $t$")
    ax.set_ylabel("AUC excess over rewired\n(mean over the nine graphs)")
    ax.set_xticks(list(WALK_LENGTHS))
    ax.legend(fontsize=4.2, ncol=2, frameon=False, loc="best")
    ax.grid(linewidth=0.3, color="#DDDDDD")
    _box(ax)
    ax.set_title("Walk length, excess over the rewired control", loc="left",
                 fontsize=5.5)
    panel_label(ax, "i")

    fig.savefig(out_png, dpi=400)
    savefig_true_size_svg(fig, out_svg)
    plt.close(fig)


# --------------------------------------------------------------------------------------


def main() -> None:
    # --figure-only redraws from the committed JSON without recomputing. The walk and the
    # controls take tens of minutes on the two million-edge graphs, and figure layout wants
    # more iterations than that allows.
    figure_only = "--figure-only" in sys.argv
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]

    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    out_dir = osp.join(images_dir, EXPERIMENT)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    png = osp.join(out_dir, "graph_prior_probe.png")
    svg = osp.join(out_dir, "graph_prior_probe.svg")
    json_path = osp.join(results_dir, "graph_prior_probe.json")

    if figure_only:
        with open(json_path) as fh:
            existing = json.load(fh)
        make_figure(existing, png, svg)
        print(f"redrew from {json_path}\n-> {svg}")
        return

    deleted, reporters, expression_signed = load_expression(data_root)
    response = np.abs(expression_signed)
    print(f"deleted genes {len(deleted)}  reporters {len(reporters)}  "
          f"response {response.shape}", flush=True)

    # Every phenotype strand, each keyed by systematic ORF. A strand whose source is
    # missing is recorded as not run rather than estimated.
    loaders: dict[str, Callable[[], tuple[list[str], np.ndarray]]] = {
        "expression": lambda: (deleted, expression_signed),
        "morphology": lambda: load_morphology(data_root),
        "amino_acid": lambda: load_amino_acid(data_root),
        "betaxanthin": lambda: load_betaxanthin(data_root),
        "beta_carotene": lambda: load_beta_carotene(data_root),
        "fitness": lambda: load_fitness(data_root),
    }
    phenotypes: dict[str, dict[str, Any]] = {}
    for name, loader in loaders.items():
        genes, matrix = loader()
        phenotypes[name] = {
            "genes": genes,
            "matrix": _zscore(np.asarray(matrix, dtype=np.float64)),
        }
        print(f"  {name:14s} genes {len(genes):5d}  features "
              f"{phenotypes[name]['matrix'].shape[1]:5d}", flush=True)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    scer = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(scer, BUILDER_NAMES)

    # One node index shared by every graph and every phenotype, so an AUC is comparable
    # across them.
    all_pheno_genes = set()
    for block in phenotypes.values():
        all_pheno_genes |= set(block["genes"])
    nodes = sorted(set(genome.gene_set) | set(deleted) | set(reporters) | all_pheno_genes)
    index = {g: i for i, g in enumerate(nodes)}
    deleted_cols = np.array([index[g] for g in deleted], dtype=np.int64)
    reporter_rows = np.array([index[g] for g in reporters], dtype=np.int64)
    print(f"node index {len(nodes)}", flush=True)

    for name, block in phenotypes.items():
        lookup = np.full(len(nodes), -1, dtype=np.int64)
        lookup[[index[g] for g in block["genes"]]] = np.arange(len(block["genes"]))
        block["row_lookup"] = lookup
        # A NAMESPACE MISMATCH IS THE FAILURE MODE THIS PROBE CANNOT SURVIVE SILENTLY: a
        # gene set that does not intersect the graphs returns a clean AUC of 0.5 on every
        # panel and reads as a biological null. The morphology mirror stores lowercase
        # ORFs and did exactly this. Fail loudly instead.
        shared = len(set(block["genes"]) & set(genome.gene_set))
        fraction = shared / len(block["genes"])
        print(f"  {name:14s} genes in the genome gene set: {shared}/"
              f"{len(block['genes'])} ({fraction:.1%})", flush=True)
        if fraction < 0.5:
            raise RuntimeError(
                f"{name}: only {fraction:.1%} of its genes are in the genome gene set, "
                "which means the identifiers are in the wrong namespace"
            )

    rng = np.random.default_rng(RANDOM_SEED)
    out: dict[str, Any] = {
        "n_deleted": len(deleted),
        "n_reporters": len(reporters),
        "n_nodes": len(nodes),
        "responder_quantile": RESPONDER_QUANTILE,
        "responder_quantile_alt": RESPONDER_QUANTILE_ALT,
        "walk_lengths": list(WALK_LENGTHS),
        "graphs": {},
    }
    pair_out: dict[str, dict[str, Any]] = {
        name: {
            "status": "ok",
            "n_genes": len(block["genes"]),
            "n_features": int(block["matrix"].shape[1]),
            "similarity": (
                "negative absolute difference"
                if block["matrix"].shape[1] == 1
                else "across-feature Pearson"
            ),
            "graphs": {},
        }
        for name, block in phenotypes.items()
    }

    for name in GRAPH_NAMES:
        gene_graph = multigraph.graphs.get(CONFIG_TO_BUILDER.get(name, name))
        if gene_graph is None:
            print(f"  {name}: absent from the multigraph, skipped", flush=True)
            continue
        graph = gene_graph.graph
        per_graph: dict[str, Any] = {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
        }
        rewired_graph, degree_preserved = rewired(graph, rng)
        per_graph["rewired_degree_rank_corr"] = degree_preserved
        variants = {
            "": graph,
            "_rewired": rewired_graph,
            "_random": random_matched(graph, rng),
        }
        # The random control is generated with integer node labels, so it is relabeled
        # onto this graph's own node names to share the index.
        node_list = list(graph.nodes())
        variants["_random"] = nx.relabel_nodes(
            variants["_random"], {i: node_list[i] for i in range(len(node_list))}
        )
        per_graph["directed"] = bool(graph.is_directed())

        # Pair-form background: uniform pairs over the genes this graph carries, drawn once
        # and shared by every variant and every walk length, so the comparisons that get
        # read differ only in the adjacency they came from.
        graph_nodes = np.array(
            sorted({index[g] for g in graph.nodes() if g in index}), dtype=np.int64
        )
        # The similarity of those background pairs is computed once here too: it does not
        # depend on the variant or on t, and recomputing it would add sampling noise to
        # exactly the real-vs-rewired contrast the figure reads.
        backgrounds: dict[str, np.ndarray] = {}
        pools: dict[str, np.ndarray] = {}
        for pname, block in phenotypes.items():
            pools[pname] = graph_nodes[block["row_lookup"][graph_nodes] >= 0]
            rows = block["row_lookup"][graph_nodes]
            rows = rows[rows >= 0]
            if rows.size < 2:
                backgrounds[pname] = np.empty(0, dtype=np.float64)
                continue
            n_draw = min(BACKGROUND_DRAWS, PAIR_CAP)
            a = rng.integers(0, rows.size, size=n_draw)
            b = rng.integers(0, rows.size, size=n_draw)
            keep = a != b
            backgrounds[pname] = pair_similarity(
                block["matrix"], rows[a[keep]], rows[b[keep]]
            )

        for suffix, variant in variants.items():
            adjacency = row_normalized(variant, index)

            # --- reporter form (expression) -------------------------------------------
            blocks = walk_columns(adjacency, deleted_cols, WALK_LENGTHS)
            for t, block_mat in blocks.items():
                weights = block_mat[reporter_rows, :]
                auc, rho, frac = score(weights, response, RESPONDER_QUANTILE)
                key = f"t{t}"
                per_graph.setdefault(key, {})
                per_graph[key][f"auc{suffix}"] = auc
                per_graph[key][f"spearman{suffix}"] = rho
                per_graph[key][f"scorable_fraction{suffix}"] = frac
                if suffix == "" and t == 1:
                    auc_alt, _, _ = score(weights, response, RESPONDER_QUANTILE_ALT)
                    per_graph[key]["auc_alt_quantile"] = auc_alt

            # --- pair form (every phenotype) ------------------------------------------
            for t in WALK_LENGTHS:
                for pname, block in phenotypes.items():
                    start, end = sample_walk_pairs(
                        adjacency, pools[pname], t, WALK_DRAWS, rng
                    )
                    stats = score_pair_form(
                        block["matrix"],
                        block["row_lookup"],
                        start,
                        end,
                        backgrounds[pname],
                        rng,
                    )
                    entry = pair_out[pname]["graphs"].setdefault(name, {})
                    slot = entry.setdefault(f"t{t}", {})
                    if suffix == "":
                        slot.update(stats)
                    else:
                        slot[f"auc{suffix}"] = stats["auc"]
                        slot[f"auc_se{suffix}"] = stats["auc_se"]
                        slot[f"n_pairs{suffix}"] = stats["n_pairs"]
                    if suffix == "":
                        entry["n_nodes"] = int(graph.number_of_nodes())
                        entry["n_edges"] = int(graph.number_of_edges())
                        entry["directed"] = bool(graph.is_directed())
                        entry["rewired_degree_rank_corr"] = degree_preserved

            if suffix == "":
                # ORIENTATION IS LOAD-BEARING for the two directed graphs (regulatory,
                # tflink) and the deployed answer is SYMMETRIC. `_build_attention_mask` in
                # equivariant_cell_graph_transformer.py sets both
                # `head_mask[edge_index[0]+1, edge_index[1]+1]` and its transpose, so
                # whatever direction `to_cell_data` emitted is symmetrized before the model
                # ever sees it.
                #
                # The symmetric orientation is therefore the one that answers "is the
                # DEPLOYED prior sound", and it is the primary number. `forward` follows
                # edges as stored, `transposed` follows them the other way, and for a TF
                # graph stored as TF -> target only ONE of those is "deleting this TF
                # perturbs its targets". For the seven undirected graphs all three
                # coincide, which makes the extra columns self-checking.
                directed_adj = row_normalized(variant, index, symmetrize=False)
                for label, matrix in (
                    ("forward", directed_adj),
                    ("transposed", directed_adj.T.tocsr()),
                ):
                    for t, block_mat in walk_columns(
                        matrix, deleted_cols, WALK_LENGTHS
                    ).items():
                        weights = block_mat[reporter_rows, :]
                        auc, _rho, _frac = score(weights, response, RESPONDER_QUANTILE)
                        per_graph[f"t{t}"][f"auc_{label}"] = auc
                    # The pair form is direction-blind at t = 1 by construction (see the
                    # module docstring); computed anyway so that is visible in the numbers.
                    for t in WALK_LENGTHS:
                        for pname, block in phenotypes.items():
                            start, end = sample_walk_pairs(
                                matrix, pools[pname], t, WALK_DRAWS, rng
                            )
                            stats = score_pair_form(
                                block["matrix"],
                                block["row_lookup"],
                                start,
                                end,
                                backgrounds[pname],
                                rng,
                            )
                            pair_out[pname]["graphs"][name][f"t{t}"][
                                f"auc_{label}"
                            ] = stats["auc"]

        out["graphs"][name] = per_graph
        t1 = per_graph["t1"]
        print(
            f"  {name:26s} edges={per_graph['n_edges']:>8,}  "
            f"reporter AUC t1 sym {t1['auc']:.4f}  fwd {t1['auc_forward']:.4f}  "
            f"rev {t1['auc_transposed']:.4f}  rewired {t1['auc_rewired']:.4f}",
            flush=True,
        )
        for pname in PHENOTYPE_ORDER:
            entry = pair_out[pname]["graphs"][name]["t1"]
            print(
                f"      pair {pname:14s} AUC {entry['auc']:.4f} "
                f"(+-{entry['auc_se']:.4f})  rewired {entry['auc_rewired']:.4f}  "
                f"random {entry['auc_random']:.4f}  n={entry['n_pairs']:,}",
                flush=True,
            )

    out["pair_form"] = pair_out
    out["pair_form_config"] = {
        "walk_draws": WALK_DRAWS,
        "background_draws": BACKGROUND_DRAWS,
        "pair_cap": PAIR_CAP,
        "note": (
            "adjacent pairs are t-step random-walk endpoints; background pairs are "
            "uniform draws from the same phenotyped gene pool on the same graph. The walk "
            "is degree-biased and the background is not, so the read is real vs rewired, "
            "not real vs 0.5."
        ),
    }
    make_figure(out, png, svg)
    out["figure"] = {"png": png, "svg": svg, "written_at": timestamp()}
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {svg}")


if __name__ == "__main__":
    main()
