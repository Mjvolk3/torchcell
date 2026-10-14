# experiments/025-solid-growth/scripts/wt_gene_embedding_clusters.py
# [[experiments.025-solid-growth.scripts.wt_gene_embedding_clusters]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/wt_gene_embedding_clusters
"""Do the wild-type gene representations of the S3 closure transformer cluster by GO?

The model is the 025 "S3 closure, joint fitness 1.0" cell graph transformer
(``conf/cgt_s3_r_kl_fit_031.yaml``), seed 2, W&B run 8fuwsk48 (IGB job 2409672), at its
best-validation-Pearson checkpoint (epoch 43, val gene-interaction Pearson 0.5009). The
checkpoint and the run's resolved config were copied from IGB to
``$DATA_ROOT/models/025-s3-closure/8fuwsk48/`` with a sha256 manifest beside them; this
script refuses to run if the bytes no longer match the manifest.

WHAT IS EXTRACTED. With ``attention_mask.enabled: false`` the encoder sees only the gene
table: it runs once, at batch 1, on ``[CLS; preprocessor(table[arange(N)])]`` and nothing
from the strain enters until the perturbation operator afterwards
([[experiments.010-kuzmin-tmi.additive-baseline-analysis]], "What the 010 transformer
actually computes"). Two matrices therefore describe every gene independently of any
strain:

  table     ``model.gene_embedding.weight``         [6607, 180], the learnable input
  encoder   ``representations["H_genes"]``           [6607, 180], after the 8th layer

LOAD CHECK, so a wrong gene order or a mis-loaded weight cannot pass silently. The cell
graph is rebuilt from the genome exactly as the trainer builds it (the base graph from
``genome.gene_set`` fixes the node order), its nine edge counts are asserted against the
cell graph W&B recorded for this run, the state dict loads with ``strict=True``, and the
model re-scores the 37,673 pinned validation triples; the Pearson must reproduce the
checkpoint's recorded 0.5009 to within 0.002 (DDP pads the last validation batch, so the
match is not bit-exact).

CLUSTERING (same recipe for both matrices). Center, PCA to 50 components, kNN graph with
k = 15 (Euclidean, symmetrized union), Leiden on modularity at resolution 1.0 (python-igraph
``community_leiden``, Python ``random`` seeded 0). UMAP of the same 50 components to 2-D
for display only (n_neighbors 15, min_dist 0.3, random_state 0). Silhouette is computed in
the 50-component space.

GO ENRICHMENT. Annotations are the SGD R64 GFF ``Ontology_term`` GO ids the genome object
carries per gene (``genome[g].go``), propagated to every ``is_a`` ancestor in
``$DATA_ROOT/data/go/go.obo`` (goatools). Terms with 10 to 500 annotated genes in the 6,607
gene universe are tested. Per cluster and term, a one-sided hypergeometric test; one
Benjamini-Hochberg correction over every (cluster, term) test of a representation. A
cluster is "GO-enriched" when its best term has q < 0.05.

Outputs: ``results/wt_gene_embeddings/`` (two .npy matrices, gene order, cluster CSV,
enrichment CSV, summary JSON) and three third-width panels under
``$ASSET_IMAGES_DIR/025-solid-growth/``.

    python experiments/025-solid-growth/scripts/wt_gene_embedding_clusters.py           # all
    python experiments/025-solid-growth/scripts/wt_gene_embedding_clusters.py --analyze-only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import os.path as osp
import random
import re

import matplotlib

matplotlib.use("Agg")
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.colors import to_rgba
from pydantic import BaseModel
from scipy.stats import hypergeom, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RUN_ID = "8fuwsk48"
MODEL_DIR = osp.join(DATA_ROOT, "models", "025-s3-closure", RUN_ID)
RESULTS_025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
OUT_DIR = osp.join(RESULTS_025, "wt_gene_embeddings")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")
BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build"
)
RECAP = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)

RECORDED_VAL_PEARSON = 0.5009
VAL_TOLERANCE = 0.002
SCORE_BATCH = 512

N_PCS = 50
K_NN = 15
LEIDEN_RESOLUTION = 1.0
SEED = 0
GO_MIN, GO_MAX = 10, 500
Q_THRESHOLD = 0.05
N_HIGHLIGHT = 5  # clusters drawn in the five non-gray primaries


# ---------------------------------------------------------------- records
class CheckpointFile(BaseModel):
    """The copied checkpoint: where it came from and its hash."""

    file: str
    sha256: str
    bytes: int
    source_host: str
    source_path: str
    retrieval_command: str


class ConfigFile(BaseModel):
    """The run's resolved config as stored beside the checkpoint."""

    file: str
    sha256: str
    source: str


class CheckpointManifest(BaseModel):
    """manifest.json beside the checkpoint under $DATA_ROOT/models."""

    run_id: str
    wandb_run_url: str
    config_name: str
    overrides: list[str]
    igb_job_id: str
    igb_code_worktree: str
    selection: str
    checkpoint: CheckpointFile
    config: ConfigFile
    retrieved_at: str


class LoadCheck(BaseModel):
    """Evidence that the weights and the gene order loaded correctly."""

    n_tensors: int
    edge_counts: dict[str, int]
    val_n: int
    val_pearson_recomputed: float
    val_pearson_recorded: float
    abs_diff: float


class ClusterTop(BaseModel):
    """One cluster and its best GO term (smallest hypergeometric p)."""

    cluster: int
    size: int
    go_id: str
    go_name: str
    namespace: str
    k: int
    K: int
    fold: float
    p: float
    q: float
    n_in_train_triple: int  # cluster genes that occur in a pinned TRAINING triple


class RepresentationResult(BaseModel):
    """Clustering and enrichment outcome for one gene matrix."""

    name: str
    dim: int
    n_clusters: int
    modularity: float
    silhouette: float
    n_enriched_clusters: int
    n_genes_in_enriched_clusters: int
    n_significant_tests: int
    clusters: list[ClusterTop]


class Summary(BaseModel):
    """Everything summary.json records."""

    manifest: CheckpointManifest
    load_check: LoadCheck
    n_genes: int
    clustering: dict[str, object]
    go: dict[str, object]
    representations: list[RepresentationResult]
    ari_table_vs_encoder: float


# ---------------------------------------------------------------- model side
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> CheckpointManifest:
    with open(osp.join(MODEL_DIR, "manifest.json")) as f:
        man = CheckpointManifest.model_validate(json.load(f))
    for rec in (man.checkpoint, man.config):
        got = sha256(osp.join(MODEL_DIR, rec.file))
        assert got == rec.sha256, f"{rec.file}: sha256 {got} != manifest {rec.sha256}"
    return man


def recorded_edge_counts(cfg: dict) -> dict[str, int]:
    """The cell graph W&B stored for the run is a repr string; read its edge counts."""
    text = str(cfg["cell_graph"])
    found = re.findall(r"\(gene, (\w+), gene\)=\{\s*edge_index=\[2, (\d+)\]", text)
    assert len(found) == 9, (
        f"expected nine relations in the recorded cell graph, {found}"
    )
    return {rel: int(n) for rel, n in found}


def build_model(cfg: dict):
    """Model + cell graph, built the way equivariant_cell_graph_transformer.py builds them."""
    from sortedcontainers import SortedDict

    from torchcell.data.cell_data import to_cell_data
    from torchcell.data.neo4j_cell import create_graph_from_gene_set
    from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
    from torchcell.graph import GeneMultiGraph, SCerevisiaeGraph, build_gene_multigraph
    from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
    from torchcell.sequence import GeneSet
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(
        graph=graph, graph_names=cfg["cell_dataset"]["graphs"]
    )
    graphs_dict = SortedDict(multigraph.graphs.copy())
    graphs_dict["base"] = create_graph_from_gene_set(GeneSet(genome.gene_set))
    cell_graph = to_cell_data(GeneMultiGraph(graphs=graphs_dict), incidence_graphs=None)
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=cfg["cell_dataset"]["node_embeddings"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )

    edge_counts = {
        et[1]: int(cell_graph[et].num_edges)
        for et in cell_graph.edge_types
        if et[0] == "gene" and et[2] == "gene"
    }
    recorded = recorded_edge_counts(cfg)
    assert edge_counts == recorded, (
        f"cell graph edges {edge_counts} != recorded {recorded}"
    )

    m = cfg["model"]
    lam = cfg["regression_task"]["loss"]["graph_regularization"]["lambda"]
    model = CellGraphTransformer(
        gene_num=m["gene_num"],
        hidden_channels=m["hidden_channels"],
        num_transformer_layers=m["num_transformer_layers"],
        num_attention_heads=m["num_attention_heads"],
        cell_graph=cell_graph,
        graph_regularization_config=m["graph_regularization"],
        perturbation_head_config=m["perturbation_head"],
        dropout=m["dropout"],
        graph_reg_lambda=0.0 if lam is None else float(lam),
        node_embeddings=node_embeddings,
        learnable_embedding_config=m.get("learnable_embedding"),
        attention_mask_config=m.get("attention_mask"),
        heads_config=m.get("heads"),
        perturb_cls=bool(m.get("perturb_cls", False)),
        perturbation_head_cls=str(m.get("perturbation_head_cls", "wildtype")),
    )
    return model, cell_graph, genome, edge_counts


def load_weights(model, ckpt_path: str) -> int:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return len(state)


def val_triples(node_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(
        osp.join(RESULTS_025, "pinned_splits_from_010_seed_42.json.gz"), "rt"
    ) as f:
        val_ids = np.array(sorted(json.load(f)["pinned"]["val"]), dtype=np.int64)
    labels = pd.read_parquet(
        osp.join(BUILD, "processed", "label_df.parquet"),
        columns=["index", "gene_interaction"],
    ).set_index("index")
    y = labels.loc[val_ids, "gene_interaction"].to_numpy(dtype=np.float64)
    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"])
    genes3 = recap.set_index("idx_025").loc[val_ids, ["gene_a", "gene_b", "gene_c"]]
    col = {g: i for i, g in enumerate(node_ids)}
    idx = np.vectorize(col.__getitem__)(genes3.to_numpy()).astype(np.int64)
    return idx, y


def score_val(model, cell_graph, idx3: np.ndarray) -> np.ndarray:
    import torch

    out = np.empty(idx3.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, idx3.shape[0], SCORE_BATCH):
            chunk = idx3[start : start + SCORE_BATCH]
            n = chunk.shape[0]
            batch = {
                "gene": type(
                    "G",
                    (),
                    {
                        "perturbation_indices": torch.from_numpy(chunk.reshape(-1)),
                        "perturbation_indices_batch": torch.arange(n).repeat_interleave(
                            3
                        ),
                    },
                )()
            }
            preds, _ = model(cell_graph, batch, return_attention=False)
            out[start : start + n] = preds.squeeze(-1).double().numpy()
    return out


def extract(man: CheckpointManifest) -> LoadCheck:
    import torch
    import yaml

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    with open(osp.join(MODEL_DIR, man.config.file)) as f:
        cfg = yaml.safe_load(f)
    model, cell_graph, _, edge_counts = build_model(cfg)
    n_tensors = load_weights(model, osp.join(MODEL_DIR, man.checkpoint.file))
    model.eval()
    node_ids = [str(g) for g in cell_graph["gene"].node_ids]
    assert len(node_ids) == cfg["model"]["gene_num"] == 6607

    # One wild-type pass. The dummy perturbation only feeds the post-encoder operator;
    # H_genes is computed before it and does not depend on it.
    with torch.no_grad():
        dummy = {
            "gene": type(
                "G",
                (),
                {
                    "perturbation_indices": torch.tensor([0]),
                    "perturbation_indices_batch": torch.tensor([0]),
                },
            )()
        }
        _, rep = model(cell_graph, dummy, return_attention=False)
    H = rep["H_genes"].double().numpy()
    table = model.gene_embedding.weight.detach().double().numpy()

    idx3, y = val_triples(node_ids)
    p = score_val(model, cell_graph, idx3)
    r = float(pearsonr(y, p)[0])
    check = LoadCheck(
        n_tensors=n_tensors,
        edge_counts=edge_counts,
        val_n=int(y.size),
        val_pearson_recomputed=r,
        val_pearson_recorded=RECORDED_VAL_PEARSON,
        abs_diff=abs(r - RECORDED_VAL_PEARSON),
    )
    print(check.model_dump_json(indent=1))
    assert check.abs_diff < VAL_TOLERANCE, (
        "recomputed val Pearson does not reproduce the checkpoint"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(osp.join(OUT_DIR, "encoder_H_genes.npy"), H.astype(np.float32))
    np.save(osp.join(OUT_DIR, "gene_table.npy"), table.astype(np.float32))
    with open(osp.join(OUT_DIR, "gene_order.json"), "w") as f:
        json.dump(node_ids, f)
    with open(osp.join(OUT_DIR, "load_check.json"), "w") as f:
        f.write(check.model_dump_json(indent=2))
    return check


# ---------------------------------------------------------------- analysis side
def go_matrix(
    node_ids: list[str],
) -> tuple[sp.csr_matrix, list[str], dict[str, tuple[str, str]], dict]:
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    dag = genome.go_dag
    gene_terms: list[set[str]] = []
    n_direct = 0
    skipped: set[str] = set()
    for g in node_ids:
        direct = genome[g].go or []
        n_direct += len(direct)
        terms: set[str] = set()
        for t in direct:
            if t not in dag or dag[t].is_obsolete:
                skipped.add(t)
                continue
            terms.add(dag[t].id)
            terms.update(dag[t].get_all_parents())
        gene_terms.append(terms)
    counts: dict[str, int] = {}
    for terms in gene_terms:
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
    kept = sorted(t for t, c in counts.items() if GO_MIN <= c <= GO_MAX)
    col = {t: j for j, t in enumerate(kept)}
    rows, cols = [], []
    for i, terms in enumerate(gene_terms):
        for t in terms:
            if t in col:
                rows.append(i)
                cols.append(col[t])
    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(node_ids), len(kept)),
    )
    info = {t: (dag[t].name, dag[t].namespace) for t in kept}
    stats = {
        "genes_with_any_term": int(sum(1 for t in gene_terms if t)),
        "direct_annotations": n_direct,
        "terms_skipped_not_in_dag_or_obsolete": len(skipped),
        "terms_after_propagation": len(counts),
        "terms_tested": len(kept),
    }
    return A, kept, info, stats


def cluster(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PCA-50 coordinates, Leiden labels, UMAP-2 coordinates, modularity."""
    import umap

    Z = PCA(n_components=N_PCS, random_state=SEED).fit_transform(X - X.mean(axis=0))
    nn = NearestNeighbors(n_neighbors=K_NN + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    edges = {(min(i, j), max(i, j)) for i in range(len(Z)) for j in ind[i, 1:]}
    g = ig.Graph(n=len(Z), edges=sorted(edges))
    random.seed(SEED)
    part = g.community_leiden(
        objective_function="modularity", resolution=LEIDEN_RESOLUTION, n_iterations=-1
    )
    labels = np.array(part.membership, dtype=np.int64)
    # Renumber by size, largest first, so cluster ids are stable and readable.
    order = np.argsort(-np.bincount(labels), kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    labels = remap[labels]
    U = umap.UMAP(n_neighbors=K_NN, min_dist=0.3, random_state=SEED).fit_transform(Z)
    return Z, labels, U, float(g.modularity(labels))


def train_triple_genes() -> set[str]:
    """Genes occurring in at least one pinned training triple (S0 train split).

    A descriptive label only: S3 also trains on every single and the pair-in-triple
    doubles, so a gene outside this set can still appear in a training single.
    """
    with gzip.open(
        osp.join(RESULTS_025, "pinned_splits_from_010_seed_42.json.gz"), "rt"
    ) as f:
        train_ids = np.array(sorted(json.load(f)["pinned"]["train"]), dtype=np.int64)
    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"])
    genes3 = recap.set_index("idx_025").loc[train_ids, ["gene_a", "gene_b", "gene_c"]]
    return set(genes3.to_numpy().ravel().tolist())


def enrich(
    labels: np.ndarray,
    A: sp.csr_matrix,
    terms: list[str],
    info: dict,
    name: str,
    in_train: np.ndarray,
) -> tuple[pd.DataFrame, list[ClusterTop]]:
    N = A.shape[0]
    K = np.asarray(A.sum(axis=0)).ravel()
    blocks = []
    for c in range(labels.max() + 1):
        mask = labels == c
        n = int(mask.sum())
        k = np.asarray(A[mask].sum(axis=0)).ravel()
        p = hypergeom.sf(k - 1, N, K, n)
        blocks.append(
            pd.DataFrame(
                {"cluster": c, "size": n, "go_id": terms, "k": k, "K": K, "p": p}
            )
        )
    df = pd.concat(blocks, ignore_index=True)
    df["q"] = multipletests(df["p"].to_numpy(), method="fdr_bh")[1]
    df["fold"] = (df["k"] / df["size"]) / (df["K"] / N)
    df["go_name"] = [info[t][0] for t in df["go_id"]]
    df["namespace"] = [info[t][1] for t in df["go_id"]]
    df.insert(0, "representation", name)
    tops = []
    for c, sub in df.groupby("cluster"):
        best = sub.sort_values(["p", "fold"], ascending=[True, False]).iloc[0]
        tops.append(
            ClusterTop(
                cluster=int(c),
                size=int(best["size"]),
                go_id=str(best["go_id"]),
                go_name=str(best["go_name"]),
                namespace=str(best["namespace"]),
                k=int(best["k"]),
                K=int(best["K"]),
                fold=float(best["fold"]),
                p=float(best["p"]),
                q=float(best["q"]),
                n_in_train_triple=int(in_train[labels == c].sum()),
            )
        )
    return df, tops


# ---------------------------------------------------------------- figures
def style() -> None:
    from torchcell.utils import apply_paper_style

    apply_paper_style()
    plt.rcParams.update({"legend.fontsize": 5.0, "legend.frameon": False})


def short(name: str, n: int = 34) -> str:
    return name if len(name) <= n else name[: n - 1].rstrip() + "…"


def fmt_q(q: float) -> str:
    if q == 0.0:
        return "q < 1e-300"
    m, e = f"{q:.0e}".split("e")
    return f"q = {m}e{int(e)}"


def highlight_colors(tops: list[ClusterTop]) -> dict[int, str]:
    from torchcell.utils import PLOT_PALETTE

    enriched = sorted(
        (t for t in tops if t.q < Q_THRESHOLD), key=lambda t: (t.p, -t.fold)
    )
    return {t.cluster: PLOT_PALETTE[i] for i, t in enumerate(enriched[:N_HIGHLIGHT])}


def umap_panel(
    U: np.ndarray, labels: np.ndarray, res: RepresentationResult, title: str, stem: str
) -> None:
    from torchcell.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    colors = highlight_colors(res.clusters)
    tops = {t.cluster: t for t in res.clusters}
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(64)))
    ax = fig.add_axes((0.02, 0.27, 0.96, 0.66))
    bg = ~np.isin(labels, list(colors))
    ax.scatter(
        U[bg, 0],
        U[bg, 1],
        s=0.4,
        lw=0,
        color=to_rgba(PLOT_PALETTE[5], 0.25),
        rasterized=True,
    )
    handles = []
    for c, col in colors.items():
        m = labels == c
        ax.scatter(U[m, 0], U[m, 1], s=0.6, lw=0, color=col, rasterized=True)
        cx, cy = np.median(U[m], axis=0)
        ax.text(
            cx,
            cy,
            str(c),
            fontsize=6,
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        t = tops[c]
        handles.append(
            plt.Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=3,
                color=col,
                label=f"{c} {short(t.go_name)} ({fmt_q(t.q)})",
            )
        )
    handles.append(
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3,
            color=to_rgba(PLOT_PALETTE[5], 0.4),
            label=f"other {res.n_clusters - len(colors)} clusters",
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=6, pad=2)
    ax.text(
        0.0,
        0.0,
        "UMAP of 6,607 genes",
        transform=ax.transAxes,
        fontsize=5,
        ha="left",
        va="top",
        color=PLOT_PALETTE[5],
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        handletextpad=0.2,
        borderaxespad=0.3,
        labelspacing=0.25,
    )
    savefig_true_size_svg(fig, stem + ".svg", dpi=600)
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)


def bar_panel(res: RepresentationResult, stem: str) -> None:
    from torchcell.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    colors = highlight_colors(res.clusters)
    tops = sorted(res.clusters, key=lambda t: t.p)
    n = len(tops)
    height_mm = min(14 + 2.6 * n, 170)
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(height_mm)))
    bottom = 9 / height_mm
    ax = fig.add_axes((0.58, bottom, 0.38, 1 - bottom - 6 / height_mm))
    floor = 1e-300
    vals = [-np.log10(max(t.q, floor)) for t in tops]
    ypos = np.arange(n)[::-1]
    ax.barh(
        ypos,
        vals,
        height=0.75,
        color=[colors.get(t.cluster, to_rgba(PLOT_PALETTE[5], 0.35)) for t in tops],
        edgecolor="none",
    )
    ax.axvline(-np.log10(Q_THRESHOLD), color="black", lw=0.5, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{t.cluster} {short(t.go_name, 26)}" for t in tops], fontsize=5
    )
    ax.tick_params(axis="y", length=0, pad=1)
    ax.tick_params(axis="x", length=1.5, width=0.5, pad=1)
    ax.set_xlabel("top term, -log10 q (BH)", labelpad=1)
    ax.set_ylim(-0.6, n - 0.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        f"Best GO term per cluster ({res.n_enriched_clusters} of {n} at q < 0.05)",
        fontsize=6,
        pad=2,
        loc="right",
    )
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main
def analyze(man: CheckpointManifest, check: LoadCheck) -> Summary:
    with open(osp.join(OUT_DIR, "gene_order.json")) as f:
        node_ids = json.load(f)
    mats = {
        "encoder": np.load(osp.join(OUT_DIR, "encoder_H_genes.npy")).astype(np.float64),
        "table": np.load(osp.join(OUT_DIR, "gene_table.npy")).astype(np.float64),
    }
    A, terms, info, go_stats = go_matrix(node_ids)
    print("GO:", go_stats)
    train_genes = train_triple_genes()
    in_train = np.array([g in train_genes for g in node_ids])
    print(f"genes in a training triple: {int(in_train.sum())} of {len(node_ids)}")

    results, assign, enr, umaps, labels_by = [], {}, [], {}, {}
    for name, X in mats.items():
        Z, labels, U, mod = cluster(X)
        sil = float(silhouette_score(Z, labels))
        df, tops = enrich(labels, A, terms, info, name, in_train)
        enriched = [t for t in tops if t.q < Q_THRESHOLD]
        res = RepresentationResult(
            name=name,
            dim=int(X.shape[1]),
            n_clusters=int(labels.max() + 1),
            modularity=mod,
            silhouette=sil,
            n_enriched_clusters=len(enriched),
            n_genes_in_enriched_clusters=int(sum(t.size for t in enriched)),
            n_significant_tests=int((df["q"] < Q_THRESHOLD).sum()),
            clusters=tops,
        )
        print(
            f"{name}: {res.n_clusters} clusters, {res.n_enriched_clusters} enriched, "
            f"silhouette {sil:.3f}, modularity {mod:.3f}"
        )
        results.append(res)
        labels_by[name] = labels
        umaps[name] = U
        assign[f"{name}_cluster"] = labels
        assign[f"{name}_umap_1"] = U[:, 0]
        assign[f"{name}_umap_2"] = U[:, 1]
        top_ids = {(t.cluster, t.go_id) for t in tops}
        keep = (df["q"] < Q_THRESHOLD).to_numpy() | np.array(
            [(c, g) in top_ids for c, g in zip(df["cluster"], df["go_id"])]
        )
        enr.append(df[keep].sort_values(["cluster", "p"]))

    pd.DataFrame({"gene": node_ids, "in_train_triple": in_train, **assign}).to_csv(
        osp.join(OUT_DIR, "cluster_assignments.csv"), index=False
    )
    pd.concat(enr, ignore_index=True)[
        [
            "representation",
            "cluster",
            "size",
            "go_id",
            "go_name",
            "namespace",
            "k",
            "K",
            "fold",
            "p",
            "q",
        ]
    ].to_csv(osp.join(OUT_DIR, "go_enrichment.csv"), index=False)

    summary = Summary(
        manifest=man,
        load_check=check,
        n_genes=len(node_ids),
        clustering={
            "input": "centered matrix -> PCA",
            "n_pcs": N_PCS,
            "graph": f"kNN k={K_NN}, Euclidean in PCA space, symmetrized union, unweighted",
            "algorithm": "Leiden (python-igraph community_leiden), modularity",
            "resolution": LEIDEN_RESOLUTION,
            "seed": SEED,
            "display": "UMAP n_neighbors=15 min_dist=0.3 random_state=0 on the PCA space",
            "silhouette_space": "PCA-50, Euclidean",
        },
        go={
            "source": "SGD GFF Ontology_term per gene (genome[g].go), is_a ancestors via goatools",
            "obo": osp.join(DATA_ROOT, "data/go/go.obo"),
            "term_size_window": [GO_MIN, GO_MAX],
            "test": "one-sided hypergeometric per cluster per term",
            "correction": "Benjamini-Hochberg over all cluster x term tests of a representation",
            "q_threshold": Q_THRESHOLD,
            **go_stats,
        },
        representations=results,
        ari_table_vs_encoder=float(
            adjusted_rand_score(labels_by["table"], labels_by["encoder"])
        ),
    )
    with open(osp.join(OUT_DIR, "summary.json"), "w") as f:
        f.write(summary.model_dump_json(indent=2))

    style()
    os.makedirs(IMG_DIR, exist_ok=True)
    by = {r.name: r for r in results}
    enc = by["encoder"]
    umap_panel(
        umaps["encoder"],
        labels_by["encoder"],
        enc,
        f"{enc.n_clusters} clusters, {enc.n_enriched_clusters} GO-enriched at q < 0.05",
        osp.join(IMG_DIR, "fig2_wt_gene_clusters"),
    )
    bar_panel(enc, osp.join(IMG_DIR, "fig2_wt_gene_clusters_bars"))
    tab = by["table"]
    umap_panel(
        umaps["table"],
        labels_by["table"],
        tab,
        f"Gene table: {tab.n_clusters} clusters, {tab.n_enriched_clusters} enriched",
        osp.join(IMG_DIR, "fig2_wt_gene_clusters_table_control"),
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="reuse the saved matrices and load check",
    )
    args = ap.parse_args()
    man = load_manifest()
    if args.analyze_only:
        with open(osp.join(OUT_DIR, "load_check.json")) as f:
            check = LoadCheck.model_validate(json.load(f))
    else:
        check = extract(man)
    summary = analyze(man, check)
    print(f"ARI table vs encoder: {summary.ari_table_vs_encoder:.3f}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
