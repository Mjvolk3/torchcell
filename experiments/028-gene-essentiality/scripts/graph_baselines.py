# experiments/028-gene-essentiality/scripts/graph_baselines.py
# [[experiments.028-gene-essentiality.scripts.graph_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/graph_baselines

r"""Graph-only baselines on the split: message passing over the gene graph, no deletion.

The Gene_Graph project (2021 to 2023) classified essential genes with two-layer graph
convolution and graph attention over YeastMine's physical, genetic and regulatory
interaction graphs, with all-ones ("featureless") node inputs, and saved no result.
This script reruns that design on the released split so it has a number: transductive
node classification on one gene graph, where a gene's logit is a function of its
neighborhood and its own features, and no genotype is perturbed. It is the baseline the
cell graph transformer's perturbed-graph reading is measured against, and the practical
side of the 1-WL question in ``wl_knockout_test.py``: a message-passing classifier is
bounded by what color refinement can separate.

Arms, each at three seeds, 200 epochs, class-weighted binary cross entropy, the epoch
selected on the validation genes by the trainer's rule (a five-epoch moving average of
the validation AUROC), scored once on the 195 held-out genes:

- GCN and GATv2 (two layers, width 64) on the union of the nine gene graphs and on the
  physical graph alone, with all-ones inputs, with degree and clustering as inputs, and
  with ProtT5 as inputs;
- an MLP on ProtT5 alone with no graph, the ablation that says what the edges add.

    python experiments/028-gene-essentiality/scripts/graph_baselines.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")
GRAPH_NAMES = (
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
)
SEEDS = (2700, 2701, 2702)
EPOCHS = 200
WIDTH = 64
LR = 1e-3


class ArmRow(BaseModel):
    """One (model, graph, input) arm at three seeds."""

    model: str
    graph: str
    inputs: str
    dim_in: int
    n_parameters: int
    test_auroc: list[float]
    val_auroc: list[float]
    select_epoch: list[int]
    mean: float
    sd: float
    seconds: float
    test_genes: list[str] = []
    test_logits: list[list[float]] = []


class Net(nn.Module):
    """Two message-passing layers and a linear head, or an MLP when the graph is absent."""

    def __init__(self, kind: str, dim_in: int) -> None:
        """``kind`` is GCN, GATv2 or MLP; ``dim_in`` the node input width."""
        super().__init__()
        self.kind = kind
        if kind == "GCN":
            self.c1: nn.Module = GCNConv(dim_in, WIDTH)
            self.c2: nn.Module = GCNConv(WIDTH, WIDTH)
        elif kind == "GATv2":
            self.c1 = GATv2Conv(dim_in, WIDTH // 4, heads=4)
            self.c2 = GATv2Conv(WIDTH, WIDTH // 4, heads=4)
        else:
            self.c1 = nn.Linear(dim_in, WIDTH)
            self.c2 = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """One logit per node."""
        if self.kind == "MLP":
            h = F.relu(self.c1(x))
            h = F.relu(self.c2(h))
        else:
            h = F.relu(self.c1(x, edge_index))
            h = F.dropout(h, 0.2, self.training)
            h = F.relu(self.c2(h, edge_index))
        return self.head(h).squeeze(-1)


def gene_graphs() -> tuple[list[str], dict[str, nx.Graph]]:
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    gene_ids = sorted(genome.gene_set)
    keep = set(gene_ids)
    out: dict[str, nx.Graph] = {}
    union = nx.Graph()
    union.add_nodes_from(gene_ids)
    for name in GRAPH_NAMES:
        g = nx.Graph()
        g.add_nodes_from(gene_ids)
        g.add_edges_from(
            (u, v)
            for u, v in getattr(graph, f"G_{name}").graph.edges()
            if u in keep and v in keep
        )
        if name == "physical":
            out["physical"] = g
        union.add_edges_from(g.edges())
    out["union of nine"] = union
    return gene_ids, out


def embedding(name: str, gene_ids: list[str]) -> np.ndarray:
    """One registry embedding per gene in cell-graph order; zeros where absent."""
    from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    built = NodeEmbeddingBuilder.build(
        embedding_names=[name], data_root=DATA_ROOT, genome=genome, graph=graph
    )[name]
    table = {
        item.id: torch.cat([t.flatten() for t in item.embeddings.values()]).numpy()
        for item in built
    }
    dim = next(iter(table.values())).shape[0]
    return np.stack([table.get(g, np.zeros(dim, dtype=np.float32)) for g in gene_ids])


def fba_ratios(gene_ids: list[str]) -> np.ndarray:
    """The four FBA growth ratios per gene from the baseline's table; 1.0 where absent."""
    import csv

    path = osp.join(RESULTS, "fba_essentiality_per_gene.csv")
    with open(path, newline="") as fh:
        table = {
            r["gene"]: [float(r[m]) for m in ("shipped", "SM", "SC", "YPD")]
            for r in csv.DictReader(fh)
        }
    return np.array([table.get(g, [1.0] * 4) for g in gene_ids], dtype=np.float32)


def structural(g: nx.Graph, gene_ids: list[str]) -> np.ndarray:
    deg = dict(g.degree())
    clust = nx.clustering(g)
    x = np.array([[np.log1p(deg[n]), clust[n]] for n in gene_ids], dtype=np.float32)
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def edges(g: nx.Graph, index: dict[str, int]) -> torch.Tensor:
    e = np.array([[index[u], index[v]] for u, v in g.edges()], dtype=np.int64).T
    e = np.concatenate([e, e[::-1]], axis=1)
    return torch.from_numpy(e)


def smoothed_argmax(values: list[float]) -> int:
    v = np.array(values)
    k = np.ones(5) / 5
    s = np.convolve(v, k, mode="same")
    return int(np.argmax(s))


def run_arm(
    kind: str,
    graph_name: str,
    inputs: str,
    x: np.ndarray,
    edge_index: torch.Tensor,
    split: dict,
    index: dict[str, int],
    device: torch.device,
    epochs: int = EPOCHS,
) -> ArmRow:
    labels = {k: int(v) for k, v in split["labels"].items()}
    parts = {
        k: torch.tensor([index[g] for g in split["genes"][k] if g in index])
        for k in ("train", "val", "test")
    }
    ys = {
        k: torch.tensor(
            [labels[g] for g in split["genes"][k] if g in index], dtype=torch.float32
        )
        for k in parts
    }
    xt = torch.from_numpy(x).to(device)
    ei = edge_index.to(device)
    pos = float(ys["train"].sum())
    pos_weight = torch.tensor((len(ys["train"]) - pos) / pos, device=device)
    tests, vals, sel, n_params = [], [], [], 0
    logits_at_sel: list[np.ndarray] = []
    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = Net(kind, x.shape[1]).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        val_curve, test_curve, test_logits = [], [], []
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            logit = net(xt, ei)
            loss = F.binary_cross_entropy_with_logits(
                logit[parts["train"]], ys["train"].to(device), pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                logit = net(xt, ei).cpu()
            val_curve.append(
                float(roc_auc_score(ys["val"].numpy(), logit[parts["val"]].numpy()))
            )
            test_curve.append(
                float(roc_auc_score(ys["test"].numpy(), logit[parts["test"]].numpy()))
            )
            test_logits.append(logit[parts["test"]].numpy().copy())
        e = smoothed_argmax(val_curve)
        logits_at_sel.append(test_logits[e])
        sel.append(e)
        vals.append(val_curve[e])
        tests.append(test_curve[e])
    row = ArmRow(
        model=kind,
        graph=graph_name,
        inputs=inputs if epochs == EPOCHS else f"{inputs} ({epochs} epochs)",
        dim_in=int(x.shape[1]),
        n_parameters=n_params,
        test_auroc=tests,
        val_auroc=vals,
        select_epoch=sel,
        mean=float(np.mean(tests)),
        sd=float(np.std(tests)),
        seconds=time.time() - t0,
    )
    row.test_genes = [g for g in split["genes"]["test"] if g in index]
    row.test_logits = [x.tolist() for x in logits_at_sel]
    print(
        f"{kind:6s} {graph_name:15s} {inputs:14s} test "
        + ", ".join(f"{t:.3f}" for t in tests)
        + f"  mean {row.mean:.3f} sd {row.sd:.3f} sel {sel} ({row.seconds:.0f}s)",
        flush=True,
    )
    return row


def fold_ensembles(device: torch.device) -> None:
    """The MLP and the 400-epoch graph convolution on the five folds, logits averaged.

    The transformer's reported 0.893 is a five-fold ensemble; this gives the two cheap
    architectures on the same three inputs the same averaging, one seed per fold as in
    ``cv_ensemble_readout.py``.
    """
    gene_ids, graphs = gene_graphs()
    index = {g: i for i, g in enumerate(gene_ids)}
    x = np.concatenate(
        [
            embedding("prot_T5_all", gene_ids),
            embedding("normalized_chrom_pathways", gene_ids),
            fba_ratios(gene_ids),
        ],
        axis=1,
    )
    ei = edges(graphs["union of nine"], index)
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    arms = {"MLP": ("MLP", no_edges, EPOCHS), "GCN": ("GCN", ei, 400)}
    report: dict[str, object] = {
        "inputs": "ProtT5+chrom-pathway+FBA ratios",
        "arms": {},
    }
    for name, (kind, edge_index, epochs) in arms.items():
        per_fold: list[dict[str, float]] = []
        fold_aurocs: list[float] = []
        labels: dict[str, int] = {}
        for k in range(5):
            with gzip.open(
                osp.join(RESULTS, "splits", f"ess_fcl_fold{k}of5_s0.json.gz"), "rt"
            ) as fh:
                split = json.load(fh)
            labels = {g: int(v) for g, v in split["labels"].items()}
            row = run_arm(
                kind,
                "union of nine" if kind == "GCN" else "none",
                f"fold {k}",
                x,
                edge_index,
                split,
                index,
                device,
                epochs=epochs,
            )
            # one seed per fold, the first
            per_fold.append(dict(zip(row.test_genes, row.test_logits[0], strict=True)))
            fold_aurocs.append(row.test_auroc[0])
        genes = sorted(per_fold[0])
        mean_logit = np.array([np.mean([f[g] for f in per_fold]) for g in genes])
        y = np.array([labels[g] for g in genes])
        auroc = float(roc_auc_score(y, mean_logit))
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                draws.append(float(roc_auc_score(y[idx], mean_logit[idx])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        report["arms"][name] = {  # type: ignore[index]
            "kind": kind,
            "epochs": epochs,
            "fold_aurocs": fold_aurocs,
            "fold_mean": float(np.mean(fold_aurocs)),
            "ensemble_auroc": auroc,
            "ensemble_ci": [float(lo), float(hi)],
            "n_genes": len(genes),
        }
        print(
            f"{name}: folds {[round(a, 3) for a in fold_aurocs]} mean "
            f"{np.mean(fold_aurocs):.3f}; ensemble {auroc:.4f} [{lo:.3f}, {hi:.3f}]",
            flush=True,
        )
    out = osp.join(RESULTS, "graph_baselines_folds.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        default="base",
        choices=("base", "extra", "folds"),
        help=(
            "base: the Gene_Graph designs (ones, structure, ProtT5) on both graphs with "
            "GCN and GATv2. extra: graph convolution on the union with the transformer's "
            "winning inputs (the chromosome-pathway indicator, the four FBA ratios) and a "
            "400-epoch budget."
        ),
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arms == "folds":
        fold_ensembles(device)
        return
    with gzip.open(SPLIT, "rt") as fh:
        split = json.load(fh)
    gene_ids, graphs = gene_graphs()
    index = {g: i for i, g in enumerate(gene_ids)}
    ones = np.ones((len(gene_ids), 1), dtype=np.float32)
    t5 = embedding("prot_T5_all", gene_ids)
    rows: list[ArmRow] = []
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    if args.arms == "extra":
        chrom = embedding("normalized_chrom_pathways", gene_ids)
        fba = fba_ratios(gene_ids)
        feats_extra = {
            "ProtT5+chrom-pathway": np.concatenate([t5, chrom], axis=1),
            "ProtT5+FBA ratios": np.concatenate([t5, fba], axis=1),
            "ProtT5+chrom-pathway+FBA ratios": np.concatenate([t5, chrom, fba], axis=1),
        }
        ei = edges(graphs["union of nine"], index)
        for inputs, x in feats_extra.items():
            rows.append(
                run_arm("MLP", "none", inputs, x, no_edges, split, index, device)
            )
            rows.append(
                run_arm("GCN", "union of nine", inputs, x, ei, split, index, device)
            )
        rows.append(
            run_arm(
                "GCN",
                "union of nine",
                "ProtT5",
                t5,
                ei,
                split,
                index,
                device,
                epochs=400,
            )
        )
        rows.append(
            run_arm(
                "GCN",
                "union of nine",
                "ProtT5+chrom-pathway+FBA ratios",
                feats_extra["ProtT5+chrom-pathway+FBA ratios"],
                ei,
                split,
                index,
                device,
                epochs=400,
            )
        )
        out = osp.join(RESULTS, "graph_baselines_extra.json")
    else:
        rows.append(
            run_arm("MLP", "none", "ProtT5", t5, no_edges, split, index, device)
        )
        for graph_name, g in graphs.items():
            ei = edges(g, index)
            feats = {
                "ones": ones,
                "degree+clustering": structural(g, gene_ids),
                "ProtT5": t5,
            }
            for kind in ("GCN", "GATv2"):
                for inputs, x in feats.items():
                    rows.append(
                        run_arm(kind, graph_name, inputs, x, ei, split, index, device)
                    )
        out = osp.join(RESULTS, "graph_baselines.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "protocol": (
                    "transductive node classification on one gene graph, no perturbation; "
                    "class-weighted BCE, Adam 1e-3, 200 epochs, width 64, dropout 0.2; "
                    "epoch selected by a five-epoch moving average of the validation "
                    "AUROC; scored once on the 195 held-out genes"
                ),
                "rows": [
                    r.model_dump(exclude={"test_genes", "test_logits"}) for r in rows
                ],
            },
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
