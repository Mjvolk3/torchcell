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
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, RGCNConv, SAGEConv

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


#: The message-passing kinds, each two layers of width 64 and a linear head:
#: GCN, the symmetric-normalized mean; GATv2, four attention heads; GIN, the
#: injective sum with a two-layer MLP (the 1-WL-matching aggregator of Xu et al. 2019);
#: RGCN and RGAT, one weight (or one attention) per relation over the nine graphs as
#: typed edges, with basis decomposition to keep the parameter count near GCN's.
KINDS = ("MLP", "GCN", "GATv2", "GIN", "GraphSAGE", "RGCN", "RGAT")
N_RELATIONS = len(GRAPH_NAMES)


class Net(nn.Module):
    """Two message-passing layers and a linear head, or an MLP when the graph is absent."""

    def __init__(self, kind: str, dim_in: int) -> None:
        """``kind`` is one of ``KINDS``; ``dim_in`` the node input width."""
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        if kind == "GCN":
            self.c1: nn.Module = GCNConv(dim_in, WIDTH)
            self.c2: nn.Module = GCNConv(WIDTH, WIDTH)
        elif kind == "GATv2":
            self.c1 = GATv2Conv(dim_in, WIDTH // 4, heads=4)
            self.c2 = GATv2Conv(WIDTH, WIDTH // 4, heads=4)
        elif kind == "GIN":
            self.c1 = GINConv(
                nn.Sequential(
                    nn.Linear(dim_in, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH)
                ),
                train_eps=True,
            )
            self.c2 = GINConv(
                nn.Sequential(
                    nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH)
                ),
                train_eps=True,
            )
        elif kind == "GraphSAGE":
            self.c1 = SAGEConv(dim_in, WIDTH)
            self.c2 = SAGEConv(WIDTH, WIDTH)
        elif kind == "RGCN":
            self.c1 = RGCNConv(dim_in, WIDTH, num_relations=N_RELATIONS, num_bases=4)
            self.c2 = RGCNConv(WIDTH, WIDTH, num_relations=N_RELATIONS, num_bases=4)
        elif kind == "RGAT":
            # One GATv2 attention per relation, summed: alpha_ij^(r) and Theta_r per edge
            # type, no basis sharing. PyG's RGATConv materializes a per-edge weight
            # matrix (4.87M typed edges x in x heads*out: 20 GB at the raw input, 80 GB
            # at width 64 behind a projection; jobs 2340 and 2345 OOM on a 44 GiB card),
            # so the per-relation form is built from the same attention layer the GATv2
            # arm uses, each on its own edge subset.
            self.c1 = nn.ModuleList(
                [GATv2Conv(dim_in, WIDTH // 4, heads=4) for _ in range(N_RELATIONS)]
            )
            self.c2 = nn.ModuleList(
                [GATv2Conv(WIDTH, WIDTH // 4, heads=4) for _ in range(N_RELATIONS)]
            )
        else:
            self.c1 = nn.Linear(dim_in, WIDTH)
            self.c2 = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One logit per node; ``edge_type`` is read by the relational kinds only."""
        if self.kind == "MLP":
            h = F.relu(self.c1(x))
            h = F.relu(self.c2(h))
        elif self.kind == "RGAT":
            assert edge_type is not None, "relational kinds need edge_type"
            subsets = [edge_index[:, edge_type == r] for r in range(N_RELATIONS)]
            h = F.relu(sum(c(x, e) for c, e in zip(self.c1, subsets)))
            h = F.dropout(h, 0.2, self.training)
            h = F.relu(sum(c(h, e) for c, e in zip(self.c2, subsets)))
        elif self.kind == "RGCN":
            assert edge_type is not None, "relational kinds need edge_type"
            h = F.relu(self.c1(x, edge_index, edge_type))
            h = F.dropout(h, 0.2, self.training)
            h = F.relu(self.c2(h, edge_index, edge_type))
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
        out[f"one:{name}"] = g
        if name == "physical":
            out["physical"] = g
        union.add_edges_from(g.edges())
    out["union of nine"] = union
    return gene_ids, out


def typed_edges(
    graphs: dict[str, nx.Graph], index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The nine graphs as one edge set with a relation type per graph, both directions."""
    parts, types = [], []
    for r, name in enumerate(GRAPH_NAMES):
        e = edges(graphs[f"one:{name}"], index)
        parts.append(e)
        types.append(torch.full((e.shape[1],), r, dtype=torch.long))
    return torch.cat(parts, dim=1), torch.cat(types)


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
        item.id: torch.cat([t.flatten() for t in item.embeddings.values()])
        .cpu()
        .numpy()
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


def solve_log1p(gene_ids: list[str], store: str | None = None) -> np.ndarray:
    """The deletion's solved flux vector as ``log(1 + |v|)`` per gene, 0 where absent.

    The round-33 encoding over the store ``ess_reference_flux_table.py`` writes (the
    pFBA vector by default; ``ESS_SOLVE_TABLE`` or ``store`` names another, such as
    the interior-point store). The store covers the 1,121 labeled genes; every other
    gene reads the zero vector, which under this encoding is the flux of a gene with
    no reaction, not a column mean.
    """
    path = store or os.environ.get(
        "ESS_SOLVE_TABLE",
        osp.join(DATA_ROOT, "data/torchcell/reference_flux/ess_solve_encodings_YPD.npz"),
    )
    payload = np.load(path, allow_pickle=False)
    rows = {str(g): i for i, g in enumerate(payload["genes"])}
    v = np.log1p(np.abs(payload["v_deletion"].astype(np.float32)))
    out = np.zeros((len(gene_ids), v.shape[1]), dtype=np.float32)
    for i, g in enumerate(gene_ids):
        if g in rows:
            out[i] = v[rows[g]]
    return out


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
    edge_type: torch.Tensor | None = None,
    seeds: tuple[int, ...] = SEEDS,
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
    et = edge_type.to(device) if edge_type is not None else None
    pos = float(ys["train"].sum())
    pos_weight = torch.tensor((len(ys["train"]) - pos) / pos, device=device)
    tests, vals, sel, n_params = [], [], [], 0
    logits_at_sel: list[np.ndarray] = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        net = Net(kind, x.shape[1]).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        val_curve, test_curve, test_logits = [], [], []
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            logit = net(xt, ei, et)
            loss = F.binary_cross_entropy_with_logits(
                logit[parts["train"]], ys["train"].to(device), pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                logit = net(xt, ei, et).cpu()
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


def fold_ensembles(
    device: torch.device,
    which: str = "three",
    save_logits: bool = False,
    seeds: tuple[int, ...] = (SEEDS[0],),
) -> None:
    """Cheap architectures on the five folds, logits averaged on the held-out genes.

    The transformer's reported 0.893 is a five-fold ensemble; this gives the cheap
    architectures the same averaging, one seed per fold as in
    ``cv_ensemble_readout.py``. ``which`` selects the input and the arms: ``three``
    is the MLP and the 400-epoch convolution on the transformer's three inputs;
    ``go`` is convolution on the propagated GO multi-hot alone at 400 and 800 epochs.
    With more than one seed, each fold's logit is the mean over its seeds before the
    folds are averaged, and the report goes to its own file, suffixed ``_s<n>``.
    """
    gene_ids, graphs = gene_graphs()
    index = {g: i for i, g in enumerate(gene_ids)}
    ei = edges(graphs["union of nine"], index)
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    if which == "three":
        x = np.concatenate(
            [
                embedding("prot_T5_all", gene_ids),
                embedding("normalized_chrom_pathways", gene_ids),
                fba_ratios(gene_ids),
            ],
            axis=1,
        )
        arms = {"MLP": ("MLP", no_edges, EPOCHS), "GCN": ("GCN", ei, 400)}
        inputs_label = "ProtT5+chrom-pathway+FBA ratios"
        out_name = "graph_baselines_folds.json"
    else:
        assert which == "go", which
        x = embedding("go", gene_ids)
        arms = {"GCN400": ("GCN", ei, 400), "GCN800": ("GCN", ei, 800)}
        inputs_label = "GO"
        out_name = "graph_baselines_go_folds.json"
    if len(seeds) > 1:
        out_name = out_name.replace(".json", f"_s{len(seeds)}.json")
    report: dict[str, object] = {
        "inputs": inputs_label,
        "seeds_per_fold": list(seeds),
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
                seeds=seeds,
            )
            # The fold's logit is the mean over its seeds (one seed by default).
            fold_logit = np.mean(np.array(row.test_logits, dtype=float), axis=0)
            per_fold.append(dict(zip(row.test_genes, fold_logit, strict=True)))
            fold_y = np.array([labels[g] for g in row.test_genes])
            fold_aurocs.append(float(roc_auc_score(fold_y, fold_logit)))
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
        if save_logits:
            # The per-fold held-out logits, for family_ensembles.py --folds.
            report["arms"][name]["genes"] = genes  # type: ignore[index]
            report["arms"][name]["fold_logits"] = [  # type: ignore[index]
                [float(f[g]) for g in genes] for f in per_fold
            ]
        print(
            f"{name}: folds {[round(a, 3) for a in fold_aurocs]} mean "
            f"{np.mean(fold_aurocs):.3f}; ensemble {auroc:.4f} [{lo:.3f}, {hi:.3f}]",
            flush=True,
        )
    out = osp.join(RESULTS, out_name)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        default="base",
        choices=(
            "base",
            "extra",
            "folds",
            "relational",
            "split",
            "go",
            "go400",
            "go800",
            "go_folds",
            "go_rel",
            "go_solve",
        ),
        help=(
            "base: the Gene_Graph designs (ones, structure, ProtT5) on both graphs with "
            "GCN and GATv2. extra: graph convolution on the union with the transformer's "
            "winning inputs (the chromosome-pathway indicator, the four FBA ratios) and a "
            "400-epoch budget."
        ),
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help=(
            "Keep each row's per-gene held-out logits (one list per seed) in the "
            "result file, for hard_genes.py; off by default to keep the files small."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds per arm (default 2700 2701 2702). A non-default list writes to its "
            "own result file, suffixed _s<n>. Under --arms folds/go_folds these are the "
            "seeds per fold, whose logits are averaged before the fold ensemble."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "A split file under results/splits other than the released one "
            "(ess_fcl_val0.2_s0), such as ess_genome_heldout_val0.2_s0.json.gz; the "
            "result file is then suffixed with the file's label source."
        ),
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds: tuple[int, ...] = tuple(args.seeds) if args.seeds else SEEDS
    split_path = SPLIT if args.split is None else osp.join(RESULTS, "splits", args.split)
    split_tag = ""
    if args.split is not None:
        split_tag = "_" + args.split.split("_val")[0].removeprefix("ess_")
    if args.arms in ("folds", "go_folds"):
        fold_ensembles(
            device,
            "three" if args.arms == "folds" else "go",
            args.save_logits,
            seeds=(SEEDS[0],) if args.seeds is None else seeds,
        )
        return
    with gzip.open(split_path, "rt") as fh:
        split = json.load(fh)
    print(
        f"split {osp.basename(split_path)}: "
        + ", ".join(f"{k} {len(v)}" for k, v in split["genes"].items()),
        flush=True,
    )
    gene_ids, graphs = gene_graphs()
    index = {g: i for i, g in enumerate(gene_ids)}
    ones = np.ones((len(gene_ids), 1), dtype=np.float32)
    t5 = embedding("prot_T5_all", gene_ids)
    rows: list[ArmRow] = []
    no_edges = torch.zeros(2, 0, dtype=torch.long)
    if args.arms == "relational":
        # THE MATH-490 AND GENE_GRAPH ARCHITECTURES ON THIS SPLIT: GIN (the 1-WL
        # aggregator), GraphSAGE, and the two relational kinds over the nine graphs as
        # typed edges (RGCN with basis decomposition, RGAT), each with ProtT5 alone and
        # with the transformer's three inputs, 200 epochs, beside GCN on the same edges.
        chrom = embedding("normalized_chrom_pathways", gene_ids)
        fba = fba_ratios(gene_ids)
        feats_rel = {
            "ProtT5": t5,
            "ProtT5+chrom-pathway+FBA ratios": np.concatenate([t5, chrom, fba], axis=1),
        }
        ei = edges(graphs["union of nine"], index)
        ei_typed, et_typed = typed_edges(graphs, index)
        for inputs, x in feats_rel.items():
            for kind in ("GIN", "GraphSAGE"):
                rows.append(
                    run_arm(kind, "union of nine", inputs, x, ei, split, index, device)
                )
            for kind in ("RGCN", "RGAT"):
                rows.append(
                    run_arm(
                        kind,
                        "nine typed",
                        inputs,
                        x,
                        ei_typed,
                        split,
                        index,
                        device,
                        edge_type=et_typed,
                    )
                )
        out = osp.join(RESULTS, "graph_baselines_relational.json")
    elif args.arms == "split":
        # THE INDICATOR SPLIT INTO ITS BLOCKS: the pathway multi-hot, the chromosome
        # one-hot and the seven scaled properties, each beside ProtT5 and the four FBA
        # ratios, and the pathway block alone with the ratios, on the MLP and on GCN.
        blocks = {
            "pathways": embedding("pathways", gene_ids),
            "chromosome": embedding("chromosome", gene_ids),
            "properties": embedding("normalized_props", gene_ids),
        }
        fba = fba_ratios(gene_ids)
        feats_split = {
            f"ProtT5+{name}+FBA ratios": np.concatenate([t5, block, fba], axis=1)
            for name, block in blocks.items()
        }
        feats_split["pathways+FBA ratios"] = np.concatenate(
            [blocks["pathways"], fba], axis=1
        )
        feats_split["pathways"] = blocks["pathways"]
        ei = edges(graphs["union of nine"], index)
        for inputs, x in feats_split.items():
            rows.append(
                run_arm("MLP", "none", inputs, x, no_edges, split, index, device)
            )
            rows.append(
                run_arm("GCN", "union of nine", inputs, x, ei, split, index, device)
            )
        out = osp.join(RESULTS, "graph_baselines_split.json")
    elif args.arms == "go":
        # THE GO MULTI-HOT (round 26): propagated over terms with ten or more genes,
        # the biological-process namespace alone, and the direct annotations, each
        # beside ProtT5 and the ratios; GO alone with the ratios and alone; and GO with
        # the pathway multi-hot, on the MLP and on GCN.
        go = embedding("go", gene_ids)
        go_bp = embedding("go_bp", gene_ids)
        go_direct = embedding("go_direct", gene_ids)
        pathways = embedding("pathways", gene_ids)
        fba = fba_ratios(gene_ids)
        # The pathway multi-hot is nonzero for 517 of 6,607 genes, so one column that
        # says "annotated to any pathway" is the control for whether the 173 columns
        # carry which pathway or only that there is one.
        pathway_any = (pathways.sum(axis=1, keepdims=True) > 0).astype(np.float32)
        feats_go = {
            "ProtT5+pathway any+FBA ratios": np.concatenate(
                [t5, pathway_any, fba], axis=1
            ),
            "ProtT5+GO+FBA ratios": np.concatenate([t5, go, fba], axis=1),
            "ProtT5+GO BP+FBA ratios": np.concatenate([t5, go_bp, fba], axis=1),
            "ProtT5+GO direct+FBA ratios": np.concatenate([t5, go_direct, fba], axis=1),
            "GO+FBA ratios": np.concatenate([go, fba], axis=1),
            "GO": go,
            "ProtT5+GO+pathways+FBA ratios": np.concatenate(
                [t5, go, pathways, fba], axis=1
            ),
        }
        ei = edges(graphs["union of nine"], index)
        for inputs, x in feats_go.items():
            rows.append(
                run_arm("MLP", "none", inputs, x, no_edges, split, index, device)
            )
            rows.append(
                run_arm("GCN", "union of nine", inputs, x, ei, split, index, device)
            )
        out = osp.join(RESULTS, "graph_baselines_go.json")
    elif args.arms in ("go400", "go800"):
        # Graph convolution on GO selects epochs 183 to 197 at a 200-epoch budget, so
        # the three GO rows that matter run to 400 (the round-8 table's 400-epoch rows
        # are the precedent), and, still selecting 382 to 397 there, to 800.
        n_epochs = 400 if args.arms == "go400" else 800
        go = embedding("go", gene_ids)
        fba = fba_ratios(gene_ids)
        ei = edges(graphs["union of nine"], index)
        # run_arm appends "(N epochs)" itself when the budget is not the default.
        feats_long = {
            "GO": go,
            "GO+FBA ratios": np.concatenate([go, fba], axis=1),
            "ProtT5+GO+FBA ratios": np.concatenate([t5, go, fba], axis=1),
        }
        for inputs, x in feats_long.items():
            rows.append(
                run_arm(
                    "GCN",
                    "union of nine",
                    inputs,
                    x,
                    ei,
                    split,
                    index,
                    device,
                    epochs=n_epochs,
                    seeds=seeds,
                )
            )
        # A non-default seed list writes its own file, so the three-seed read stays.
        suffix = ("" if seeds == SEEDS else f"_s{len(seeds)}") + split_tag
        out = osp.join(RESULTS, f"graph_baselines_{args.arms}{suffix}.json")
    elif args.arms == "go_solve":
        # ONE MODEL THAT HOLDS BOTH (the open item of the family averages): the
        # annotation through the graph and the deletion's solved flux, as the
        # round-33 cell reads it, on the same node. Graph convolution over the union
        # on GO with the log1p vector, with the ratios beside them, and the solve
        # alone; 400 epochs as the GO rows; and the MLP on the same inputs as the
        # no-graph control.
        go = embedding("go", gene_ids)
        fba = fba_ratios(gene_ids)
        solve = solve_log1p(gene_ids)
        ei = edges(graphs["union of nine"], index)
        feats_solve = {
            "solve log1p": solve,
            "GO+solve log1p": np.concatenate([go, solve], axis=1),
            "GO+FBA ratios+solve log1p": np.concatenate([go, fba, solve], axis=1),
        }
        for inputs, x in feats_solve.items():
            rows.append(
                run_arm(
                    "MLP", "none", inputs, x, no_edges, split, index, device, seeds=seeds
                )
            )
            rows.append(
                run_arm(
                    "GCN",
                    "union of nine",
                    inputs,
                    x,
                    ei,
                    split,
                    index,
                    device,
                    epochs=400,
                    seeds=seeds,
                )
            )
        suffix = ("" if seeds == SEEDS else f"_s{len(seeds)}") + split_tag
        out = osp.join(RESULTS, f"graph_baselines_go_solve{suffix}.json")
    elif args.arms == "go_rel":
        # The typed-edge models on GO alone at 400 epochs (RGAT beat convolution on the
        # three inputs), and convolution on GO with the pathway multi-hot.
        go = embedding("go", gene_ids)
        pathways = embedding("pathways", gene_ids)
        ei = edges(graphs["union of nine"], index)
        ei_typed, et_typed = typed_edges(graphs, index)
        for kind in ("RGCN", "RGAT"):
            rows.append(
                run_arm(
                    kind,
                    "nine typed",
                    "GO",
                    go,
                    ei_typed,
                    split,
                    index,
                    device,
                    edge_type=et_typed,
                    epochs=400,
                )
            )
        rows.append(
            run_arm(
                "GCN",
                "union of nine",
                "GO+pathways",
                np.concatenate([go, pathways], axis=1),
                ei,
                split,
                index,
                device,
                epochs=400,
            )
        )
        rows.append(
            run_arm(
                "GraphSAGE",
                "union of nine",
                "GO",
                go,
                ei,
                split,
                index,
                device,
                epochs=400,
            )
        )
        out = osp.join(RESULTS, "graph_baselines_go_rel.json")
    elif args.arms == "extra":
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
        out = osp.join(RESULTS, f"graph_baselines_extra{split_tag}.json")
    else:
        rows.append(
            run_arm("MLP", "none", "ProtT5", t5, no_edges, split, index, device)
        )
        for graph_name, g in graphs.items():
            if graph_name.startswith("one:"):
                continue  # the single graphs serve the typed edge set only
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
        out = osp.join(RESULTS, f"graph_baselines{split_tag}.json")
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
                    r.model_dump(
                        exclude=set()
                        if args.save_logits
                        else {"test_genes", "test_logits"}
                    )
                    for r in rows
                ],
            },
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
