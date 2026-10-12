# experiments/031-betaxanthin-module/scripts/target_gcn_readout.py
# [[experiments.031-betaxanthin-module.scripts.target_gcn_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_gcn_readout

r"""Message passing read out at the deleted gene: ego nets on the gene graphs and on the GO incidence graph.

Two inductive readings of a deletion strain's target from graphs, with no label and no
measurement of the strain entering its own row:

- **Ego net on a gene graph.** Two graph-convolution layers over a gene graph (the
  union of the nine the transformer masks, or STRING coexpression alone) with the
  gene's own GO multi-hot as its input, read out at the deleted gene: its prediction is
  a function of its two-hop neighborhood's annotations, the learned form of the one-hop
  neighbor mean that ``target_graph_feature_store.py`` writes (forest 0.279 on the pin).
- **GO as a graph, not a vector.** The annotation is an incidence graph: gene nodes,
  term nodes, an edge for each annotation, and on ``go_incidence_dag`` the ontology's
  own ``is_a`` and ``part_of`` edges between terms. Two hops from a gene reach the genes
  that share its terms (and on the DAG variant, the terms one level up), so the read is
  a co-annotation neighborhood rather than a fixed propagated vector. Gene nodes carry
  the multi-hot, term nodes their identity, in one input of width 2 x 2,362.

Every arm regresses the standardized target with two message-passing layers of width
64 (``graph_baselines.py``'s design), 200 full-batch epochs, the epoch selected on the
validation genes by the trainer's rule (five-epoch moving average of the validation
Pearson), scored once on the pinned genes as Spearman, three seeds. Training genes are
the trainer's train split, validation its validation split, and every pinned gene's
target is masked from the loss: the pinned gene is a node whose features and edges are
seen, as any inductive test gene's would be, and whose target is not. A gene absent
from the graph carries no message and reads its own features through the same layers.
Writes ``results/target_gcn_readout_<target>.json``. Run from the repo root under
slurm (``gh_target_gcn_readout.slurm``)::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_gcn_readout.py --target betaxanthin
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from goatools.obo_parser import GODag
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch_geometric.nn import GCNConv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
import train_bx  # noqa: E402
from linear_baselines_bx import TARGETS, open_dataset, read_targets, record_genes  # noqa: E402
from target_graph_homophily import GRAPHS  # noqa: E402
from target_store_baselines import CSV_TARGETS, STUDY  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
GO_STORE = {"betaxanthin": "YPD_bx_go", "amino_acids": "SM_gem_go"}
WIDTH = 64
EPOCHS = 200
LR = 1e-3
SEEDS = (0, 1, 2)
SMOOTH = 5
ARMS = [
    # (name, graph, inputs)
    ("mlp_go", "none", "go"),
    ("gcn_union_go", "union", "go"),
    ("gcn_coexpression_go", "string12_0_coexpression", "go"),
    ("gcn_go_incidence", "go_incidence", "go+term"),
    ("gcn_go_incidence_dag", "go_incidence_dag", "go+term"),
    ("gcn_go_incidence_ones", "go_incidence", "ones"),
    ("gcn_go_incidence_dag_ones", "go_incidence_dag", "ones"),
]


class ArmRead(BaseModel):
    """One arm: pinned Spearman per seed at the validation-selected epoch."""

    name: str
    graph: str
    inputs: str
    n_nodes: int
    n_edges: int
    dim_in: int
    n_parameters: int
    select_epoch: list[int]
    val_pearson: list[float]
    pinned_spearman: list[float]
    pinned_pearson: list[float]
    mean: float
    sd: float
    seconds: float


class Net(nn.Module):
    def __init__(self, graph: bool, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.graph = graph
        if graph:
            self.c1: nn.Module = GCNConv(dim_in, WIDTH)
            self.c2: nn.Module = GCNConv(WIDTH, WIDTH)
        else:
            self.c1 = nn.Linear(dim_in, WIDTH)
            self.c2 = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, dim_out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.graph:
            h = F.relu(self.c1(x, edge_index))
            h = F.dropout(h, 0.2, self.training)
            h = F.relu(self.c2(h, edge_index))
        else:
            h = F.relu(self.c1(x))
            h = F.dropout(h, 0.2, self.training)
            h = F.relu(self.c2(h))
        return self.head(h)


def smoothed_argmax(values: list[float]) -> int:
    v = np.array(values)
    k = min(SMOOTH, len(v))
    sm = np.convolve(v, np.ones(k) / k, mode="valid")
    return int(np.argmax(sm)) + k - 1


def mean_over_columns(fn, pred: np.ndarray, true: np.ndarray) -> float:
    out = []
    for j in range(true.shape[1]):
        fin = np.isfinite(true[:, j])
        if fin.sum() < 3:
            continue
        out.append(float(fn(pred[fin, j], true[fin, j])[0]))
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(GO_STORE))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-suffix", default="", help="appended to the results file name")
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    torch.set_num_threads(args.threads)
    started = time.time()
    spec = CSV_TARGETS.get(args.target) or TARGETS[args.target]
    columns = list(spec["columns"])
    assert "csv" not in spec

    # ---- targets per gene from the trainer's own split ----------------------------
    cfg = train_bx.load_config()
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
    dataset = open_dataset(spec, genome, graph)
    pinned, _ = train_bx.resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, spec["dataset_subdir"], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cfg.model.batch_size,
        random_seed=cfg.data.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    base = osp.join(DATA_ROOT, spec["dataset_subdir"])
    idx_to_genes = record_genes(dataset)
    with open(osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])
    y_by_split: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        indices = [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        values = read_targets(base, indices, spec["label"], columns)
        rows: dict[str, np.ndarray] = {}
        for i, v in zip(indices, values, strict=True):
            gs = idx_to_genes.get(i, [])
            if len(gs) == 1 and np.isfinite(v).any():
                rows[gs[0]] = v
        if name == "test":
            rows = {g: v for g, v in rows.items() if g in pinned_genes}
        y_by_split[name] = rows
    print(
        "  genes with a target: "
        + ", ".join(f"{k} {len(v)}" for k, v in y_by_split.items()),
        flush=True,
    )
    assert not (set(y_by_split["train"]) | set(y_by_split["val"])) & pinned_genes

    # ---- nodes and inputs ----------------------------------------------------------
    genes = sorted(genome.gene_set)
    gene_index = {g: i for i, g in enumerate(genes)}
    z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{GO_STORE[args.target]}.npz"))
    store_genes = [str(g) for g in z["genes"]]
    terms = [str(t) for t in z["rxn_ids"]]
    term_index = {t: i for i, t in enumerate(terms)}
    go = np.zeros((len(genes), len(terms)), dtype=np.float32)
    for g, row in zip(store_genes, z["v_deletion"], strict=True):
        if g in gene_index:
            go[gene_index[g]] = row
    n_gene, n_term = len(genes), len(terms)
    print(f"  {n_gene} gene nodes, {n_term} term nodes, GO multi-hot {go.shape}", flush=True)

    # ---- edges ---------------------------------------------------------------------
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    undirected = {}
    for name in GRAPHS:
        g = multigraph[name].graph
        undirected[name] = g.to_undirected() if g.is_directed() else g
    undirected["union"] = nx.compose_all([undirected[n] for n in GRAPHS])

    def gene_edges(name: str) -> torch.Tensor:
        pairs = [
            (gene_index[u], gene_index[v])
            for u, v in undirected[name].edges
            if u != v and u in gene_index and v in gene_index
        ]
        src = [u for u, _ in pairs] + [v for _, v in pairs]
        dst = [v for _, v in pairs] + [u for u, _ in pairs]
        return torch.tensor([src, dst], dtype=torch.long)

    # Incidence: gene node i <-> term node n_gene + j for every 1 of the propagated
    # multi-hot; the DAG variant uses the DIRECT annotations from G_go instead and adds
    # the ontology's is_a and part_of edges between the term nodes, so propagation is
    # what message passing does rather than what the store did.
    def incidence_edges(direct: bool) -> torch.Tensor:
        if direct:
            pairs = set()
            for node, d in graph.G_go.nodes(data=True):
                if node in term_index and d.get("genes"):
                    for gname in d["genes"]:
                        if gname in gene_index:
                            pairs.add((gene_index[gname], n_gene + term_index[node]))
            dag = GODag(osp.join(genome.go_root, "go.obo"), optional_attrs={"relationship"}, prt=None)
            for t in terms:
                if t not in dag:
                    continue
                rec = dag[t]
                parents = {p.item_id for p in rec.parents} | {
                    p.item_id for p in rec.relationship.get("part_of", set())
                }
                for p in parents:
                    if p in term_index:
                        pairs.add((n_gene + term_index[t], n_gene + term_index[p]))
            pairs_list = sorted(pairs)
        else:
            gi, tj = np.nonzero(go)
            pairs_list = [(int(a), n_gene + int(b)) for a, b in zip(gi, tj, strict=True)]
        src = [u for u, _ in pairs_list] + [v for _, v in pairs_list]
        dst = [v for _, v in pairs_list] + [u for u, _ in pairs_list]
        return torch.tensor([src, dst], dtype=torch.long)

    # ---- targets as tensors ----------------------------------------------------------
    c = len(columns)

    def target_tensor(rows: dict[str, np.ndarray]) -> tuple[torch.Tensor, np.ndarray]:
        idx = torch.tensor([gene_index[g] for g in rows if g in gene_index])
        y = np.stack([rows[g] for g in rows if g in gene_index]).astype(np.float32)
        return idx, y

    idx_tr, y_tr = target_tensor(y_by_split["train"])
    idx_va, y_va = target_tensor(y_by_split["val"])
    idx_te, y_te = target_tensor(y_by_split["test"])
    mu = np.nanmean(y_tr, axis=0)
    sd = np.nanstd(y_tr, axis=0)
    sd[sd == 0] = 1.0
    z_tr = torch.from_numpy(np.nan_to_num((y_tr - mu) / sd, nan=0.0))
    m_tr = torch.from_numpy(np.isfinite(y_tr))
    device = torch.device("cpu")

    reads: list[ArmRead] = []
    for name, gname, inputs in ARMS:
        t0 = time.time()
        if gname == "none":
            x = torch.from_numpy(go)
            ei = torch.zeros((2, 0), dtype=torch.long)
            n_nodes = n_gene
        elif gname.startswith("go_incidence"):
            n_nodes = n_gene + n_term
            if inputs == "ones":
                x = torch.ones((n_nodes, 1))
            else:
                x = torch.zeros((n_nodes, 2 * n_term))
                x[:n_gene, :n_term] = torch.from_numpy(go)
                x[n_gene:, n_term:] = torch.eye(n_term)
            ei = incidence_edges(direct=gname.endswith("_dag"))
        else:
            x = torch.from_numpy(go)
            ei = gene_edges(gname)
            n_nodes = n_gene
        sel, vals, tests, pears, n_params = [], [], [], [], 0
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = Net(gname != "none", x.shape[1], c).to(device)
            n_params = sum(p.numel() for p in net.parameters())
            opt = torch.optim.Adam(net.parameters(), lr=LR)
            val_curve: list[float] = []
            test_preds: list[np.ndarray] = []
            for _ in range(args.epochs):
                net.train()
                opt.zero_grad()
                out = net(x, ei)
                p = out[idx_tr]
                loss = ((p - z_tr) ** 2 * m_tr).sum() / m_tr.sum()
                loss.backward()
                opt.step()
                net.eval()
                with torch.no_grad():
                    out = net(x, ei).numpy()
                val_curve.append(mean_over_columns(pearsonr, out[idx_va.numpy()], y_va))
                test_preds.append(out[idx_te.numpy()].copy())
            e = smoothed_argmax(val_curve)
            sel.append(e)
            vals.append(val_curve[e])
            tests.append(mean_over_columns(spearmanr, test_preds[e], y_te))
            pears.append(mean_over_columns(pearsonr, test_preds[e], y_te))
        read = ArmRead(
            name=name,
            graph=gname,
            inputs=inputs,
            n_nodes=n_nodes,
            n_edges=int(ei.shape[1] // 2),
            dim_in=int(x.shape[1]),
            n_parameters=n_params,
            select_epoch=sel,
            val_pearson=vals,
            pinned_spearman=tests,
            pinned_pearson=pears,
            mean=float(np.mean(tests)),
            sd=float(np.std(tests, ddof=1)),
            seconds=time.time() - t0,
        )
        reads.append(read)
        print(
            f"  {name:26s} nodes {n_nodes:5d} edges {read.n_edges:8d} | pinned Spearman "
            + ", ".join(f"{t:+.3f}" for t in tests)
            + f" mean {read.mean:+.3f} sd {read.sd:.3f} sel {sel} ({read.seconds:.0f}s)",
            flush=True,
        )
    out_path = osp.join(EXPERIMENT_ROOT, STUDY[args.target], "results", f"target_gcn_readout_{args.target}{args.out_suffix}.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "generated_by": "experiments/031-betaxanthin-module/scripts/target_gcn_readout.py",
                "target": args.target,
                "columns": columns,
                "n_train": int(len(idx_tr)),
                "n_val": int(len(idx_va)),
                "n_pinned": int(len(idx_te)),
                "epochs": args.epochs,
                "seeds": list(SEEDS),
                "statistic": (
                    "pinned-test Spearman (mean over columns) of the node prediction at the "
                    "deleted gene, epoch selected by the five-epoch moving average of the "
                    "validation Pearson; pinned targets never enter the loss"
                ),
                "arms": [r.model_dump() for r in reads],
                "seconds": time.time() - started,
            },
            fh,
            indent=2,
        )
    print(out_path)


if __name__ == "__main__":
    main()
