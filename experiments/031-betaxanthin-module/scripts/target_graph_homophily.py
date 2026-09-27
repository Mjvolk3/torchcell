# experiments/031-betaxanthin-module/scripts/target_graph_homophily.py
# [[experiments.031-betaxanthin-module.scripts.target_graph_homophily]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_graph_homophily

r"""Do the nine gene graphs carry the target at all: label propagation on the pin.

Before asking whether the transformer can read the graphs (rounds 213 to 216, the KL
regularizer and the hard mask on the second block), the question is whether the target is
smooth on them. For each of the nine gene graphs the trainer masks and regularizes, and
their union, every pinned test gene is predicted by the mean target of its labeled
neighbors, the training and validation genes with a measurement: what one round of
message passing over the labels carries, with nothing fitted. The read is the pinned-test
Spearman of that predictor on the genes with at least one labeled neighbor beside the
coverage, the same read within two hops, the trainer's top-50 enrichment on the filled
predictor, and a degree-matched null in which each pinned gene's labeled neighbor set is
replaced by a random labeled set of the same size (the neighbors' identity destroyed,
their count kept). The same predictor scored leave-one-out on the labeled genes is the
larger-sample assortativity. A graph on which the neighbor mean reads at its null has
nothing for a mask or a prior to hand the model on this target; one that reads above it
sets the bar the graph arms have to clear.

Targets: betaxanthin (031), the nineteen M\"ulleder amino acids and the seventeen Cooper
peaks (032), each on the trainer's pinned split and its single-deletion genotypes, loaded
as ``target_store_baselines.py`` loads them. Writes
``results/target_graph_homophily_<target>.json`` in the target's study. Run from the
repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_graph_homophily.py \
        --target betaxanthin
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import fisher_exact, spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
import train_bx  # noqa: E402
from linear_baselines_bx import (  # noqa: E402
    TARGETS,
    open_dataset,
    read_targets,
    record_genes,
)
from target_store_baselines import CSV_TARGETS, STUDY, top50_enrichment  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

#: The nine graphs the trainer's hard mask burns into nine heads and the KL regularizer
#: targets (``STRONG_GRAPHS`` in train_essentiality.py), in that order.
GRAPHS = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
TOP_DECILE = 0.9


class GraphRead(BaseModel):
    """One graph at one reach, the neighbor-mean predictor on the pinned genes."""

    graph: str
    hops: int
    n_nodes: int
    n_edges: int
    n_pinned: int
    n_pinned_covered: int
    coverage: float
    mean_labeled_neighbors: float
    #: Spearman of the neighbor mean against the target on the covered pinned genes,
    #: the mean over columns for a vector target.
    spearman_covered: float
    #: The same with every uncovered gene filled by the labeled mean, on all pinned genes.
    spearman_filled: float
    top50_enrichment: float
    null_mean: float
    null_sd: float
    z: float
    #: Rate of a top-decile labeled neighbor among top-decile pinned genes over the rate
    #: among the rest, and Fisher's exact p (two-sided); column means.
    top_decile_fold: float
    top_decile_fisher_p: float
    #: On the pinned genes the --coverage-store covers (NaN without one).
    spearman_store_covered: float
    n_store_covered: int
    #: Leave-one-out neighbor mean on the labeled genes themselves.
    n_labeled_covered: int
    labeled_loo_spearman: float


def _mean_spearman(pred: np.ndarray, target: np.ndarray, rows: np.ndarray) -> float:
    """Mean over columns of the Spearman on ``rows`` with a finite target in the column."""
    out = []
    for j in range(target.shape[1]):
        fin = rows & np.isfinite(target[:, j]) & np.isfinite(pred[:, j])
        if fin.sum() < 3:
            continue
        out.append(float(spearmanr(pred[fin, j], target[fin, j])[0]))
    assert out, "no column had three scorable rows"
    return float(np.mean(out))


def neighbor_mean(reach: np.ndarray, y_lab: np.ndarray, lab_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: mean over reachable labeled genes of each column, and the count.

    ``reach`` is ``[rows, n]`` binary, ``y_lab`` is ``[n, c]`` with NaN off the labeled
    genes, ``lab_mask`` ``[n]``. A column's mean uses the labeled neighbors finite in it.
    """
    fin = np.isfinite(y_lab) & lab_mask[:, None]
    y0 = np.where(fin, y_lab, 0.0)
    sums = reach @ y0
    counts = reach @ fin.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means, (reach @ lab_mask.astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(STUDY))
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--coverage-store",
        default=None,
        help=(
            "A store name (ess_solve_encodings_<name>.npz) whose gene coverage splits "
            "the pin, so the neighbor mean is also read on the covered genes alone, as "
            "target_coverage_split.py reads the store arms (the M\"ulleder pool's 73 "
            "uncovered pinned genes are an indicator worth 0.196 on betaxanthin)."
        ),
    )
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    rng = np.random.default_rng(args.seed)
    spec = CSV_TARGETS.get(args.target) or TARGETS[args.target]
    columns = list(spec["columns"])
    csv_rows: dict[str, np.ndarray] = {}
    if "csv" in spec:
        import pandas as pd

        frame = pd.read_csv(spec["csv"])
        assert not frame[spec["gene_column"]].duplicated().any(), spec["csv"]
        frame = frame.set_index(spec["gene_column"])
        csv_rows = {
            str(g): frame.loc[g, columns].to_numpy(np.float64) for g in frame.index
        }

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
    pinned, pinned_report = train_bx.resolve_pinned_test(dataset, cfg)
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
    splits = {
        name: [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        for name in ("train", "val", "test")
    }
    idx_to_genes = record_genes(dataset)
    with open(osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])

    # gene -> target row, for single-deletion genotypes with a measurement.
    per_split: dict[str, dict[str, np.ndarray]] = {}
    for name, indices in splits.items():
        genes = [idx_to_genes.get(i, []) for i in indices]
        if csv_rows:
            values = np.full((len(indices), len(columns)), np.nan)
            for i, gs in enumerate(genes):
                if len(gs) == 1 and gs[0] in csv_rows:
                    values[i] = csv_rows[gs[0]]
        else:
            values = read_targets(base, indices, spec["label"], columns)
        rows: dict[str, np.ndarray] = {}
        for gs, v in zip(genes, values, strict=True):
            if len(gs) == 1 and np.isfinite(v).any():
                assert gs[0] not in rows, f"{gs[0]} twice in {name}"
                rows[gs[0]] = v
        per_split[name] = rows
    labeled = {**per_split["train"], **per_split["val"]}
    pinned_rows = {g: v for g, v in per_split["test"].items() if g in pinned_genes}
    assert not set(labeled) & set(pinned_rows), "a pinned gene is labeled"
    print(
        f"  labeled (train+val, single deletion): {len(labeled)}; pinned test with a "
        f"target: {len(pinned_rows)}",
        flush=True,
    )

    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    undirected = {}
    for name in GRAPHS:
        g = multigraph[name].graph
        undirected[name] = g.to_undirected() if g.is_directed() else g
    import networkx as nx

    undirected["union_of_9"] = nx.compose_all([undirected[n] for n in GRAPHS])

    genes = sorted(
        set().union(*(set(g.nodes) for g in undirected.values()))
        | set(labeled)
        | set(pinned_rows)
    )
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    y = np.full((n, len(columns)), np.nan)
    lab_mask = np.zeros(n, dtype=bool)
    pin_mask = np.zeros(n, dtype=bool)
    for g, v in labeled.items():
        y[index[g]] = v
        lab_mask[index[g]] = True
    for g, v in pinned_rows.items():
        y[index[g]] = v
        pin_mask[index[g]] = True
    y_lab = np.where(lab_mask[:, None], y, np.nan)
    pin_idx = np.flatnonzero(pin_mask)
    lab_idx = np.flatnonzero(lab_mask)
    store_covered = np.ones(len(pin_idx), dtype=bool)
    if args.coverage_store is not None:
        payload = np.load(
            osp.join(
                DATA_ROOT,
                "data/torchcell/reference_flux",
                f"ess_solve_encodings_{args.coverage_store}.npz",
            ),
            allow_pickle=False,
        )
        in_store = {str(g) for g in payload["genes"]}
        store_covered = np.array([genes[i] in in_store for i in pin_idx])
        print(
            f"  coverage store {args.coverage_store}: {int(store_covered.sum())} of "
            f"{len(pin_idx)} pinned genes covered",
            flush=True,
        )
    labeled_mean = np.nanmean(y_lab, axis=0)
    # Top decile of the labeled genes, per column, as the enrichment class.
    lab_top = np.zeros_like(y_lab, dtype=bool)
    pin_top = np.zeros((len(pin_idx), len(columns)), dtype=bool)
    for j in range(len(columns)):
        col = y_lab[:, j]
        fin = np.isfinite(col)
        lab_top[fin, j] = col[fin] >= np.nanquantile(col[fin], TOP_DECILE)
        pc = y[pin_idx, j]
        pf = np.isfinite(pc)
        pin_top[pf, j] = pc[pf] >= np.nanquantile(pc[pf], TOP_DECILE)

    reads: list[GraphRead] = []
    for name, g in undirected.items():
        a = np.zeros((n, n), dtype=np.float32)
        src = [index[u] for u, v in g.edges if u != v]
        dst = [index[v] for u, v in g.edges if u != v]
        a[src, dst] = 1.0
        a[dst, src] = 1.0
        n_edges = int(a.sum() // 2)
        two = None
        for hops in (1, 2):
            if hops == 2:
                two = ((a + a @ a) > 0).astype(np.float32)
                np.fill_diagonal(two, 0.0)
            reach = a if hops == 1 else two
            assert reach is not None
            r_pin = reach[pin_idx]
            means, counts = neighbor_mean(r_pin, y_lab, lab_mask)
            covered = counts > 0
            y_pin = y[pin_idx]
            sp_cov = _mean_spearman(means, y_pin, covered)
            filled = np.where(np.isfinite(means), means, labeled_mean[None, :])
            sp_fill = _mean_spearman(filled, y_pin, np.ones(len(pin_idx), bool))
            top50 = float("nan")
            if len(columns) == 1:
                fin = np.isfinite(y_pin[:, 0])
                top50 = top50_enrichment(filled[fin, 0], y_pin[fin, 0])[0]
            sp_store = (
                _mean_spearman(filled, y_pin, store_covered)
                if args.coverage_store is not None
                else float("nan")
            )
            # Degree-matched null: the same number of labeled neighbors, drawn at random.
            null = []
            k = (r_pin @ lab_mask.astype(np.float32)).astype(int)
            for _ in range(args.permutations):
                r_null = np.zeros_like(r_pin)
                for i, ki in enumerate(k):
                    if ki > 0:
                        r_null[i, rng.choice(lab_idx, ki, replace=False)] = 1.0
                m_null, _ = neighbor_mean(r_null, y_lab, lab_mask)
                null.append(_mean_spearman(m_null, y_pin, covered))
            null_mean, null_sd = float(np.mean(null)), float(np.std(null, ddof=1))
            # Top-decile enrichment: does a top pinned gene touch a top labeled gene?
            folds, ps = [], []
            for j in range(len(columns)):
                has = (r_pin @ lab_top[:, j].astype(np.float32)) > 0
                fin = np.isfinite(y_pin[:, j]) & covered
                t, h = pin_top[fin, j], has[fin]
                table = [
                    [int((t & h).sum()), int((t & ~h).sum())],
                    [int((~t & h).sum()), int((~t & ~h).sum())],
                ]
                rate_top = table[0][0] / max(t.sum(), 1)
                rate_rest = table[1][0] / max((~t).sum(), 1)
                folds.append(rate_top / rate_rest if rate_rest > 0 else float("nan"))
                ps.append(float(fisher_exact(table)[1]))
            # Leave-one-out on the labeled genes: no self-loops, so the row excludes itself.
            m_lab, c_lab = neighbor_mean(reach[lab_idx], y_lab, lab_mask)
            lab_cov = c_lab > 0
            reads.append(
                GraphRead(
                    graph=name,
                    hops=hops,
                    n_nodes=int(g.number_of_nodes()),
                    n_edges=n_edges,
                    n_pinned=int(len(pin_idx)),
                    n_pinned_covered=int(covered.sum()),
                    coverage=float(covered.mean()),
                    mean_labeled_neighbors=float(counts[covered].mean()),
                    spearman_covered=sp_cov,
                    spearman_filled=sp_fill,
                    top50_enrichment=top50,
                    null_mean=null_mean,
                    null_sd=null_sd,
                    z=(sp_cov - null_mean) / null_sd if null_sd > 0 else float("nan"),
                    top_decile_fold=float(np.nanmean(folds)),
                    top_decile_fisher_p=float(np.mean(ps)),
                    spearman_store_covered=sp_store,
                    n_store_covered=int(store_covered.sum()),
                    n_labeled_covered=int(lab_cov.sum()),
                    labeled_loo_spearman=_mean_spearman(m_lab, y[lab_idx], lab_cov),
                )
            )
            r = reads[-1]
            print(
                f"{name:26s} hop {hops} covered {r.n_pinned_covered:4d}/{r.n_pinned} "
                f"(mean {r.mean_labeled_neighbors:6.1f} nbrs) rho {r.spearman_covered:+.3f} "
                f"filled {r.spearman_filled:+.3f} null {r.null_mean:+.3f}+-{r.null_sd:.3f} "
                f"z {r.z:+.1f} top50 {r.top50_enrichment:.2f} decile-fold {r.top_decile_fold:.2f} "
                f"p {r.top_decile_fisher_p:.1e} store-covered {r.spearman_store_covered:+.3f} "
                f"LOO {r.labeled_loo_spearman:+.3f} "
                f"(n {r.n_labeled_covered})",
                flush=True,
            )

    out = {
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_graph_homophily.py",
        "target": args.target,
        "columns": columns,
        "n_labeled": len(labeled),
        "n_pinned": len(pinned_rows),
        "pinned_split": pinned_report,
        "permutations": args.permutations,
        "coverage_store": args.coverage_store,
        "statistic": (
            "pinned-test Spearman of the labeled-neighbor mean (train and validation "
            "genes) on the pinned genes with at least one labeled neighbor within the "
            "reach, column mean; filled with the labeled mean on all pinned genes; a "
            "degree-matched random-neighbor null; top-decile neighbor enrichment; "
            "leave-one-out on the labeled genes"
        ),
        "reads": [r.model_dump() for r in reads],
        "seconds": time.time() - started,
    }
    path = osp.join(
        EXPERIMENT_ROOT, STUDY[args.target], "results", f"target_graph_homophily_{args.target}.json"
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
