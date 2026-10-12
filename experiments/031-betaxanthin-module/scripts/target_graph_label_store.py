# experiments/031-betaxanthin-module/scripts/target_graph_label_store.py
# [[experiments.031-betaxanthin-module.scripts.target_graph_label_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_graph_label_store

r"""The graphs' one-hop label propagation of the target as an affine store.

``target_graph_homophily.py`` (2026-09-27) found the betaxanthin target smooth on the
STRING coexpression graph: the mean target of a pinned gene's labeled one-hop neighbors
reads the pin at 0.244 (null $-0.005 \pm 0.041$ over degree-matched random neighbor
sets), the union of the nine graphs at 0.221 with a top-50 of 5.19, with nothing fitted.
The transformer's graph arms (the KL regularizer, the hard mask) shape attention over
gene embeddings and never see a neighbor's label, so that signal is not what they read.
This script writes it as a store over the gene axis, so the same slot that reads the
amino-acid pool reads the graphs' propagated label and the two can be concatenated:
whether the pool adds to the graphs' own read of betaxanthin is then one paired
comparison in the trainer.

Columns, per graph (the nine the trainer masks and regularizes, and their union): the
mean target over the gene's labeled one-hop neighbors, and $\log(1 + n)$ of their count.
Labeled means the training and validation genes with a measurement on the trainer's
pinned split; a pinned test gene's own label is never read. A labeled gene's row is
leave-one-out (no graph has self-loops after their removal, so its neighbors never
include itself). A gene with no labeled neighbor in a graph carries the labeled mean and
a count of zero. Rows: every gene of the genome, so no coverage indicator enters the
slot (``target_coverage_split.py``). The target is signed, so each mean column is
shifted by the labeled minimum of its column to be nonnegative, which keeps
$\log(1 + v)$ monotone: the store is then read exactly as the pool is, ``log1p`` in the
trainer and ``flux: True`` in the ladder, so the pool, the graphs and their
concatenation share one encoding. Stores are never overwritten. Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_graph_label_store.py \
        --target betaxanthin --out YPD_bx_graphlabel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

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
from target_graph_homophily import GRAPHS, neighbor_mean  # noqa: E402
from target_store_baselines import CSV_TARGETS, STUDY  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")


class LabelStoreProvenance(BaseModel):
    """Where the propagated labels came from and which genes' labels were read."""

    model_config = {"extra": "forbid"}

    method: str
    target: str
    target_columns: list[str]
    pinned_split_file: str
    split_seed: int
    n_labeled_genes: int
    n_pinned_genes_excluded: int
    graphs: list[str]
    n_genes: int
    n_columns: int
    column_layout: str
    leave_one_out: bool
    fill_for_no_labeled_neighbor: str
    #: Subtracted from every mean column of the target column named, so the stored
    #: values are nonnegative and log1p is monotone.
    mean_column_shift: dict[str, float]
    python_hash_seed: str
    flux_sha256: str


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(STUDY))
    ap.add_argument("--out", required=True, help="store name: ess_solve_encodings_<out>")
    ap.add_argument("--out-dir", default=TABLE_DIR)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    path = osp.join(args.out_dir, f"ess_solve_encodings_{args.out}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    spec = CSV_TARGETS.get(args.target) or TARGETS[args.target]
    columns = list(spec["columns"])
    assert "csv" not in spec, "the csv targets are not in the trainer's dataset"

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
    labeled: dict[str, np.ndarray] = {}
    for name in ("train", "val"):
        indices = [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        values = read_targets(base, indices, spec["label"], columns)
        for i, v in zip(indices, values, strict=True):
            gs = idx_to_genes.get(i, [])
            if len(gs) == 1 and np.isfinite(v).any():
                assert gs[0] not in labeled, f"{gs[0]} twice"
                assert gs[0] not in pinned_genes, f"{gs[0]} is pinned and {name}"
                labeled[gs[0]] = v
    test_indices = [int(i) for i in dm.test_dataset.indices]
    n_pinned_excluded = len(
        {
            idx_to_genes[i][0]
            for i in test_indices
            if len(idx_to_genes.get(i, [])) == 1 and idx_to_genes[i][0] in pinned_genes
        }
    )
    print(
        f"  labeled (train+val, single deletion): {len(labeled)}; pinned genes whose "
        f"label is never read: {n_pinned_excluded}",
        flush=True,
    )

    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    import networkx as nx

    undirected = {}
    for name in GRAPHS:
        g = multigraph[name].graph
        undirected[name] = g.to_undirected() if g.is_directed() else g
    undirected["union_of_9"] = nx.compose_all([undirected[n] for n in GRAPHS])

    genes = sorted(genome.gene_set)
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    assert all(g in index for g in labeled), "a labeled gene outside the genome"
    y_lab = np.full((n, len(columns)), np.nan)
    lab_mask = np.zeros(n, dtype=bool)
    for g, v in labeled.items():
        y_lab[index[g]] = v
        lab_mask[index[g]] = True
    labeled_mean = np.nanmean(y_lab, axis=0)
    labeled_min = np.nanmin(y_lab, axis=0)

    blocks: list[np.ndarray] = []
    names: list[str] = []
    wild: list[float] = []
    for name, g in undirected.items():
        a = np.zeros((n, n), dtype=np.float32)
        edges = [(index[u], index[v]) for u, v in g.edges if u != v and u in index and v in index]
        src = [u for u, _ in edges]
        dst = [v for _, v in edges]
        a[src, dst] = 1.0
        a[dst, src] = 1.0
        assert not a.diagonal().any()
        means, counts = neighbor_mean(a, y_lab, lab_mask)
        filled = np.where(np.isfinite(means), means, labeled_mean[None, :])
        for j, c in enumerate(columns):
            blocks.append((filled[:, j] - labeled_min[j]).astype(np.float32))
            names.append(f"{name}_nbr_mean_{c}")
            wild.append(float(labeled_mean[j] - labeled_min[j]))
        blocks.append(np.log1p(counts).astype(np.float32))
        names.append(f"{name}_log1p_labeled_neighbors")
        wild.append(float(np.log1p(counts).mean()))
        print(
            f"  {name:26s} genes with a labeled neighbor {int((counts > 0).sum()):5d}/{n}; "
            f"mean count {counts[counts > 0].mean():.1f}",
            flush=True,
        )
    values = np.stack(blocks, axis=1)
    assert np.isfinite(values).all() and (values >= 0).all()
    provenance = LabelStoreProvenance(
        method=(
            "one-hop label propagation: the mean target over a gene's labeled neighbors "
            "in each graph and log1p of their count; written by target_graph_label_store.py"
        ),
        target=args.target,
        target_columns=columns,
        pinned_split_file=cfg.data.pinned_test_split_file,
        split_seed=int(cfg.data.split_seed),
        n_labeled_genes=len(labeled),
        n_pinned_genes_excluded=n_pinned_excluded,
        graphs=list(undirected),
        n_genes=n,
        n_columns=len(names),
        column_layout="per graph: <graph>_nbr_mean_<column> ..., <graph>_log1p_labeled_neighbors",
        leave_one_out=True,
        fill_for_no_labeled_neighbor="the labeled mean of the column; count 0",
        mean_column_shift={
            c: float(labeled_min[j]) for j, c in enumerate(columns)
        },
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        flux_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=values,
        v_wild_type=np.array(wild, np.float32),
        rxn_ids=np.array(names),
        provenance=np.array(provenance.model_dump_json()),
    )
    results_dir = osp.join(EXPERIMENT_ROOT, STUDY[args.target], "results")
    with open(osp.join(results_dir, f"target_graph_label_store_{args.out}.json"), "w") as fh:
        json.dump(
            {
                "generated_by": "experiments/031-betaxanthin-module/scripts/target_graph_label_store.py",
                "store": path,
                "shape": list(values.shape),
                "columns": names,
                "provenance": provenance.model_dump(),
                "seconds": time.time() - started,
            },
            fh,
            indent=2,
        )
    print(f"{path}: {values.shape}", flush=True)


if __name__ == "__main__":
    main()
