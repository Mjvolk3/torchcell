# experiments/025-solid-growth/scripts/query_pair_disjoint_gene_coverage.py
# [[experiments.025-solid-growth.scripts.query_pair_disjoint_gene_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/query_pair_disjoint_gene_coverage
"""How cold is a held-out gene on the query-pair-disjoint split.

The Q split holds out whole query PAIRS, not genes. A learnable per-gene row is trained
only on genes that are perturbed in some training triple, so what decides whether the
table is cold on val and test is how many held-out genes appear in a training triple in
any position (query member or array gene). This script counts that from the recap table
(one row per S0 triple with its three genes) and the split artifact, and writes the counts
to results/query_pair_disjoint_gene_coverage.json.
"""

import gzip
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
RECAP = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)
SUBSET = "subset_S0_indices.json.gz"
SPLITS = {
    "Q": ("query_pair_disjoint_splits_025.json.gz", "splits"),
    "R": ("pinned_splits_from_010_seed_42.json.gz", "pinned"),
}


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def main() -> None:
    subset = np.array(sorted(load_gz(SUBSET)), dtype=np.int64)
    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"])
    recap = recap.set_index("idx_025")
    genes3 = recap.loc[subset, ["gene_a", "gene_b", "gene_c"]].to_numpy()
    id_to_row = {int(r): i for i, r in enumerate(subset)}

    out: dict[str, dict] = {}
    for arm, (fname, key) in SPLITS.items():
        parts = load_gz(fname)[key]
        rows = {
            name: np.array([id_to_row[int(r)] for r in parts[name]], dtype=np.int64)
            for name in ("train", "val", "test")
        }
        train_genes = set(genes3[rows["train"]].ravel().tolist())
        arm_out: dict[str, object] = {
            "n_train_records": int(rows["train"].size),
            "n_train_genes": len(train_genes),
        }
        for name in ("val", "test"):
            g = genes3[rows[name]]
            held = set(g.ravel().tolist())
            seen = held & train_genes
            in_train = np.isin(g, list(train_genes))
            arm_out[name] = {
                "n_records": int(rows[name].size),
                "n_genes": len(held),
                "n_genes_in_a_training_triple": len(seen),
                "frac_genes_in_a_training_triple": len(seen) / len(held),
                "frac_records_all_three_genes_in_training": float(
                    in_train.all(axis=1).mean()
                ),
                "frac_records_with_a_gene_never_in_training": float(
                    (~in_train).any(axis=1).mean()
                ),
                "n_genes_never_in_training": len(held - train_genes),
            }
        out[arm] = arm_out
        print(f"=== {arm}: {json.dumps(arm_out, indent=1)}")

    path = osp.join(RESULTS_DIR, "query_pair_disjoint_gene_coverage.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
