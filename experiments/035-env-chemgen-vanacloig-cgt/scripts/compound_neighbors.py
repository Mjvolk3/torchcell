# experiments/035-env-chemgen-vanacloig-cgt/scripts/compound_neighbors.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.compound_neighbors]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/compound_neighbors
"""Which measured compounds does each chemical encoder place nearest to a query molecule?

For a query InChIKey that is in the embedded library (5,472 molecules, 12 encoders from
experiment 031), every compound with chemogenomic measurements in the pooled 033 cell table
is ranked by similarity to the query under each encoder:

- fingerprints (``ecfp4_bit``, ``ecfp4_count``, ``fcfp4_count``, ``maccs``): min/max
  Tanimoto on the raw counts or bits;
- every other encoder (learned embeddings and RDKit descriptors): cosine similarity after
  each dimension is standardized over the library, because the raw vectors of a language
  model share a large common component and their raw cosines all sit near 1.

Writes ``results/compound_neighbors_<name>.csv``, one row per (encoder, measured
compound): the similarity, its rank among all measured compounds, and its rank among the
41 Vanacloig compounds; and prints the top Vanacloig neighbors per encoder with a consensus
(mean rank over encoders).

    python compound_neighbors.py --inchikey QTBSBXVTEAMEQO-UHFFFAOYSA-N --name acetic_acid
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from baseline_ladder import EMBEDDING_DIR  # noqa: E402
from train_factorized import CELL_TABLE  # noqa: E402

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "035-env-chemgen-vanacloig-cgt", "results"
)
FINGERPRINTS = {"ecfp4_bit", "ecfp4_count", "fcfp4_count", "maccs"}
VANACLOIG = "EnvChemgenVanacloig2022Dataset"


def measured_compounds() -> pd.DataFrame:
    """One row per (InChIKey, name) with the datasets that measured it."""
    table = pd.read_parquet(
        CELL_TABLE, columns=["dataset", "compound_names", "inchikeys", "n_compounds"]
    )
    single = table[table["n_compounds"] == 1].drop_duplicates()
    return (
        single.groupby("inchikeys")
        .agg(
            name=("compound_names", "first"),
            datasets=("dataset", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
        .rename(columns={"inchikeys": "inchikey"})
    )


def similarity(encoder: str, x: np.ndarray, query: int) -> np.ndarray:
    """Similarity of every library row to row ``query`` under one encoder."""
    x = x.astype(np.float64)
    x = x[:, np.isfinite(x).all(axis=0)]
    if encoder in FINGERPRINTS:
        q = x[query]
        return np.minimum(x, q).sum(1) / np.maximum(x, q).sum(1)
    sd = x.std(axis=0)
    z = (x[:, sd > 0] - x[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z @ z[query]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inchikey", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    measured = measured_compounds()
    rows = []
    for path in sorted(glob.glob(osp.join(EMBEDDING_DIR, "*.npz"))):
        encoder = osp.basename(path).removesuffix(".npz")
        data = np.load(path, allow_pickle=True)
        keys = list(data["inchikey"])
        assert args.inchikey in keys, f"{args.inchikey} is not embedded by {encoder}"
        sim = pd.Series(
            similarity(encoder, data["X"], keys.index(args.inchikey)), index=keys
        )
        frame = measured[measured["inchikey"].isin(keys)].copy()
        frame = frame[frame["inchikey"] != args.inchikey]
        frame["similarity"] = sim.loc[frame["inchikey"]].to_numpy()
        frame["rank_all_measured"] = frame["similarity"].rank(ascending=False)
        vana = frame["datasets"].str.contains(VANACLOIG)
        frame.loc[vana, "rank_vanacloig"] = frame.loc[vana, "similarity"].rank(
            ascending=False
        )
        rows.append(frame.assign(encoder=encoder))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(osp.join(RESULTS, f"compound_neighbors_{args.name}.csv"), index=False)

    vana = out[out["rank_vanacloig"].notna()]
    print(f"nearest Vanacloig compounds to {args.name}, per encoder (similarity):")
    for encoder, g in vana.groupby("encoder"):
        top = g.nsmallest(args.top, "rank_vanacloig")
        print(
            f"  {encoder:18s}"
            + "; ".join(f"{r.name} {r.similarity:.2f}" for r in top.itertuples())
        )
    consensus = (
        vana.groupby("name")["rank_vanacloig"]
        .agg(mean_rank="mean", best="min", worst="max")
        .sort_values("mean_rank")
    )
    print("\nconsensus over encoders, Vanacloig compounds (rank 1 = nearest):")
    print(consensus.head(12).round(1).to_string())
    every = (
        out.groupby(["name", "datasets"])["rank_all_measured"]
        .mean()
        .sort_values()
        .head(15)
    )
    print(
        f"\nconsensus over encoders, all {out['inchikey'].nunique():,} measured compounds:"
    )
    print(every.round(1).to_string())


if __name__ == "__main__":
    main()
