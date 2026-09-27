# experiments/031-env-chemgen-inhibitor-tolerance/scripts/environment_axis_coverage.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.environment_axis_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/environment_axis_coverage
"""Across the five served chemogenomic datasets, how much of the application panel's
chemistry is covered, and can their doses be put on one scale?

The representation design pools five datasets so one model can train across them. Two
properties of the ENVIRONMENT axis decide whether that pooling buys anything.

CHEMICAL COVERAGE. Vanacloig is the application target and its 41 compounds are the
hydrolysate panel. Hillenmeyer alone left isobutanol with no chemical neighbor anywhere
(nearest was sorbitol at Tanimoto 0.14). Wildenhain carries thousands of compounds, so the
question is whether the pooled panel now puts a real neighbor next to each Vanacloig
compound, and which dataset supplies it. For every Vanacloig compound this measures the
nearest neighbor in each other dataset and in their union, and the fraction of the panel
covered at a similarity threshold. Tanimoto on count ECFP4 is the metric, because cosine
between dense embeddings is high for any two molecules and says nothing about closeness.

DOSE COMPARABILITY. A pooled model sees one dose field. These datasets do not dose alike:
some state a target inhibition, some a fixed molarity, some a mass per volume. A dose that
means different things in different rows is a nuisance variable, not a feature. This
measures, per dataset, which units and dose bases appear, what fraction of records carry a
molar concentration at all, and how many distinct doses each compound gets. For compounds
shared between datasets it compares the actual molar values, which is the only way to see
whether a shared compound is even dosed comparably.

Writes ``results/chemical_space_coverage.csv`` (per Vanacloig compound, its nearest neighbor
in each dataset), ``results/chemical_space_thresholds.csv`` (coverage of the panel at
thresholds), ``results/compound_overlap_matrix.csv``, ``results/dose_axis_summary.csv`` and
``results/dose_shared_compounds.csv``.
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray

from torchcell.molecule.similarity import tanimoto_matrix

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMB_DIR = osp.join(RESULTS_DIR, "embeddings")
DATASETS = [
    "vanacloig2022",
    "hillenmeyer2008_hom",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
]
TARGET = "vanacloig2022"
THRESHOLDS = (0.3, 0.4, 0.5, 0.7, 0.9)
# molar conversion for the units that are unambiguous without a molecular weight
TO_MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}


def dosed_compounds(name: str) -> pd.DataFrame:
    """One row per (inchikey, dose) actually dosed in this dataset, singles only."""
    path = osp.join(RESULTS_DIR, f"records_{name}.parquet")
    df = pd.read_parquet(
        path,
        columns=[
            "compound",
            "inchikey",
            "n_small_molecules",
            "dose_value",
            "dose_unit",
            "dose_basis",
        ],
    )
    df = df[(df["n_small_molecules"] == 1) & (df["inchikey"] != "")]
    df["dataset"] = name
    return df


def load_fingerprint() -> tuple[dict[str, int], NDArray[np.float64]]:
    """Count ECFP4 for every embedded compound, the one metric where closeness is meaningful."""
    z = np.load(osp.join(EMB_DIR, "ecfp4_count.npz"), allow_pickle=True)
    keys = [str(k) for k in z["inchikey"]]
    return {k: i for i, k in enumerate(keys)}, np.asarray(z["X"], dtype=np.float64)


def nearest_in(
    X: NDArray[np.float64],
    idx: dict[str, int],
    query: list[str],
    pool: list[str],
    names: dict[str, str],
) -> pd.DataFrame:
    """For each query compound, its most similar pool compound and that similarity."""
    q = [k for k in query if k in idx]
    p = [k for k in pool if k in idx]
    if not q or not p:
        return pd.DataFrame()
    S = tanimoto_matrix(X[[idx[k] for k in q]], X[[idx[k] for k in p]])
    # a compound present in both sets would match itself at 1.0; keep it but flag it
    best = S.argmax(axis=1)
    return pd.DataFrame(
        {
            "inchikey": q,
            "compound": [names.get(k, "") for k in q],
            "neighbor_inchikey": [p[j] for j in best],
            "neighbor": [names.get(p[j], "") for j in best],
            "similarity": S[np.arange(len(q)), best],
            "exact": [q[i] == p[best[i]] for i in range(len(q))],
        }
    )


def molar(row: pd.Series) -> float:
    """The dose in molar where the unit allows it without a molecular weight."""
    unit = str(row["dose_unit"])
    try:
        value = float(row["dose_value"])
    except (TypeError, ValueError):
        return float("nan")
    return value * TO_MOLAR[unit] if unit in TO_MOLAR else float("nan")


def main() -> None:
    available = [
        n for n in DATASETS if osp.exists(osp.join(RESULTS_DIR, f"records_{n}.parquet"))
    ]
    print(f"datasets available: {available}")
    frames = [dosed_compounds(n) for n in available]
    allc = pd.concat(frames, ignore_index=True)
    names = allc.drop_duplicates("inchikey").set_index("inchikey")["compound"].to_dict()
    per_dataset = {
        n: sorted(set(allc.loc[allc["dataset"] == n, "inchikey"])) for n in available
    }
    for n in available:
        print(f"  {n:22s} {len(per_dataset[n]):5d} distinct dosed compounds")

    # ------------------------------------------------------------ overlap matrix
    ov = pd.DataFrame(index=available, columns=available, dtype=int)
    for a in available:
        for b in available:
            ov.loc[a, b] = len(set(per_dataset[a]) & set(per_dataset[b]))
    ov.to_csv(osp.join(RESULTS_DIR, "compound_overlap_matrix.csv"))
    print("\nexact compound overlap:")
    print(ov.to_string())

    # -------------------------------------------------- chemical space coverage
    idx, X = load_fingerprint()
    missing = [k for k in per_dataset[TARGET] if k not in idx]
    covered = {n: len([k for k in per_dataset[n] if k in idx]) for n in available}
    print(f"\nembedded compounds per dataset (count ECFP4): {covered}")
    if missing:
        print(f"  target compounds with no embedding: {len(missing)}")

    rows = []
    for source in available:
        if source == TARGET:
            continue
        nn = nearest_in(X, idx, per_dataset[TARGET], per_dataset[source], names)
        if nn.empty:
            continue
        nn["source"] = source
        rows.append(nn)
    union_pool = sorted({k for n in available if n != TARGET for k in per_dataset[n]})
    nn_union = nearest_in(X, idx, per_dataset[TARGET], union_pool, names)
    nn_union["source"] = "union_of_others"
    rows.append(nn_union)
    cov = pd.concat(rows, ignore_index=True)
    cov.to_csv(osp.join(RESULTS_DIR, "chemical_space_coverage.csv"), index=False)

    # coverage of the panel at thresholds, counting only NON-exact neighbors as new
    tr = []
    for source, block in cov.groupby("source"):
        n_target = len([k for k in per_dataset[TARGET] if k in idx])
        row: dict[str, object] = {
            "source": source,
            "n_target_embedded": n_target,
            "n_exact_match": int(block["exact"].sum()),
            "nn_similarity_median": float(block["similarity"].median()),
        }
        inexact = block[~block["exact"]]
        for t in THRESHOLDS:
            row[f"n_above_{t}"] = int((inexact["similarity"] >= t).sum())
        tr.append(row)
    thresholds = pd.DataFrame(tr).sort_values("nn_similarity_median", ascending=False)
    thresholds.to_csv(
        osp.join(RESULTS_DIR, "chemical_space_thresholds.csv"), index=False
    )
    print("\npanel coverage by source (non-exact neighbors only):")
    print(thresholds.to_string(index=False))

    print("\nnearest neighbor for the hydrolysate compounds, union of the other four:")
    watch = [
        "isobutanol",
        "furfural",
        "5-hydroxymethylfurfural",
        "ethanol",
        "vanillin",
        "ferulic acid",
        "p-coumaric acid",
        "syringaldehyde",
        "acetic acid",
    ]
    u = nn_union[nn_union["compound"].isin(watch)]
    if not u.empty:
        print(u[["compound", "neighbor", "similarity", "exact"]].to_string(index=False))

    # ----------------------------------------------------------- the dose axis
    allc["molar"] = allc.apply(molar, axis=1)
    ds = []
    for n in available:
        b = allc[allc["dataset"] == n]
        per_compound = b.groupby("inchikey")["dose_value"].nunique()
        ds.append(
            {
                "dataset": n,
                "records": len(b),
                "compounds": b["inchikey"].nunique(),
                "dose_units": "|".join(sorted(set(b["dose_unit"].astype(str)) - {""})),
                "dose_bases": "|".join(sorted(set(b["dose_basis"].astype(str)) - {""})),
                "frac_with_dose_value": float(
                    (b["dose_value"].astype(str) != "").mean()
                ),
                "frac_convertible_to_molar": float(b["molar"].notna().mean()),
                "doses_per_compound_median": float(per_compound.median()),
                "doses_per_compound_max": int(per_compound.max()),
                "molar_min": float(b["molar"].min(skipna=True)),
                "molar_median": float(b["molar"].median(skipna=True)),
                "molar_max": float(b["molar"].max(skipna=True)),
            }
        )
    dose = pd.DataFrame(ds)
    dose.to_csv(osp.join(RESULTS_DIR, "dose_axis_summary.csv"), index=False)
    print("\nthe dose axis per dataset:")
    print(dose.to_string(index=False))

    # for compounds in more than one dataset, are the molar doses comparable
    shared = (
        allc.dropna(subset=["molar"])
        .groupby("inchikey")
        .filter(lambda g: g["dataset"].nunique() > 1)
    )
    if not shared.empty:
        piv = (
            shared.groupby(["inchikey", "compound", "dataset"])["molar"]
            .agg(["min", "median", "max"])
            .reset_index()
        )
        piv.to_csv(osp.join(RESULTS_DIR, "dose_shared_compounds.csv"), index=False)
        span = shared.groupby("inchikey")["molar"].agg(["min", "max"])
        ratio = (span["max"] / span["min"]).replace([np.inf, -np.inf], np.nan).dropna()
        print(
            f"\ncompounds dosed in more than one dataset with a molar value: "
            f"{shared['inchikey'].nunique()}; median max/min dose ratio across datasets "
            f"{ratio.median():.1f}, 90th percentile {ratio.quantile(0.9):.1f}"
        )
    print(f"\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
