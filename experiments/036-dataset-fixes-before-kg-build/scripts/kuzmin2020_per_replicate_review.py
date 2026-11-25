# experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_per_replicate_review.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kuzmin2020_per_replicate_review]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_per_replicate_review
"""Per-replicate review of Kuzmin 2020 (#1178 sweep, #1176, #1202).

Reads the released tables and two ``SmfKuzmin2020Dataset`` stores and reports:

- Table S5 "Single mutant" rows: on which side (ORF1 or ORF2) HO (YDL227C) sits, how
  many state a fitness, how many distinct genes the stored rows name (#1202);
- the ``Array single mutant fitness`` column of Tables S1 and S3 (#1176 lead): strains
  per table, whether the value is constant per strain, and how it compares with the
  array single-mutant fitness Kuzmin 2018 releases (Data S1) and with the Costanzo 2016
  single-mutant fitness standard (26 and 30 degrees). The SI says the array values
  "were obtained from a previous study (37)" (si1.md line 45), so the comparison is
  what decides whether they are measurements of this study;
- the combined-mutant colony count back-solve of ``test_kuzmin_n_backsolve.py`` (the n
  whose single-term normal model ``p = 2 Phi(-|eps| / (sd / sqrt(n)))`` best matches the
  median released p-value), run on the digenic rows of Tables S1 and S3;
- ``--before-root`` (origin/main loader) against ``--after-root`` (the #1202 fix),
  record for record: count, how many genotypes changed and that every changed one was
  HO before, the max absolute fitness difference, and any other field that differs.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/kuzmin2020_per_replicate_review.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_per_replicate_review.py \
        --before-root $DATA_ROOT/data/torchcell/smf_kuzmin2020 \
        --after-root $DATA_ROOT/tmp/scratch/per-replicate-kuzmin2020/smf_kuzmin2020
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
HO = "YDL227C"
TOL = 1e-3
CANDIDATE_N = (2, 4, 6, 8, 12)


def sha256(path: str) -> str:
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_store(root: str) -> list[dict[str, Any]]:
    """Every record of ``<root>/processed/lmdb`` in index order, interned refs resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        records = [
            resolve_interned(pickle.loads(txn.get(str(i).encode())), interned)
            for i in range(n)
        ]
    env.close()
    return records


def differing_paths(a: Any, b: Any, path: str = "") -> list[str]:
    """Dotted paths at which two nested dicts differ (a key absent on one side counts)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            out += differing_paths(
                a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}"
            )
        return out
    return [] if a == b else [path]


def s5_review(s5: pd.DataFrame) -> dict[str, Any]:
    """Where HO sits in the S5 single-mutant pairs and what the stored rows name."""
    single = s5[s5["Mutant type"] == "Single mutant"]
    ho1, ho2 = single["ORF1"] == HO, single["ORF2"] == HO
    stored = single.dropna(subset=["Fitness"])
    gene = stored["ORF2"].where(stored["ORF1"] == HO, stored["ORF1"])
    return {
        "single_mutant_rows": len(single),
        "ho_in_orf1": int(ho1.sum()),
        "ho_in_orf2": int(ho2.sum()),
        "ho_on_both_or_neither": int((ho1 == ho2).sum()),
        "rows_with_fitness": len(stored),
        "stored_rows_ho_in_orf1": int((stored["ORF1"] == HO).sum()),
        "stored_rows_ho_in_orf2": int((stored["ORF2"] == HO).sum()),
        "distinct_genes_of_stored_rows": int(gene.nunique()),
    }


def compare(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Shared strains, max |difference| and strains differing by more than TOL."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    diff = (joined["a"] - joined["b"]).abs()
    return {
        "shared_strains": len(joined),
        "max_abs_diff": round(float(diff.max()), 6),
        "n_differ_gt_1e-3": int((diff > TOL).sum()),
        "pearson_r": round(float(joined["a"].corr(joined["b"])), 4),
    }


def array_smf_review(
    s1: pd.DataFrame, s3: pd.DataFrame, k18: pd.DataFrame, costanzo: pd.DataFrame
) -> dict[str, Any]:
    """The S1/S3 array single-mutant fitness against Kuzmin 2018 and Costanzo 2016."""
    out: dict[str, Any] = {}
    per_table: dict[str, pd.Series] = {}
    for label, table in (("S1", s1), ("S3", s3)):
        grouped = table.groupby("Array strain ID")["Array single mutant fitness"]
        per_table[label] = grouped.first()
        out[label] = {
            "array_strains": int(grouped.ngroups),
            "strains_with_more_than_one_value": int((grouped.nunique() > 1).sum()),
            "blank_rows": int(table["Array single mutant fitness"].isna().sum()),
        }
    out["strains_in_both"] = len(
        set(per_table["S1"].index) & set(per_table["S3"].index)
    )
    out["S1_vs_S3"] = compare(per_table["S1"], per_table["S3"])
    k18_smf = k18.groupby("Array strain ID")["Array single mutant fitness"].first()
    std = costanzo.set_index("Strain ID")
    for label, series in per_table.items():
        out[f"{label}_vs_kuzmin2018_data_s1"] = compare(series, k18_smf)
        for temp in ("26", "30"):
            column = f"Single mutant fitness ({temp}°)"
            out[f"{label}_vs_costanzo2016_{temp}C"] = compare(series, std[column])
        out[f"{label}_strains_absent_from_costanzo2016"] = int(
            (~series.index.isin(std.index)).sum()
        )
        out[f"{label}_strains_listed_with_blank_costanzo2016_26C"] = int(
            std.reindex(series.index)["Single mutant fitness (26°)"].isna().sum()
            - (~series.index.isin(std.index)).sum()
        )
    return out


def n_backsolve(table: pd.DataFrame) -> dict[str, Any]:
    """Median-p gap per candidate colony count on the digenic rows of one table."""
    digenic = table[table["Combined mutant type"] == "digenic"]
    eps = digenic["Raw genetic interaction score (epsilon)"].to_numpy(dtype=float)
    sd = digenic["Double/triple mutant fitness standard deviation"].to_numpy(
        dtype=float
    )
    pv = digenic["P-value"].to_numpy(dtype=float)
    keep = np.isfinite(eps) & np.isfinite(sd) & np.isfinite(pv) & (sd > 0) & (pv > 0)
    eps, sd, pv = eps[keep], sd[keep], pv[keep]
    reported = float(np.median(pv))
    gap = {
        n: round(
            abs(
                float(np.median(2 * norm.cdf(-np.abs(eps) / (sd / np.sqrt(n)))))
                - reported
            ),
            4,
        )
        for n in CANDIDATE_N
    }
    return {
        "rows": int(keep.sum()),
        "reported_median_p": round(reported, 4),
        "median_p_gap_per_n": gap,
        "best_n": min(gap, key=lambda n: gap[n]),
    }


def store_review(before_root: str, after_root: str) -> dict[str, Any]:
    """Record-for-record comparison of the two Smf stores."""
    before, after = read_store(before_root), read_store(after_root)
    assert len(before) == len(after), (len(before), len(after))
    changed, was_ho = 0, 0
    other: Counter[str] = Counter()
    max_diff = 0.0
    for old, new in zip(before, after, strict=True):
        old_exp, new_exp = old["experiment"], new["experiment"]
        max_diff = max(
            max_diff,
            abs(old_exp["phenotype"]["fitness"] - new_exp["phenotype"]["fitness"]),
        )
        if old_exp["genotype"] != new_exp["genotype"]:
            changed += 1
            names = [
                p["systematic_gene_name"] for p in old_exp["genotype"]["perturbations"]
            ]
            was_ho += int(names == [HO])
        for key in ("experiment", "reference"):
            for path in differing_paths(old[key], new[key], key):
                if not path.startswith("experiment.genotype"):
                    other[path] += 1
    genes_after = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in after
    }
    return {
        "records_before": len(before),
        "records_after": len(after),
        "genotypes_changed": changed,
        "changed_genotypes_that_were_ho": was_ho,
        "max_abs_fitness_diff": max_diff,
        "other_differing_fields": dict(other),
        "distinct_genes_after": len(genes_after),
        "ho_among_genes_after": HO in genes_after,
    }


def main() -> None:
    """Run the three reviews and write the results JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    raw = osp.join(data_root, "data/torchcell/smf_kuzmin2020/raw")
    files = {
        "S1": osp.join(raw, "aaz5667-Table-S1.xlsx"),
        "S3": osp.join(raw, "aaz5667-Table-S3.xlsx"),
        "S5": osp.join(raw, "aaz5667-Table-S5.xlsx"),
        "kuzmin2018_data_s1": osp.join(
            data_root, "data/torchcell/smf_kuzmin2018/raw/aao1729_data_s1.tsv"
        ),
        "costanzo2016_smf": osp.join(
            data_root,
            "data/torchcell/smf_costanzo2016/raw/strain_ids_and_single_mutant_fitness.xlsx",
        ),
    }
    s1, s3, s5 = (pd.read_excel(files[t], skiprows=1) for t in ("S1", "S3", "S5"))
    k18 = pd.read_csv(files["kuzmin2018_data_s1"], sep="\t")
    costanzo = pd.read_excel(files["costanzo2016_smf"])
    result = {
        "files": {
            name: {"path": path, "sha256": sha256(path)} for name, path in files.items()
        },
        "table_s5_single_mutants": s5_review(s5),
        "array_single_mutant_fitness": array_smf_review(s1, s3, k18, costanzo),
        "combined_mutant_n_backsolve": {"S1": n_backsolve(s1), "S3": n_backsolve(s3)},
        "smf_store_before_vs_after": store_review(args.before_root, args.after_root),
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "kuzmin2020_per_replicate_review.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
