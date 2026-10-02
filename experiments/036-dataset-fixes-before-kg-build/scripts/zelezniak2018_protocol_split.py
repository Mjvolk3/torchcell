# experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_protocol_split.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.zelezniak2018_protocol_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_protocol_split.py
"""Measure the Zelezniak 2018 metabolome protocol split (issue #595) before and after.

Reads, all read-only:

- the pinned raw file ``$DATA_ROOT/data/torchcell/metabolite_zelezniak2018/raw/
  metabolites_dataset.data_prep.tsv`` (sha256 verified against the loader pin);
- the OLD dev-tree LMDB ``$DATA_ROOT/data/torchcell/metabolite_zelezniak2018`` (built by
  the pooling loader);
- the NEW build given by ``--new-root`` (a scratch build of the fixed loader).

Counts, from the raw file: (genotype, metabolite) cells holding rows of more than one
protocol, their rows, and the ratio of per-protocol means inside such a cell. From the old
store: records and their pooled experiment / reference cells. From the new store: records
per protocol, every stored value and reference value checked against the raw rows of ONE
protocol (mean, n, and SE), cells that pool protocols (must be 0), and the two example
cells of the issue. Writes ``results/zelezniak2018_protocol_split.json`` and
``results/zelezniak2018_protocol_split_examples.csv``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_protocol_split.py \
        --new-root <scratch>/metabolite_zelezniak2018
"""

import argparse
import hashlib
import json
import math
import os
import os.path as osp
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.zelezniak2018 import (
    METABOLITE_DATA_FILENAME,
    METABOLITE_DATA_SHA256,
    ZELEZNIAK_METABOLITE_PROTOCOLS,
)
from torchcell.verification.runners import load_records

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
DEV_SUBPATH = "data/torchcell/metabolite_zelezniak2018"
EXAMPLES = [("WT", "3pg;2pg"), ("YIL042C", "r5p")]
_PROTOCOL_BY_TYPE = {
    p.measurement_type: d for d, p in ZELEZNIAK_METABOLITE_PROTOCOLS.items()
}


def _raw_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Cells of (genotype, metabolite) that hold rows of more than one protocol."""
    n_protocols = df.groupby(["genotype", "metabolite_id"])["dataset"].nunique()
    pooled = n_protocols[n_protocols > 1]
    cells = df.set_index(["genotype", "metabolite_id"]).loc[pooled.index]
    means = cells.groupby(["genotype", "metabolite_id", "dataset"])["value"].mean()
    ratios = means.groupby(["genotype", "metabolite_id"]).agg(
        lambda s: s.max() / s.min()
    )
    return {
        "n_rows": len(df),
        "n_genotypes": int(df["genotype"].nunique()),
        "n_metabolites": int(df["metabolite_id"].nunique()),
        "n_cells": int(len(n_protocols)),
        "n_cells_pooling_protocols": int(len(pooled)),
        "n_rows_in_pooled_cells": int(len(cells)),
        "pooled_metabolites": sorted(pooled.index.get_level_values(1).unique()),
        "n_pooled_genotypes": int(pooled.index.get_level_values(0).nunique()),
        "ratio_of_protocol_means_min": float(ratios.min()),
        "ratio_of_protocol_means_median": float(ratios.median()),
        "ratio_of_protocol_means_max": float(ratios.max()),
        "n_cells_ratio_over_100": int((ratios > 100).sum()),
        "n_cells_ratio_over_1000": int((ratios > 1000).sum()),
        "duplicate_key_rows_with_dataset": int(
            df.duplicated(["dataset", "metabolite_id", "genotype", "replicate"]).sum()
        ),
        "rows_per_protocol": {
            str(k): int(v) for k, v in df.groupby("dataset").size().items()
        },
    }


def _old_stats(records: list[dict[str, Any]], pooled: set[tuple[str, str]]) -> Any:
    """Records of the pooling build and how many of their cells pool protocols."""
    exp_cells = 0
    exp_records = 0
    ref_cells = 0
    ref_records = 0
    for rec in records:
        orf = rec["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        hit = sum(
            (orf, m) in pooled
            for m in rec["experiment"]["phenotype"]["metabolite_level"]
        )
        exp_cells += hit
        exp_records += hit > 0
        ref = rec["reference"]["phenotype_reference"]["metabolite_level"]
        rhit = sum(("WT", m) in pooled for m in ref)
        ref_cells += rhit
        ref_records += rhit > 0
    return {
        "n_records": len(records),
        "measurement_types": sorted(
            {r["experiment"]["phenotype"]["measurement_type"] for r in records}
        ),
        "n_records_with_pooled_experiment_cells": exp_records,
        "n_pooled_experiment_cells": exp_cells,
        "n_records_with_pooled_reference_cells": ref_records,
        "n_pooled_reference_cells": ref_cells,
    }


def _matches(
    phen: dict[str, Any],
    genotype: str,
    dataset: int,
    truth: dict[tuple[int, str, str], tuple[float, float, int]],
) -> tuple[int, int]:
    """(cells checked, cells whose mean, n and SE do not equal the one-protocol raw rows)."""
    bad = 0
    se = phen["metabolite_level_se"]
    for met, level in phen["metabolite_level"].items():
        mean, std, count = truth[(dataset, genotype, met)]
        n = int(count)
        expected_se = std / math.sqrt(n) if n > 1 else float("nan")
        got_se = float("nan") if se is None else se[met]
        se_ok = (math.isnan(expected_se) and math.isnan(got_se)) or math.isclose(
            got_se, expected_se, rel_tol=1e-12, abs_tol=1e-15
        )
        if not (
            math.isclose(level, mean, rel_tol=1e-12)
            and phen["n_replicates"][met] == n
            and se_ok
        ):
            bad += 1
    return len(phen["metabolite_level"]), bad


def _new_stats(records: list[dict[str, Any]], df: pd.DataFrame) -> dict[str, Any]:
    """Every new record checked against the raw rows of exactly one protocol."""
    grouped = df.groupby(["dataset", "genotype", "metabolite_id"])["value"]
    truth: dict[tuple[int, str, str], tuple[float, float, int]] = {}
    for (dataset, genotype, met), values in grouped:
        vals = [float(v) for v in values]
        mean = sum(vals) / len(vals)
        std = (
            math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            if len(vals) > 1
            else float("nan")
        )
        truth[(int(str(dataset)), str(genotype), str(met))] = (mean, std, len(vals))
    per_protocol: dict[str, int] = {}
    checked = mismatched = ref_checked = ref_mismatched = 0
    type_mismatch = 0
    for rec in records:
        phen = rec["experiment"]["phenotype"]
        ref = rec["reference"]["phenotype_reference"]
        dataset = _PROTOCOL_BY_TYPE[phen["measurement_type"]]
        per_protocol[str(dataset)] = per_protocol.get(str(dataset), 0) + 1
        type_mismatch += ref["measurement_type"] != phen["measurement_type"]
        orf = rec["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        c, b = _matches(phen, orf, dataset, truth)
        checked += c
        mismatched += b
        c, b = _matches(ref, "WT", dataset, truth)
        ref_checked += c
        ref_mismatched += b
    keys = [
        (
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"],
            r["experiment"]["phenotype"]["measurement_type"],
        )
        for r in records
    ]
    return {
        "n_records": len(records),
        "records_per_protocol": dict(sorted(per_protocol.items())),
        "n_duplicate_genotype_protocol_keys": len(keys) - len(set(keys)),
        "n_experiment_cells_checked": checked,
        "n_experiment_cells_not_matching_one_protocol": mismatched,
        "n_reference_cells_checked": ref_checked,
        "n_reference_cells_not_matching_one_protocol": ref_mismatched,
        "n_cells_pooling_protocols": mismatched + ref_mismatched,
        "n_records_reference_protocol_differs": int(type_mismatch),
    }


def _examples(
    old: list[dict[str, Any]], new: list[dict[str, Any]], df: pd.DataFrame
) -> list[dict[str, Any]]:
    """Raw rows, old stored value and new per-protocol stored values for each example."""
    out: list[dict[str, Any]] = []
    for genotype, met in EXAMPLES:
        raw = df[(df["genotype"] == genotype) & (df["metabolite_id"] == met)]
        side = "reference" if genotype == "WT" else "experiment"
        old_vals = _stored(old, genotype, met, side)
        new_vals = _stored(new, genotype, met, side)
        for dataset in sorted(int(d) for d in raw["dataset"].unique()):
            sub = raw[raw["dataset"] == dataset]
            mtype = ZELEZNIAK_METABOLITE_PROTOCOLS[dataset].measurement_type
            stored = [v for v in new_vals if v["measurement_type"] == mtype]
            out.append(
                {
                    "genotype": genotype,
                    "metabolite_id": met,
                    "dataset": dataset,
                    "raw_values": ";".join(f"{v:.6f}" for v in sub["value"]),
                    "old_stored": json.dumps(old_vals[0]) if old_vals else "",
                    "new_stored_mean": stored[0]["level"],
                    "new_stored_se": stored[0]["se"],
                    "new_stored_n": stored[0]["n"],
                    "new_measurement_type": mtype,
                    "n_distinct_new_values": len({json.dumps(v) for v in stored}),
                }
            )
    return out


def _stored(
    records: list[dict[str, Any]], genotype: str, met: str, side: str
) -> list[dict[str, Any]]:
    """Every stored (level, se, n, measurement_type) for ``met`` of ``genotype``."""
    vals = []
    for rec in records:
        if side == "experiment":
            orf = rec["experiment"]["genotype"]["perturbations"][0][
                "systematic_gene_name"
            ]
            if orf != genotype:
                continue
            phen = rec["experiment"]["phenotype"]
        else:
            phen = rec["reference"]["phenotype_reference"]
        if met not in phen["metabolite_level"]:
            continue
        se = phen["metabolite_level_se"]
        vals.append(
            {
                "level": phen["metabolite_level"][met],
                "se": None if se is None or math.isnan(se[met]) else se[met],
                "n": phen["n_replicates"][met],
                "measurement_type": phen["measurement_type"],
            }
        )
    # The WT reference repeats on every record; keep each distinct value once.
    unique = {json.dumps(v, sort_keys=True): v for v in vals}
    return list(unique.values())


def main() -> None:
    """Measure raw, old and new stores and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True)
    args = parser.parse_args()
    load_dotenv()
    dev_root = osp.join(os.environ["DATA_ROOT"], DEV_SUBPATH)
    raw_path = osp.join(dev_root, "raw", METABOLITE_DATA_FILENAME)
    with open(raw_path, "rb") as fh:
        raw_sha = hashlib.sha256(fh.read()).hexdigest()
    if raw_sha != METABOLITE_DATA_SHA256:
        raise RuntimeError(f"{raw_path} sha256 {raw_sha} is off the loader pin")
    df = pd.read_csv(raw_path, sep="\t")

    raw = _raw_stats(df)
    n_protocols = df.groupby(["genotype", "metabolite_id"])["dataset"].nunique()
    pooled = set(n_protocols[n_protocols > 1].index)
    old_records = load_records(dev_root)
    new_records = load_records(args.new_root)
    result = {
        "inputs": {
            "raw": raw_path,
            "raw_sha256": raw_sha,
            "old_store": dev_root,
            "new_store": args.new_root,
        },
        "raw": raw,
        "old": _old_stats(old_records, pooled),
        "new": _new_stats(new_records, df),
        "protocols": {
            str(d): {
                "measurement_type": p.measurement_type,
                "calibrated": p.calibrated,
                "unit": p.unit,
                "unit_gap": None if p.unit_gap is None else p.unit_gap.reason.value,
            }
            for d, p in ZELEZNIAK_METABOLITE_PROTOCOLS.items()
        },
    }
    examples = _examples(old_records, new_records, df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "zelezniak2018_protocol_split.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    pd.DataFrame(examples).to_csv(
        osp.join(RESULTS_DIR, "zelezniak2018_protocol_split_examples.csv"), index=False
    )
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(examples).to_string())


if __name__ == "__main__":
    main()
