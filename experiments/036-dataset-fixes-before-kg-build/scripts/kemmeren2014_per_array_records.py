# experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_per_array_records.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kemmeren2014_per_array_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_per_array_records
"""Check the Kemmeren 2014 per-array conversion (#1178, #1174) against the release.

Reads, read-only: the converted store (``--new-root``, one record per GEO array), the
pre-conversion store (``--old-root``, one record per mutant, the mean over its arrays
of the LAST spot of each gene), and the six GEO pickles of ``--raw-dir``. Reports:

1. counts: records per series, deletion mutants by array count, WT arrays, distinct
   replicate ids, genes per record, ``n_replicates`` values;
2. mean reproduction: the new deletion records grouped by genotype, the per-gene mean
   of ``expression_log2_ratio`` / ``expression`` and of the reference ``expression``,
   against the old stored values (max absolute difference), and the same with each
   array recomputed from the raw table keeping only the last spot of a gene (the old
   rule), which must reproduce the old values to float precision;
3. orientation: for every array, whether GEO ``VALUE`` equals +log2(test/reference) or
   -log2(test/reference) of the Signal Norm columns under the GEO channel assignment;
   the deleted-gene sign on deletion arrays; the MATa marker contrast on MATa arrays.

Writes ``results/kemmeren2014_per_array_records.json``. Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_per_array_records.py \
        --new-root <scratch>/microarray_kemmeren2014 \
        --old-root $DATA_ROOT/data/torchcell/microarray_kemmeren2014 \
        --raw-dir $DATA_ROOT/data/torchcell/microarray_kemmeren2014/raw
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import pickle
from collections import Counter, defaultdict
from typing import Any

import lmdb
import numpy as np
import pandas as pd

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.kemmeren2014 import (
    WT_SERIES_STRAIN,
    MicroarrayKemmeren2014Dataset,
)

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
DELETION_SERIES = ("GSE42527", "GSE42526")
A_GENES = ("YFL026W", "YDR461W", "YNL145W", "YIL015W", "YGL032C", "YKL209C")
ALPHA_GENES = ("YKL178C", "YPL187W", "YGL089C", "YJR004C")


def read_store(root: str) -> list[dict[str, Any]]:
    """All records of a built store, interned sub-objects spliced back in, key order."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        rows = [(int(k.decode()), pickle.loads(v)) for k, v in txn.cursor()]
    env.close()
    return [resolve_interned(record, interned) for _, record in sorted(rows)]


def deleted_orf(record: dict[str, Any]) -> str | None:
    """The deleted ORF of a record, ``None`` for an empty (wildtype) genotype."""
    perturbations = record["experiment"]["genotype"]["perturbations"]
    return perturbations[0]["systematic_gene_name"] if perturbations else None


def last_spot_values(
    gsm: Any, probe_to_gene: dict[str, str]
) -> dict[str, tuple[float, float, float]]:
    """Per gene (log2 ratio, test, reference) of the LAST mapped spot (the old rule)."""
    test_column, reference_column = MicroarrayKemmeren2014Dataset._channel_columns(gsm)
    out: dict[str, tuple[float, float, float]] = {}
    table = gsm.table
    for probe, test, reference in zip(
        table["ID_REF"], table[test_column], table[reference_column], strict=True
    ):
        gene = probe_to_gene.get(str(int(probe)))
        if gene is None:
            continue
        test, reference = float(test), float(reference)
        out[gene] = (float(np.log2(test / reference)), test, reference)
    return out


def max_abs_diff(a: dict[str, float], b: dict[str, float]) -> float:
    """Max |a - b| over shared keys; the key sets must match."""
    if set(a) != set(b):
        raise ValueError("key sets differ")
    return max(abs(a[k] - b[k]) for k in a)


def main() -> None:
    """Measure counts, mean reproduction and orientation; write the results JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()

    new = read_store(args.new_root)
    old = read_store(args.old_root)
    samples = pd.read_csv(
        osp.join(args.new_root, "preprocess", "data.csv"), keep_default_na=False
    )
    result: dict[str, Any] = {
        "new_root": args.new_root,
        "old_root": args.old_root,
        "records_old": len(old),
        "records_new": len(new),
        "records_per_series": samples["series"].value_counts().to_dict(),
    }
    deletion_rows = samples[samples["systematic_gene_name"] != ""]
    result["deletion_records"] = len(deletion_rows)
    result["wildtype_records"] = int((samples["systematic_gene_name"] == "").sum())
    result["mutants_by_array_count"] = {
        str(k): int(v)
        for k, v in deletion_rows.groupby("systematic_gene_name")
        .size()
        .value_counts()
        .items()
    }
    result["replicate_ids_unique"] = bool(samples["replicate_id"].is_unique)
    result["reference_dye_counts"] = samples["reference_dye"].value_counts().to_dict()
    result["strain_counts"] = samples["strain"].value_counts().to_dict()
    result["genes_per_record"] = dict(
        Counter(len(r["experiment"]["phenotype"]["expression_log2_ratio"]) for r in new)
    )
    result["n_replicates_values"] = sorted(
        {v for r in new for v in r["experiment"]["phenotype"]["n_replicates"].values()}
    )
    result["records_with_se"] = sum(
        r["experiment"]["phenotype"]["expression_log2_ratio_se"] is not None
        for r in new
    )

    # 2. mean reproduction against the old one-record-per-mutant store
    old_by_orf = {deleted_orf(r): r for r in old}
    new_by_orf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in new:
        orf = deleted_orf(record)
        if orf is not None:
            new_by_orf[orf].append(record)
    if set(old_by_orf) != set(new_by_orf):
        raise ValueError("the two stores hold different deletion mutants")
    worst = {"log2_ratio": 0.0, "expression": 0.0, "reference_expression": 0.0}
    for orf, records in new_by_orf.items():
        phenotypes = [r["experiment"]["phenotype"] for r in records]
        references = [r["reference"]["phenotype_reference"] for r in records]
        genes = phenotypes[0]["expression_log2_ratio"]
        old_record = old_by_orf[orf]
        for key, values, stored in (
            (
                "log2_ratio",
                [p["expression_log2_ratio"] for p in phenotypes],
                old_record["experiment"]["phenotype"]["expression_log2_ratio"],
            ),
            (
                "expression",
                [p["expression"] for p in phenotypes],
                old_record["experiment"]["phenotype"]["expression"],
            ),
            (
                "reference_expression",
                [p["expression"] for p in references],
                old_record["reference"]["phenotype_reference"]["expression"],
            ),
        ):
            mean = {g: float(np.mean([v[g] for v in values])) for g in genes}
            worst[key] = max(worst[key], max_abs_diff(mean, stored))
    result["mean_reproduction_spot_mean_vs_old_max_abs_diff"] = worst

    # last-spot reconstruction from the raw tables, and the orientation audit
    gsms: dict[str, Any] = {}
    probe_to_gene: dict[str, str] = {}
    for accession in (*DELETION_SERIES, *WT_SERIES_STRAIN):
        with open(osp.join(args.raw_dir, f"{accession}.pkl"), "rb") as handle:
            gse = pickle.load(handle)
        if not probe_to_gene:
            gpl = gse.gpls["GPL11232"].table
            probe_to_gene = {
                str(int(i)): str(o).upper()
                for i, o in zip(gpl["ID"], gpl["ORF"], strict=True)
                if str(o) not in ("nan", "")
            }
        gsms.update(gse.gsms)
    spots = Counter(probe_to_gene.values())
    result["platform_spots_per_gene"] = dict(Counter(spots.values()))

    last_worst = {"log2_ratio": 0.0, "expression": 0.0, "reference_expression": 0.0}
    by_orf_arrays = deletion_rows.groupby("systematic_gene_name")["geo_accession"]
    for group_orf, accessions in by_orf_arrays:
        per_array = [last_spot_values(gsms[a], probe_to_gene) for a in accessions]
        old_phenotype = old_by_orf[str(group_orf)]["experiment"]["phenotype"]
        old_reference = old_by_orf[str(group_orf)]["reference"]["phenotype_reference"]
        for index, key, stored in (
            (0, "log2_ratio", old_phenotype["expression_log2_ratio"]),
            (1, "expression", old_phenotype["expression"]),
            (2, "reference_expression", old_reference["expression"]),
        ):
            mean = {g: float(np.mean([a[g][index] for a in per_array])) for g in stored}
            last_worst[key] = max(last_worst[key], max_abs_diff(mean, stored))
    result["mean_reproduction_last_spot_vs_old_max_abs_diff"] = last_worst

    spot_gap = []
    for record, accession in zip(new, samples["geo_accession"], strict=True):
        last = last_spot_values(gsms[accession], probe_to_gene)
        stored = record["experiment"]["phenotype"]["expression_log2_ratio"]
        spot_gap.append(max(abs(stored[g] - last[g][0]) for g in stored))
    result["per_array_spot_mean_vs_last_spot_log2_max_abs_diff"] = {
        "max": float(np.max(spot_gap)),
        "median_of_per_array_max": float(np.median(spot_gap)),
    }

    orientation: Counter[str] = Counter()
    deleted_sign: Counter[str] = Counter()
    marker = {}
    mata = set(samples.loc[samples["strain"] == "BY4741", "geo_accession"])
    orf_of = dict(zip(samples["geo_accession"], samples["systematic_gene_name"]))
    for accession, series in zip(samples["geo_accession"], samples["series"]):
        gsm = gsms[accession]
        test_column, reference_column = MicroarrayKemmeren2014Dataset._channel_columns(
            gsm
        )
        table = gsm.table
        ratio = np.log2(
            table[test_column].astype(float) / table[reference_column].astype(float)
        )
        value = table["VALUE"].astype(float)
        kind = "deletion" if series in DELETION_SERIES else "wildtype"
        if float(np.max(np.abs(ratio - value))) < 1e-4:
            orientation[f"{kind}_value_equals_log2_test_over_reference"] += 1
        elif float(np.max(np.abs(ratio + value))) < 1e-4:
            orientation[f"{kind}_value_equals_log2_reference_over_test"] += 1
        else:
            orientation[f"{kind}_value_matches_neither"] += 1
        per_gene = (
            pd.DataFrame(
                {"gene": table["ID_REF"].astype(int).astype(str).map(probe_to_gene)}
            )
            .assign(m=ratio)
            .dropna()
            .groupby("gene")["m"]
            .mean()
        )
        orf = orf_of[accession]
        if orf and orf in per_gene.index:
            deleted_sign["negative" if per_gene[orf] < 0 else "not_negative"] += 1
        if accession in mata:
            marker[accession] = float(
                per_gene.reindex(list(A_GENES)).mean()
                - per_gene.reindex(list(ALPHA_GENES)).mean()
            )
    result["value_orientation"] = dict(orientation)
    result["deleted_gene_sign_on_deletion_arrays"] = dict(deleted_sign)
    result["mata_marker_contrast"] = {
        "n_arrays": len(marker),
        "min": min(marker.values()),
        "max": max(marker.values()),
    }

    out = osp.join(RESULTS, "kemmeren2014_per_array_records.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
