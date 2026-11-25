# experiments/036-dataset-fixes-before-kg-build/scripts/schmidt2016_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.schmidt2016_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/schmidt2016_per_replicate_counts
"""Prove the per-culture Schmidt 2016 counts against the release and the previous stores.

Reads already-built stores in scratch roots, never the shared dev tree. ``--before-root``
holds origin/main's ``growth_rate_schmidt2016`` (6 ``FitnessExperiment`` ratios over a
loader-computed wild-type mean) and ``proteome_schmidt2016`` (21 copies/cell records with
a loader-computed SE); ``--after-root`` holds the converted growth store (22 culture
rates), the converted copies/cell store (21 records, no SE) and the new
``proteome_replicate_intensity_schmidt2016`` (66 culture records). It reports:

- records per replicate position and per kind, against ``si2.xlsx`` read directly;
- growth: the after-records grouped by (deleted gene, medium) and averaged, divided by
  the mean of that medium's wild-type records, against the before-store's fitness, and
  the wild-type group's mean against each reference's released Average;
- copies/cell: every before-record's abundances against the after-record of the same
  environment (identical values expected; only the SE map left);
- replicate intensity: per condition, the median over the three culture records against
  the released ``medianNormInt`` of Table S8 (the authors' aggregate), and the
  reference against ``medianNormInt_Glucose.1``.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/schmidt2016_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/schmidt2016_per_replicate_counts.py \
        --before-root <scratch>/before --after-root <scratch>/after
"""

import argparse
import json
import os
import os.path as osp
import statistics
from collections import Counter, defaultdict
from typing import Any

import openpyxl
from dotenv import load_dotenv

import torchcell.datasets.ecoli.schmidt2016 as sm
import torchcell.datasets.ecoli.schmidt2016_growth_rate as gr
import torchcell.datasets.ecoli.schmidt2016_replicate_intensity as ri
from torchcell.verification.runners import load_records

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
GROWTH = "growth_rate_schmidt2016"
COPIES = "proteome_schmidt2016"
INTENSITY = "proteome_replicate_intensity_schmidt2016"


def genes(record: dict[str, Any]) -> tuple[str, ...]:
    """The deleted genes of a record (empty for a wild-type culture)."""
    return tuple(
        sorted(
            p["systematic_gene_name"]
            for p in record["experiment"]["genotype"]["perturbations"]
        )
    )


def environment(record: dict[str, Any]) -> str:
    """A record's environment, serialized, the grouping key beside the genotype."""
    return json.dumps(record["experiment"]["environment"], sort_keys=True)


def growth_report(before_root: str, after_root: str, workbook: str) -> dict[str, Any]:
    """Counts per replicate position and the mean reproduction of the six ratios."""
    blocks = gr.read_table_s24(workbook)
    before = load_records(osp.join(before_root, GROWTH))
    after = load_records(osp.join(after_root, GROWTH))
    positions = Counter(r["experiment"]["phenotype"]["replicate_id"] for r in after)
    kinds = Counter("wild_type" if not genes(r) else "deletion" for r in after)
    groups: dict[tuple[tuple[str, ...], str], list[float]] = defaultdict(list)
    for record in after:
        groups[(genes(record), environment(record))].append(
            record["experiment"]["phenotype"]["environment_response"]
        )
    wild_type = {env: values for (g, env), values in groups.items() if not g}
    worst = 0.0
    compared = []
    for record in before:
        key = (genes(record), environment(record))
        mean = statistics.fmean(groups[key]) / statistics.fmean(wild_type[key[1]])
        diff = abs(mean - record["experiment"]["phenotype"]["fitness"])
        worst = max(worst, diff)
        compared.append(
            {
                "genes": list(key[0]),
                "before_fitness": record["experiment"]["phenotype"]["fitness"],
                "after_mean_over_wild_type_mean": mean,
                "n_cultures": len(groups[key]),
            }
        )
    reference_worst = 0.0
    for record in after:
        reference = record["reference"]["phenotype_reference"]
        mean = statistics.fmean(wild_type[environment(record)])
        reference_worst = max(
            reference_worst, abs(mean - reference["environment_response"])
        )
    released = {
        f"{block.medium}/{row.strain}": list(row.replicate_labels)
        for block in blocks
        for row in block.rows
    }
    return {
        "records_before": len(before),
        "records_after": len(after),
        "released_replicate_cells": sum(len(v) for v in released.values()),
        "released_cells_by_row": released,
        "records_by_replicate_position": dict(sorted(positions.items())),
        "records_by_kind": dict(kinds),
        "mean_reproduction": {
            "rule": "mean(after records of a (gene, medium)) / mean(after wild-type "
            "records of that medium) against the before-store fitness",
            "n_groups_compared": len(compared),
            "max_abs_diff": worst,
            "groups": compared,
        },
        "reference_is_mean_of_wild_type_records_max_abs_diff": reference_worst,
    }


def copies_report(before_root: str, after_root: str) -> dict[str, Any]:
    """Every copies/cell value is unchanged; only the loader-computed SE left."""
    before = {environment(r): r for r in load_records(osp.join(before_root, COPIES))}
    after = {environment(r): r for r in load_records(osp.join(after_root, COPIES))}
    worst = 0.0
    values = 0
    for env, record in before.items():
        old = record["experiment"]["phenotype"]["protein_abundance"]
        new = after[env]["experiment"]["phenotype"]["protein_abundance"]
        if set(old) != set(new):
            raise RuntimeError("a copies/cell record changed its protein keys")
        for key, value in old.items():
            values += 1
            worst = max(worst, abs(value - new[key]))
    se_before = sum(
        1
        for r in before.values()
        if r["experiment"]["phenotype"].get("protein_abundance_se") is not None
    )
    se_after = sum(
        1
        for r in after.values()
        if r["experiment"]["phenotype"].get("protein_abundance_se") is not None
    )
    return {
        "records_before": len(before),
        "records_after": len(after),
        "values_compared": values,
        "max_abs_diff": worst,
        "records_with_se_map_before": se_before,
        "records_with_se_map_after": se_after,
    }


def released_medians(workbook: str) -> dict[str, dict[str, float]]:
    """``{Table S8 accession: {condition column: released medianNormInt}}``."""
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = list(book[sm.SHEET_S8].iter_rows(values_only=True))
    finally:
        book.close()
    index = sm._header_index(rows[sm.HEADER_ROW - 1])
    out: dict[str, dict[str, float]] = {}
    for row in rows[sm.HEADER_ROW :]:
        out.setdefault(str(row[0]).strip(), {}).update(
            {
                spec.s6_column: float(row[index[f"medianNormInt_{spec.s8_suffix}"]])
                for spec in sm.CONDITIONS
            }
        )
    return out


def intensity_report(after_root: str, workbook: str) -> dict[str, Any]:
    """Per condition, the median over the culture records is the released median."""
    root = osp.join(after_root, INTENSITY)
    records = load_records(root)
    identifiers = {}
    with open(osp.join(root, "preprocess", "protein_identifiers.csv")) as handle:
        next(handle)
        for line in handle:
            uniprot, _, _, key = line.rstrip("\n").split(",", 3)
            identifiers[key] = uniprot
    medians = released_medians(workbook)
    counts = Counter(identifiers.values())
    repeated = {key for key, uniprot in identifiers.items() if counts[uniprot] > 1}
    condition_of = {
        name: column for column, names in ri.condition_files().items() for name in names
    }
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[
            condition_of[record["experiment"]["phenotype"]["replicate_id"]]
        ].append(record)
    worst = 0.0
    compared = 0
    for column, group in by_condition.items():
        for key, uniprot in identifiers.items():
            if key in repeated:
                continue
            median = statistics.median(
                r["experiment"]["phenotype"]["protein_abundance"][key] for r in group
            )
            compared += 1
            worst = max(worst, abs(median - medians[uniprot][column]))
    reference_worst = 0.0
    reference = records[0]["reference"]["phenotype_reference"]["protein_abundance"]
    for key, uniprot in identifiers.items():
        if key in repeated:
            continue
        reference_worst = max(
            reference_worst,
            abs(reference[key] - medians[uniprot][sm.REFERENCE_CONDITION]),
        )
    return {
        "records": len(records),
        "records_by_condition": {c: len(g) for c, g in by_condition.items()},
        "replicate_ids": sorted(
            r["experiment"]["phenotype"]["replicate_id"] for r in records
        ),
        "protein_keys_per_record": sorted(
            {len(r["experiment"]["phenotype"]["protein_abundance"]) for r in records}
        ),
        "median_reproduction": {
            "rule": "median over a condition's three culture records against the "
            "released Table S8 medianNormInt (the two P63284 rows skipped: Table S8 "
            "files that accession twice)",
            "n_values_compared": compared,
            "max_abs_diff": worst,
        },
        "reference_vs_released_glucose_median_max_abs_diff": reference_worst,
    }


def main() -> None:
    """Build the report from the scratch stores and write it."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()
    workbook = osp.join(os.environ["DATA_ROOT"], sm.RAW_DIR_REL, sm.SI2_MIRROR_RELPATH)
    report = {
        "source": {
            "workbook": f"$DATA_ROOT/{sm.RAW_DIR_REL}/{sm.SI2_MIRROR_RELPATH}",
            "sha256": sm.SI2_SHA256,
        },
        "growth_rate": growth_report(args.before_root, args.after_root, workbook),
        "copies_per_cell": copies_report(args.before_root, args.after_root),
        "replicate_intensity": intensity_report(args.after_root, workbook),
        "expected": {
            "growth_rate": gr.EXPECTED_RECORDS,
            "replicate_intensity": ri.EXPECTED_RECORDS,
            "copies_per_cell": sm.EXPECTED_RECORDS,
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "schmidt2016_per_replicate_counts.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "source"},
            indent=2,
            ensure_ascii=False,
        )[:4000]
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
