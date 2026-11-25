# experiments/036-dataset-fixes-before-kg-build/scripts/campos2018_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.campos2018_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/campos2018_per_replicate_counts
"""Prove the per-well Campos 2018 counts against the release and the previous stores.

Reads four already-built stores, all in scratch roots, never the shared dev tree:
``--before-root`` holds the two stores of origin/main (one record per writable deletion
label: a ``FitnessExperiment`` ratio over the wild-type median and a
``MorphologyExperiment``, 3,664 each, the 240 wild-type wells and the 120 rows of
repeated labels dropped) and ``--after-root`` the two per-well stores (every released
well a record, 4,024 each). It reports:

- records per kind (deletion wells heading one row, rows of repeated labels, wild-type
  wells per released label) and the total, against Dataset EV2 read directly;
- that every before-record has exactly one after-record at the same Keio position whose
  value reproduces it: for growth, ``alpha_max / reference`` against the stored ratio;
  for morphology, every feature value;
- the mean-reproduction check: the after-records grouped by (deleted locus set,
  environment) and averaged, against the before-store's value for that group, plus the
  wild-type group against the before-store's reference. Max absolute differences are
  stated; a group the before-store never held (a repeated label) is counted, not
  compared.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/campos2018_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/campos2018_per_replicate_counts.py \
        --before-root <scratch>/before --after-root <scratch>/after
"""

import argparse
import json
import os
import os.path as osp
import statistics
from collections import Counter, defaultdict
from typing import Any

from dotenv import load_dotenv

import torchcell.datasets.ecoli.campos2018 as c
from torchcell.verification.runners import load_records

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
GROWTH = "ecoli_growth_rate_campos2018"
MORPHOLOGY = "ecoli_morphology_campos2018"


def position(record: dict[str, Any]) -> str:
    """The Keio position of a record: its construction for a deletion, else replicate_id."""
    perturbations = record["experiment"]["genotype"]["perturbations"]
    if perturbations:
        construction = perturbations[0]["construction"]
        return f"{construction['plate']}/{construction['well']}"
    return str(record["experiment"]["phenotype"]["replicate_id"])


def strain(record: dict[str, Any]) -> tuple[str, ...]:
    """The deleted loci of a record, the grouping key (empty for a wild-type well)."""
    return tuple(
        sorted(
            p["systematic_gene_name"]
            for p in record["experiment"]["genotype"]["perturbations"]
        )
    )


def features(phenotype: dict[str, Any]) -> dict[str, float]:
    """Every morphology feature of a phenotype, values and CVs in one mapping."""
    return {
        **phenotype["morphology"],
        **(phenotype["morphology_coefficient_of_variation"] or {}),
    }


def growth_check(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-position and per-group reproduction of the old ratio store."""
    reference = after[0]["reference"]["phenotype_reference"]["environment_response"]
    by_position = {position(r): r for r in after}
    assert len(by_position) == len(after), "two after-records share a position"
    worst_position = 0.0
    for record in before:
        match = by_position[position(record)]
        assert strain(match) == strain(record)
        ratio = match["experiment"]["phenotype"]["environment_response"] / reference
        worst_position = max(
            worst_position, abs(ratio - record["experiment"]["phenotype"]["fitness"])
        )
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for record in after:
        groups[strain(record)].append(
            record["experiment"]["phenotype"]["environment_response"]
        )
    old = {strain(r): r["experiment"]["phenotype"]["fitness"] for r in before}
    worst_group = 0.0
    compared = 0
    for key, values in groups.items():
        if key in old:
            compared += 1
            worst_group = max(
                worst_group, abs(statistics.fmean(values) / reference - old[key])
            )
    wild_type = groups[()]
    return {
        "reference_alpha_max_min-1": reference,
        "before_reference_fitness": before[0]["reference"]["phenotype_reference"][
            "fitness"
        ],
        "per_position_max_abs_diff_ratio": worst_position,
        "groups_after": len(groups),
        "groups_compared_with_before": compared,
        "groups_new_repeated_labels": sum(
            1 for key, values in groups.items() if key and len(values) > 1
        ),
        "group_mean_max_abs_diff_ratio": worst_group,
        "wild_type_wells": len(wild_type),
        "wild_type_median_over_reference": statistics.median(wild_type) / reference,
        "wild_type_mean_over_reference": statistics.fmean(wild_type) / reference,
    }


def morphology_check(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-position feature reproduction, and the wild-type median against the reference."""
    by_position = {position(r): r for r in after}
    assert len(by_position) == len(after), "two after-records share a position"
    worst = 0.0
    n_values = 0
    for record in before:
        match = by_position[position(record)]
        old = features(record["experiment"]["phenotype"])
        new = features(match["experiment"]["phenotype"])
        assert set(old) == set(new)
        assert (
            match["experiment"]["phenotype"]["n_samples"]
            == record["experiment"]["phenotype"]["n_samples"]
        )
        for symbol, value in old.items():
            worst = max(worst, abs(new[symbol] - value))
            n_values += 1
    old_reference = features(before[0]["reference"]["phenotype_reference"])
    new_reference = features(after[0]["reference"]["phenotype_reference"])
    wild_type = [r for r in after if not strain(r)]
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for record in wild_type:
        for symbol, value in features(record["experiment"]["phenotype"]).items():
            by_symbol[symbol].append(value)
    median_vs_reference = max(
        abs(statistics.median(values) - old_reference[symbol])
        for symbol, values in by_symbol.items()
    )
    mean_vs_reference = max(
        abs(statistics.fmean(values) - old_reference[symbol])
        for symbol, values in by_symbol.items()
    )
    groups: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for record in after:
        groups[strain(record)].append(features(record["experiment"]["phenotype"]))
    old_by_strain = {strain(r): features(r["experiment"]["phenotype"]) for r in before}
    worst_group = 0.0
    for key, profiles in groups.items():
        if key in old_by_strain:
            for symbol, value in old_by_strain[key].items():
                mean = statistics.fmean(p[symbol] for p in profiles)
                worst_group = max(worst_group, abs(mean - value))
    return {
        "per_position_values_compared": n_values,
        "per_position_max_abs_diff": worst,
        "group_mean_max_abs_diff": worst_group,
        "reference_identical": old_reference == new_reference,
        "reference_n_samples": after[0]["reference"]["phenotype_reference"][
            "n_samples"
        ],
        "wild_type_median_vs_reference_max_abs_diff": median_vs_reference,
        "wild_type_mean_vs_reference_max_abs_diff": mean_vs_reference,
        "determined_values_after": sum(
            len(features(r["experiment"]["phenotype"])) for r in after
        ),
    }


def release_counts(data_root: str) -> dict[str, Any]:
    """What Dataset EV2 itself holds, read directly from the raw mirror."""
    table = c.read_normalized_table(c.raw_mirror_dir(data_root) / c.DATA_REL)
    labels = Counter(row.label for row in table.mutants)
    return {
        "released_rows": table.released_rows,
        "footer_rows": table.footer_rows,
        "wild_type_wells_by_label": dict(
            sorted(Counter(row.label for row in table.wild_type).items())
        ),
        "wild_type_plates": sorted({row.plate for row in table.wild_type}),
        "mutant_rows": len(table.mutants),
        "distinct_mutant_labels": len(labels),
        "repeated_label_histogram": dict(
            sorted(Counter(n for n in labels.values() if n > 1).items())
        ),
    }


def kinds(after: list[dict[str, Any]]) -> dict[str, int]:
    """After-records per kind: wild type, a locus heading one row, a repeated locus."""
    per_locus = Counter(strain(r) for r in after if strain(r))
    return {
        "wild_type_wells": sum(1 for r in after if not strain(r)),
        "deletion_wells_single": sum(1 for n in per_locus.values() if n == 1),
        "deletion_wells_repeated": sum(n for n in per_locus.values() if n > 1),
        "repeated_loci": sum(1 for n in per_locus.values() if n > 1),
        "total": len(after),
    }


def main() -> None:
    """Read the four stores, run the checks, write the JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    growth_before = load_records(osp.join(args.before_root, GROWTH))
    growth_after = load_records(osp.join(args.after_root, GROWTH))
    morph_before = load_records(osp.join(args.before_root, MORPHOLOGY))
    morph_after = load_records(osp.join(args.after_root, MORPHOLOGY))
    result = {
        "release": release_counts(data_root),
        "growth": {
            "records_before": len(growth_before),
            "records_after": len(growth_after),
            "kinds_after": kinds(growth_after),
            **growth_check(growth_before, growth_after),
        },
        "morphology": {
            "records_before": len(morph_before),
            "records_after": len(morph_after),
            "kinds_after": kinds(morph_after),
            **morphology_check(morph_before, morph_after),
        },
        "same_positions_in_both_stores": sorted(map(position, growth_after))
        == sorted(map(position, morph_after)),
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "campos2018_per_replicate_counts.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
