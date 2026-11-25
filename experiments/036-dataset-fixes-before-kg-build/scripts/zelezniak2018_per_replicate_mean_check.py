# experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_per_replicate_mean_check.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.zelezniak2018_per_replicate_mean_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_per_replicate_mean_check.py
"""Check the Zelezniak 2018 per-replicate stores (#1173, #1174) against the old means.

Reads, all read-only:

- the OLD dev-tree LMDBs ``$DATA_ROOT/data/torchcell/{proteome,metabolite}_zelezniak2018``
  (one mean + SE record per strain, or per (strain, protocol));
- the NEW per-replicate builds under ``--new-parent`` (one record per released replicate,
  WT included).

For each modality, groups the new records by (deleted genes, measurement_type), averages
each key over its records, and compares with the old record of that group: the largest
absolute mean difference, the largest difference between the recomputed SD / sqrt(n) and
the old SE, and that each group's record count equals the old ``n_replicates``. Also
counts records, WT records and records per group. Writes
``results/zelezniak2018_per_replicate_mean_check.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/zelezniak2018_per_replicate_mean_check.py \
        --new-parent <scratch>/per-replicate-zelezniak2018
"""

import argparse
import json
import math
import os
import os.path as osp
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

from torchcell.verification.runners import load_records

MODALITIES = {
    "proteome_zelezniak2018": ("protein_abundance", "protein_abundance_se"),
    "metabolite_zelezniak2018": ("metabolite_level", "metabolite_level_se"),
}


def _group_key(experiment: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    """(sorted deleted genes, measurement_type): one strain under one protocol."""
    genes = tuple(
        sorted(
            p["systematic_gene_name"] for p in experiment["genotype"]["perturbations"]
        )
    )
    return genes, str(experiment["phenotype"]["measurement_type"])


def check(old_root: str, new_root: str, label: str, se_key: str) -> dict[str, Any]:
    """Compare one modality's new per-replicate records with its old aggregates."""
    old = load_records(old_root)
    new = load_records(new_root)
    values: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records_per_group: dict[tuple[Any, ...], int] = defaultdict(int)
    for rec in new:
        key = _group_key(rec["experiment"])
        records_per_group[key] += 1
        for name, value in rec["experiment"]["phenotype"][label].items():
            values[key][name].append(float(value))
    max_mean = 0.0
    max_se = 0.0
    n_values = 0
    n_count_mismatch = 0
    n_unmatched = 0
    for rec in old:
        phenotype = rec["experiment"]["phenotype"]
        key = _group_key(rec["experiment"])
        if key not in values or set(values[key]) != set(phenotype[label]):
            n_unmatched += 1
            continue
        old_se = phenotype.get(se_key) or {}
        for name, stored in phenotype[label].items():
            cell = values[key][name]
            n = len(cell)
            mean = sum(cell) / n
            n_values += 1
            max_mean = max(max_mean, abs(mean - float(stored)))
            n_count_mismatch += int(n != int(phenotype["n_replicates"][name]))
            se = old_se.get(name)
            if n > 1 and se is not None and not math.isnan(float(se)):
                sd = math.sqrt(sum((v - mean) ** 2 for v in cell) / (n - 1))
                max_se = max(max_se, abs(sd / math.sqrt(n) - float(se)))
    group_sizes: dict[int, int] = defaultdict(int)
    for size in records_per_group.values():
        group_sizes[size] += 1
    return {
        "old_records": len(old),
        "new_records": len(new),
        "new_wt_records": sum(
            1 for rec in new if not rec["experiment"]["genotype"]["perturbations"]
        ),
        "old_records_without_a_matching_group": n_unmatched,
        "values_compared": n_values,
        "max_abs_mean_difference": max_mean,
        "max_abs_se_difference": max_se,
        "cells_whose_record_count_differs_from_old_n": n_count_mismatch,
        "records_per_group": {str(k): v for k, v in sorted(group_sizes.items())},
    }


def main() -> None:
    """Run the check for both modalities and write the JSON result."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-parent", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    result = {
        name: check(
            osp.join(data_root, "data", "torchcell", name),
            osp.join(args.new_parent, name),
            label,
            se_key,
        )
        for name, (label, se_key) in MODALITIES.items()
    }
    out = osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))),
        "results",
        "zelezniak2018_per_replicate_mean_check.json",
    )
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
