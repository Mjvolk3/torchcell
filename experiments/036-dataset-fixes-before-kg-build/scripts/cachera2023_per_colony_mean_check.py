# experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_per_colony_mean_check.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_per_colony_mean_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_per_colony_mean_check.py
"""Check the Cachera 2023 per-colony store (#1177) against the old per-gene store.

Reads, all read-only:

- the OLD dev-tree LMDB ``$DATA_ROOT/data/torchcell/betaxanthin_cachera2023`` (one record
  per gene: the ``GA1_2_4_6.csv`` mean, std / sqrt(count) and count);
- the NEW per-colony build under ``--new-root`` (one record per colony, the 12 WT
  control colonies included).

Groups the new records by genotype and environment (the deleted gene token, or none for
a control colony; the environment is one for the whole store), averages each group, and
compares with the old record of that group: the largest absolute mean difference, the
largest difference between the recomputed SD / sqrt(n) and the old SE, and whether each
group's record count equals the old ``n_replicates``. The control group, which had no
old record, is compared with the new reference (the authors' WT row). Also counts
records per screen repeat and colonies per group. Writes
``results/cachera2023_per_colony_mean_check.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_per_colony_mean_check.py \
        --new-root <scratch>/per-replicate-cachera2023/betaxanthin_cachera2023
"""

import argparse
import json
import math
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Any

from dotenv import load_dotenv

from torchcell.verification.runners import load_records

LABEL = "betaxanthin"


def _group_key(experiment: dict[str, Any]) -> tuple[str, str]:
    """(deleted gene token or '', environment JSON): one strain in one environment."""
    deleted = [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p["perturbation_type"] != "gene_addition"
    ]
    if len(deleted) > 1:
        raise ValueError(f"a Cachera record deletes {deleted}")
    return (deleted[0] if deleted else ""), json.dumps(
        experiment["environment"], sort_keys=True
    )


def _sd_over_sqrt_n(cell: list[float]) -> float:
    n = len(cell)
    mean = sum(cell) / n
    return math.sqrt(sum((v - mean) ** 2 for v in cell) / (n - 1)) / math.sqrt(n)


def check(old_root: str, new_root: str) -> dict[str, Any]:
    """Compare the new per-colony records with the old per-gene records."""
    old = load_records(old_root)
    new = load_records(new_root)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_repeat: Counter[str] = Counter()
    for rec in new:
        phenotype = rec["experiment"]["phenotype"]
        values[_group_key(rec["experiment"])].append(
            float(phenotype["metabolite_level"][LABEL])
        )
        per_repeat[str(phenotype["replicate_id"]).split("/")[0]] += 1
    max_mean = 0.0
    max_se = 0.0
    n_count_mismatch = 0
    n_unmatched = 0
    matched: set[tuple[str, str]] = set()
    for rec in old:
        phenotype = rec["experiment"]["phenotype"]
        key = _group_key(rec["experiment"])
        if key not in values:
            n_unmatched += 1
            continue
        matched.add(key)
        cell = values[key]
        max_mean = max(
            max_mean, abs(sum(cell) / len(cell) - phenotype["metabolite_level"][LABEL])
        )
        n_count_mismatch += int(len(cell) != phenotype["n_replicates"][LABEL])
        se = (phenotype.get("metabolite_level_se") or {}).get(LABEL)
        if len(cell) > 1 and se is not None and not math.isnan(float(se)):
            max_se = max(max_se, abs(_sd_over_sqrt_n(cell) - float(se)))
    control = values[("", next(iter(values))[1])]
    reference = new[0]["reference"]["phenotype_reference"]
    sizes = Counter(len(cell) for cell in values.values())
    return {
        "old_records": len(old),
        "new_records": len(new),
        "new_records_per_repeat": dict(sorted(per_repeat.items())),
        "new_groups": len(values),
        "new_control_records": len(control),
        "old_records_without_a_matching_group": n_unmatched,
        "new_groups_without_an_old_record": sorted(
            k[0] or "(control)" for k in set(values) - matched
        ),
        "max_abs_mean_difference": max_mean,
        "max_abs_se_difference": max_se,
        "groups_whose_record_count_differs_from_old_n": n_count_mismatch,
        "control_mean_minus_reference_level": sum(control) / len(control)
        - reference["metabolite_level"][LABEL],
        "control_se_minus_reference_se": _sd_over_sqrt_n(control)
        - reference["metabolite_level_se"][LABEL],
        "reference_n": reference["n_replicates"][LABEL],
        "records_per_group": {str(k): v for k, v in sorted(sizes.items())},
    }


def main() -> None:
    """Run the check and write the JSON result."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True)
    args = parser.parse_args()
    old_root = osp.join(
        os.environ["DATA_ROOT"], "data", "torchcell", "betaxanthin_cachera2023"
    )
    result = check(old_root, args.new_root)
    out = osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))),
        "results",
        "cachera2023_per_colony_mean_check.json",
    )
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
