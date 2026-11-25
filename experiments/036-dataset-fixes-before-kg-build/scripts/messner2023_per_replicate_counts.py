# experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.messner2023_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_per_replicate_counts
"""Prove the per-replicate ``ProteomeMessner2023Dataset`` counts against the release.

Reads two already-built stores, record by record with interned ``$ref`` pointers
resolved: ``--before-root`` (the per-strain loader of origin/main, 4,699 KO records with
the 388 HIS3 controls folded into the reference) and ``--after-root`` (the per-sample
loader, KO and HIS3 samples each a record). Both are built into scratch roots, never the
shared dev tree. It reports:

- records per sample type (``preprocess/data.csv``) and the total, and how many HIS3
  records each of the 57 plates carries;
- that the first 4,699 after-records are the before-records, value for value, once the
  two fields the change adds (``replicate_id``, ``provenance_gaps``) are set aside;
- the mean-reproduction check: the after-records grouped by (genotype, environment) and
  averaged per protein, against what the before-store holds for that group: a one-strain
  KO group against its one record, a multi-strain KO group against the mean of its
  before-records, and the YOR202W (HIS3 control) group against the before-store's shared
  reference summary. Max absolute and relative differences are stated.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/messner2023_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_per_replicate_counts.py \
        --before-root <scratch>/before/proteome_messner2023 \
        --after-root <scratch>/after/proteome_messner2023
"""

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict
from typing import Any

import lmdb
import pandas as pd

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
ADDED_FIELDS = ("replicate_id", "provenance_gaps")


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


def _key(experiment: dict[str, Any]) -> str:
    """The (genotype, environment) group key of one experiment."""
    return json.dumps(
        [experiment["genotype"], experiment["environment"]], sort_keys=True
    )


def _mean_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    """Per-protein arithmetic mean over the profiles that measured the protein."""
    sums: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for profile in profiles:
        for protein, value in profile.items():
            sums[protein] += value
            counts[protein] += 1
    return {protein: sums[protein] / counts[protein] for protein in sums}


def _diff(got: dict[str, float], want: dict[str, float]) -> tuple[float, float]:
    """Max absolute and relative difference over identical key sets (raises if not)."""
    if set(got) != set(want):
        raise RuntimeError(
            f"protein sets differ: {len(set(got) ^ set(want))} keys in one only"
        )
    abs_max = max(abs(got[k] - want[k]) for k in got)
    rel_max = max(abs(got[k] - want[k]) / abs(want[k]) for k in got if want[k] != 0)
    return abs_max, rel_max


def main() -> None:
    """Read both stores, compute the counts and the mean-reproduction check, write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()

    before = read_store(args.before_root)
    after = read_store(args.after_root)
    summary_csv = pd.read_csv(osp.join(args.after_root, "preprocess", "data.csv"))
    meta = pd.read_csv(
        osp.join(args.after_root, "raw", "yeast5k_metadata.csv")
    ).set_index("Filename")

    by_type = summary_csv["sampletype"].value_counts().to_dict()
    his3_files = summary_csv.loc[summary_csv["sampletype"] == "HIS3", "filename"]
    per_plate = Counter(meta.loc[his3_files, "Plate (batch) nr"].tolist())
    replicate_ids = [r["experiment"]["phenotype"]["replicate_id"] for r in after]

    # The KO records are the before-records, value for value.
    n_ko = int(by_type["ko"])
    ko_mismatch = 0
    reference_mismatch = 0
    for old, new in zip(before, after[:n_ko], strict=True):
        stripped = {
            k: v
            for k, v in new["experiment"]["phenotype"].items()
            if k not in ADDED_FIELDS
        }
        old_ph = {
            k: v
            for k, v in old["experiment"]["phenotype"].items()
            if k not in ADDED_FIELDS
        }
        same = (
            stripped == old_ph
            and new["experiment"]["genotype"] == old["experiment"]["genotype"]
            and new["experiment"]["environment"] == old["experiment"]["environment"]
        )
        ko_mismatch += not same
        reference_mismatch += new["reference"] != old["reference"]

    # Shared reference summary of the before-store, over the proteins any record holds.
    old_reference: dict[str, float] = {}
    for rec in before:
        old_reference.update(
            rec["reference"]["phenotype_reference"]["protein_abundance"]
        )

    groups_after: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rec in after:
        groups_after[_key(rec["experiment"])].append(
            rec["experiment"]["phenotype"]["protein_abundance"]
        )
    groups_before: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rec in before:
        groups_before[_key(rec["experiment"])].append(
            rec["experiment"]["phenotype"]["protein_abundance"]
        )
    his3_key = _key(after[n_ko]["experiment"])

    checks: dict[str, dict[str, float | int]] = {}
    for label, multi_strain in (("ko_one_strain", False), ("ko_multi_strain", True)):
        abs_max = rel_max = 0.0
        n_groups = 0
        for key, profiles in groups_after.items():
            if key == his3_key or (len(profiles) > 1) != multi_strain:
                continue
            n_groups += 1
            a, r = _diff(_mean_profiles(profiles), _mean_profiles(groups_before[key]))
            abs_max, rel_max = max(abs_max, a), max(rel_max, r)
        checks[label] = {"groups": n_groups, "max_abs": abs_max, "max_rel": rel_max}
    his3_mean = _mean_profiles(groups_after[his3_key])
    a, r = _diff(his3_mean, old_reference)
    checks["his3_vs_before_reference"] = {
        "groups": 1,
        "records": len(groups_after[his3_key]),
        "proteins": len(his3_mean),
        "max_abs": a,
        "max_rel": r,
    }

    multiplicity = Counter(len(v) for k, v in groups_after.items() if k != his3_key)
    summary = {
        "before_records": len(before),
        "after_records": len(after),
        "after_by_sampletype": {k: int(v) for k, v in by_type.items()},
        "his3_records_per_plate": {
            str(n): c for n, c in sorted(Counter(per_plate.values()).items())
        },
        "his3_plates": len(per_plate),
        "replicate_ids_distinct": len(set(replicate_ids)),
        "ko_genotype_groups": sum(multiplicity.values()),
        "ko_group_multiplicity": {str(k): v for k, v in sorted(multiplicity.items())},
        "ko_records_differing_from_before": ko_mismatch,
        "ko_references_differing_from_before": reference_mismatch,
        "mean_reproduction": checks,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(
        osp.join(RESULTS, "messner2023_per_replicate_counts.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
