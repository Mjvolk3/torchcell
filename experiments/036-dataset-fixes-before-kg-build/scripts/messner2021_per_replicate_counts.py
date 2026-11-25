# experiments/036-dataset-fixes-before-kg-build/scripts/messner2021_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.messner2021_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/messner2021_per_replicate_counts
"""Prove the per-replicate ``ProteomeMessner2021Dataset`` counts against the release.

Reads two already-built stores, record by record with interned ``$ref`` pointers
resolved: ``--before-root`` (the loader of origin/main, 70 drug-run records with the 24
DMSO runs folded into the reference) and ``--after-root`` (the per-well loader, every kept
drug run and every DMSO run a record). Both are built into scratch roots, never the shared
dev tree. It reports:

- records per run role and per replicate block (``preprocess/runs.csv``; a block is one
  of the paper's four "blocked by replicates" sets of 24 wells) and the total;
- that the first 70 after-records are the before-records, value for value, once the two
  fields the change adds (``replicate_id``, ``provenance_gaps``) are set aside, and that
  every reference is unchanged;
- the mean-reproduction check: the after-records grouped by (genotype, environment) and
  averaged per protein, against what the before-store holds for that group: a drug group
  against the mean of its before-records, and the DMSO group against the before-store's
  shared reference summary. Max absolute and relative differences are stated.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/messner2021_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/messner2021_per_replicate_counts.py \
        --before-root <scratch>/before/proteome_messner2021 \
        --after-root <scratch>/after/proteome_messner2021
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


def _strip(phenotype: dict[str, Any]) -> dict[str, Any]:
    """A phenotype without the fields this change adds."""
    return {k: v for k, v in phenotype.items() if k not in ADDED_FIELDS}


def main() -> None:
    """Read both stores, compute the counts and the mean-reproduction check, write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()

    before = read_store(args.before_root)
    after = read_store(args.after_root)
    runs = pd.read_csv(osp.join(args.after_root, "preprocess", "runs.csv"))
    stored = runs[runs["record_index"].notna()]

    by_role = runs["role"].value_counts().to_dict()
    per_block = {
        str(role): {
            str(block): int(n)
            for block, n in sorted(Counter(group["block"].astype(int)).items())
        }
        for role, group in stored.groupby("role")
    }
    replicate_ids = [r["experiment"]["phenotype"]["replicate_id"] for r in after]
    index_to_run = dict(
        zip(stored["record_index"].astype(int), stored["run"], strict=True)
    )
    ids_are_run_names = all(replicate_ids[i] == run for i, run in index_to_run.items())

    # The drug records are the before-records, value for value.
    n_drug = len(before)
    drug_mismatch = 0
    reference_mismatch = 0
    for old, new in zip(before, after[:n_drug], strict=True):
        same = (
            _strip(new["experiment"]["phenotype"])
            == _strip(old["experiment"]["phenotype"])
            and new["experiment"]["genotype"] == old["experiment"]["genotype"]
            and new["experiment"]["environment"] == old["experiment"]["environment"]
        )
        drug_mismatch += not same
        reference_mismatch += new["reference"] != old["reference"]

    old_reference: dict[str, float] = {}
    for rec in before:
        old_reference.update(
            rec["reference"]["phenotype_reference"]["protein_abundance"]
        )
    dmso_references_equal_shared = all(
        rec["reference"]["phenotype_reference"]["protein_abundance"]
        == {
            k: old_reference[k]
            for k in rec["experiment"]["phenotype"]["protein_abundance"]
        }
        for rec in after[n_drug:]
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
    dmso_key = _key(after[n_drug]["experiment"])

    abs_max = rel_max = 0.0
    n_groups = 0
    for key, profiles in groups_after.items():
        if key == dmso_key:
            continue
        n_groups += 1
        a, r = _diff(_mean_profiles(profiles), _mean_profiles(groups_before[key]))
        abs_max, rel_max = max(abs_max, a), max(rel_max, r)
    dmso_mean = _mean_profiles(groups_after[dmso_key])
    # The before-reference covers the proteins some drug record measured; the DMSO mean
    # covers the proteins some DMSO run measured. Compare on the before-reference keys
    # and report any DMSO protein no drug record carried.
    only_dmso = sorted(set(dmso_mean) - set(old_reference))
    a_dmso, r_dmso = _diff({k: dmso_mean[k] for k in old_reference}, old_reference)

    summary = {
        "before_records": len(before),
        "after_records": len(after),
        "runs_by_role": {k: int(v) for k, v in by_role.items()},
        "records_by_role_and_block": per_block,
        "replicate_ids_distinct": len(set(replicate_ids)),
        "replicate_ids_are_run_names": ids_are_run_names,
        "group_multiplicity": {
            str(k): v
            for k, v in sorted(Counter(len(v) for v in groups_after.values()).items())
        },
        "drug_records_differing_from_before": drug_mismatch,
        "drug_references_differing_from_before": reference_mismatch,
        "dmso_references_equal_shared_before_reference": dmso_references_equal_shared,
        "mean_reproduction": {
            "drug_groups": {"groups": n_groups, "max_abs": abs_max, "max_rel": rel_max},
            "dmso_vs_before_reference": {
                "groups": 1,
                "records": len(groups_after[dmso_key]),
                "proteins_compared": len(old_reference),
                "proteins_only_in_dmso_records": only_dmso,
                "max_abs": a_dmso,
                "max_rel": r_dmso,
            },
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(
        osp.join(RESULTS, "messner2021_per_replicate_counts.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
