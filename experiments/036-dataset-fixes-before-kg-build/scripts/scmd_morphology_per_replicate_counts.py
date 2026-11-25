# experiments/036-dataset-fixes-before-kg-build/scripts/scmd_morphology_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.scmd_morphology_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/scmd_morphology_per_replicate_counts
"""Prove the per-replicate SCMD CalMorph counts against the three releases.

For each of ``ScmdOhya2005Dataset``, ``ScmdOhnuki2018Dataset`` and
``ScmdOhnuki2022Dataset`` it reads two already-built stores, record by record with
interned ``$ref`` pointers resolved: ``<before-base>/<root name>`` (the per-strain loader
of origin/main, the wild-type rows folded into the reference) and
``<after-base>/<root name>`` (the per-sample loader, every wild-type row a record). Both
are built into scratch roots, never the shared dev tree. Per dataset it reports:

- the record count before and after, the mutant and wild-type record counts, and the
  wild-type count per released ``NAME`` pattern (digits collapsed to ``#``);
- that the first N after-records are the N before-records exactly (whole dicts);
- that the wild-type records are the wild-type file's rows in release order: each
  ``replicate_id`` is the row's ``NAME`` and each value is the released cell, exactly;
- the mean-reproduction check: the after-records grouped by (genotype, environment,
  replicate_id is None) and averaged per trait. A mutant group is one record, compared
  with its before-record; the wild-type group is compared with the before-store's shared
  reference. Max absolute difference stated;
- that every after-record carries the before-store's reference exactly.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/scmd_morphology_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/scmd_morphology_per_replicate_counts.py \
        --before-base <scratch>/before --after-base <scratch>
"""

import argparse
import json
import os
import os.path as osp
import pickle
import re
from collections import Counter, defaultdict
from typing import Any

import lmdb
import pandas as pd

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"

# root name -> (wild-type file in raw/, expected mutant count, expected wild-type count)
DATASETS: dict[str, tuple[str, int, int]] = {
    "scmd_ohya2005": ("wt122data.tsv", 4718, 122),
    "scmd_ohnuki2018": ("wt114data.tsv", 1112, 114),
    "scmd_ohnuki2022": ("wt749data.tsv", 1982, 749),
}


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
        records = []
        for i in range(n):
            raw = txn.get(str(i).encode())
            assert raw is not None, f"{root}: record {i} missing"
            records.append(resolve_interned(pickle.loads(raw), interned))
    env.close()
    return records


def _traits(phenotype: dict[str, Any]) -> dict[str, float]:
    """Base and CV traits of one CalMorph phenotype, merged."""
    return {
        **phenotype["calmorph"],
        **(phenotype.get("calmorph_coefficient_of_variation") or {}),
    }


def _group_key(experiment: dict[str, Any]) -> str:
    """The (genotype, environment, replicate-or-not) group key of one experiment."""
    return json.dumps(
        [
            experiment["genotype"],
            experiment["environment"],
            experiment["phenotype"].get("replicate_id") is None,
        ],
        sort_keys=True,
    )


def check(
    before_root: str, after_root: str, spec: tuple[str, int, int]
) -> dict[str, Any]:
    """Run every count and value check for one dataset; raise on a broken invariant."""
    wt_file, n_mutant, n_wt = spec
    before = read_store(before_root)
    after = read_store(after_root)
    wt = pd.read_csv(osp.join(after_root, "raw", wt_file), sep="\t")

    assert len(before) == n_mutant, (before_root, len(before))
    assert len(after) == n_mutant + n_wt, (after_root, len(after))
    assert len(wt) == n_wt

    mutant_identical = all(
        a == b for a, b in zip(after[:n_mutant], before, strict=True)
    )
    assert mutant_identical, "a mutant record changed"

    wt_records = after[n_mutant:]
    replicate_ids = [r["experiment"]["phenotype"]["replicate_id"] for r in wt_records]
    assert replicate_ids == wt["NAME"].astype(str).tolist()
    assert all(
        r["experiment"]["phenotype"].get("replicate_id") is None
        for r in after[:n_mutant]
    )
    features = [c for c in wt.columns if c != "NAME"]
    value_mismatch = 0
    for record, (_, row) in zip(wt_records, wt.iterrows(), strict=True):
        released = {c: float(row[c]) for c in features if pd.notna(row[c])}
        if _traits(record["experiment"]["phenotype"]) != released:
            value_mismatch += 1
    assert value_mismatch == 0

    reference_identical = all(r["reference"] == before[0]["reference"] for r in after)
    assert reference_identical, "a reference changed"

    # Mean reproduction: group the after-records and average per trait.
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in after:
        groups[_group_key(record["experiment"])].append(
            _traits(record["experiment"]["phenotype"])
        )
    before_by_key = {
        _group_key(r["experiment"]): _traits(r["experiment"]["phenotype"])
        for r in before
    }
    old_reference = _traits(before[0]["reference"]["phenotype_reference"])
    max_mutant = 0.0
    max_wt = 0.0
    max_wt_rel = 0.0
    wt_groups = 0
    for key, profiles in groups.items():
        sums: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for profile in profiles:
            for trait, value in profile.items():
                sums[trait] += value
                counts[trait] += 1
        mean = {t: sums[t] / counts[t] for t in sums}
        if json.loads(key)[2] is False:  # the wild-type replicate group
            wt_groups += 1
            assert len(profiles) == n_wt and set(mean) == set(old_reference)
            max_wt = max(abs(mean[t] - old_reference[t]) for t in mean)
            max_wt_rel = max(
                abs(mean[t] - old_reference[t]) / abs(old_reference[t])
                for t in mean
                if old_reference[t] != 0
            )
        else:
            assert len(profiles) == 1
            old = before_by_key[key]
            assert set(mean) == set(old)
            max_mutant = max([max_mutant, *(abs(mean[t] - old[t]) for t in mean)])
    assert wt_groups == 1

    patterns = Counter(re.sub(r"[0-9]+", "#", str(n)) for n in wt["NAME"])
    # Ohnuki 2022 names are Q<yymmdd><well>: the acquisition dates the paper's "over 152
    # days" spans (one record per released row on each date).
    dates = Counter(
        m.group(1) for n in wt["NAME"] if (m := re.match(r"^Q(\d{6})", str(n)))
    )
    return {
        "records_before": len(before),
        "records_after": len(after),
        "mutant_records": n_mutant,
        "wild_type_records": n_wt,
        "wild_type_records_per_name_pattern": dict(patterns.most_common()),
        "first_n_after_records_equal_before": mutant_identical,
        "wild_type_values_equal_released_cells": value_mismatch == 0,
        "every_reference_equal_before": reference_identical,
        "mean_reproduction_max_abs_diff_mutant_groups": max_mutant,
        "mean_reproduction_max_abs_diff_wild_type_vs_old_reference": max_wt,
        "mean_reproduction_max_rel_diff_wild_type_vs_old_reference": max_wt_rel,
        "n_groups": len(groups),
        "wild_type_acquisition_dates_from_name": len(dates),
        "wild_type_records_per_date_min_max": (
            [min(dates.values()), max(dates.values())] if dates else []
        ),
    }


def main() -> None:
    """Check all three datasets and write the results JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-base", required=True)
    parser.add_argument("--after-base", required=True)
    args = parser.parse_args()
    results = {
        name: check(
            osp.join(args.before_base, name), osp.join(args.after_base, name), spec
        )
        for name, spec in DATASETS.items()
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "scmd_morphology_per_replicate_counts.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
