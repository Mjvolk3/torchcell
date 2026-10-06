# experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_reference_replicates.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.sameith2015_reference_replicates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_reference_replicates
"""Measure the Sameith 2015 reference ``n_replicates`` fix (#630).

Two modes.

``--build-into <dir>`` builds ``SmMicroarraySameith2015Dataset`` and
``DmMicroarraySameith2015Dataset`` under ``<dir>/{sm,dm}_microarray_sameith2015`` from the
dev tree's ``raw/`` files (symlinked; nothing in the dev tree is written) with whatever
loader code is on ``PYTHONPATH``, then exits. It is run twice: once with the loader at
the commit before #630 (the baseline, which already carries #478 and #479) and once with
the fix (the scratch build).

The default mode reads three stores per dataset read-only, the dev tree
``$DATA_ROOT/data/torchcell/{sm,dm}_microarray_sameith2015`` (built before #478/#479),
``--baseline-root`` and ``--scratch-root``, and reports per store:

- the record count;
- the distribution of the reference ``phenotype_reference.n_replicates`` over every
  (record, gene) entry, and of its largest value per record;
- how many records have a reference ``n_replicates`` equal to the experiment's own
  ``n_replicates`` (same keys, same values), and how many genes have a reference count
  below the record's largest count (a gene missing on one of the record's arrays);

and, between the baseline and the scratch build, how many records have an identical
experiment and publication, an identical reference ``expression``, and a reference that
differs only in ``n_replicates``.

Writes ``results/sameith2015_reference_replicates.json``.

Run from the repo root::

    PYTHONPATH=<pre-#630 checkout> python <this script> --build-into <baseline>
    PYTHONPATH=<fix checkout> python <this script> --build-into <scratch>
    PYTHONPATH=<fix checkout> python <this script> \
        --baseline-root <baseline> --scratch-root <scratch>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
SM_SLUG = "sm_microarray_sameith2015"
DM_SLUG = "dm_microarray_sameith2015"
RAW_FILES = (
    "GSE42536_family.soft.gz",
    "GSE42536.pkl",
    "12915_2015_222_MOESM1_ESM.xlsx",
)


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


def build(target_root: str, data_root: str) -> None:
    """Build both Sameith stores under ``target_root`` from the dev tree's raw files."""
    from torchcell.datasets.scerevisiae.sameith2015 import (
        DmMicroarraySameith2015Dataset,
        SmMicroarraySameith2015Dataset,
    )
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    for slug, cls in (
        (SM_SLUG, SmMicroarraySameith2015Dataset),
        (DM_SLUG, DmMicroarraySameith2015Dataset),
    ):
        root = osp.join(target_root, slug)
        raw = osp.join(root, "raw")
        os.makedirs(raw, exist_ok=True)
        for name in RAW_FILES:
            target = osp.join(raw, name)
            if not osp.lexists(target):
                os.symlink(
                    osp.join(data_root, "data/torchcell", slug, "raw", name), target
                )
        dataset = cls(root=root, genome=genome)
        print(f"built {slug}: {len(dataset)} records at {root}")


def nan_safe(value: Any) -> Any:
    """Replace float NaN by a string so two dumps compare exactly (NaN != NaN)."""
    if isinstance(value, dict):
        return {k: nan_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [nan_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def summarize(root: str) -> dict[str, Any]:
    """Reference n_replicates distribution and consistency for one store."""
    records = read_store(root)
    entries: Counter[int] = Counter()
    record_max: Counter[int] = Counter()
    equal_to_experiment = 0
    below_record_max = 0
    for record in records:
        ref_n = record["reference"]["phenotype_reference"]["n_replicates"]
        exp_n = record["experiment"]["phenotype"]["n_replicates"]
        entries.update(ref_n.values())
        largest = max(ref_n.values())
        record_max[largest] += 1
        below_record_max += sum(v < largest for v in ref_n.values())
        equal_to_experiment += int(ref_n == exp_n)
    return {
        "root": root,
        "n_records": len(records),
        "reference_n_entries": {str(k): v for k, v in sorted(entries.items())},
        "reference_n_record_max": {str(k): v for k, v in sorted(record_max.items())},
        "records_reference_n_equal_experiment_n": equal_to_experiment,
        "reference_entries_below_record_max": below_record_max,
    }


def compare(baseline_root: str, scratch_root: str) -> dict[str, Any]:
    """Record-by-record comparison of the pre-#630 baseline and the fixed build."""
    baseline = read_store(baseline_root)
    scratch = read_store(scratch_root)
    if len(baseline) != len(scratch):
        raise ValueError(f"record counts differ: {len(baseline)} vs {len(scratch)}")
    same_experiment = 0
    same_publication = 0
    same_reference_expression = 0
    reference_differs_only_in_n = 0
    for old, new in zip(baseline, scratch, strict=True):
        same_experiment += int(
            nan_safe(old["experiment"]) == nan_safe(new["experiment"])
        )
        same_publication += int(old["publication"] == new["publication"])
        old_ref = nan_safe(old["reference"])
        new_ref = nan_safe(new["reference"])
        old_pheno = old_ref["phenotype_reference"]
        new_pheno = new_ref["phenotype_reference"]
        same_reference_expression += int(
            old_pheno["expression"] == new_pheno["expression"]
        )
        old_rest = {k: v for k, v in old_pheno.items() if k != "n_replicates"}
        new_rest = {k: v for k, v in new_pheno.items() if k != "n_replicates"}
        outer_old = {k: v for k, v in old_ref.items() if k != "phenotype_reference"}
        outer_new = {k: v for k, v in new_ref.items() if k != "phenotype_reference"}
        reference_differs_only_in_n += int(
            old_rest == new_rest
            and outer_old == outer_new
            and old_pheno["n_replicates"] != new_pheno["n_replicates"]
        )
    return {
        "n_records": len(scratch),
        "records_identical_experiment": same_experiment,
        "records_identical_publication": same_publication,
        "records_identical_reference_expression": same_reference_expression,
        "records_reference_differs_only_in_n_replicates": reference_differs_only_in_n,
    }


def main() -> None:
    """Build one store pair, or compare the dev, baseline and scratch stores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-into")
    parser.add_argument("--baseline-root")
    parser.add_argument("--scratch-root")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.build_into:
        build(args.build_into, data_root)
        return

    result: dict[str, Any] = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_reference_replicates.py"
    }
    for slug in (SM_SLUG, DM_SLUG):
        result[slug] = {
            "dev": summarize(osp.join(data_root, "data/torchcell", slug)),
            "baseline": summarize(osp.join(args.baseline_root, slug)),
            "scratch": summarize(osp.join(args.scratch_root, slug)),
            "baseline_vs_scratch": compare(
                osp.join(args.baseline_root, slug), osp.join(args.scratch_root, slug)
            ),
        }
    os.makedirs(RESULTS, exist_ok=True)
    with open(
        osp.join(RESULTS, "sameith2015_reference_replicates.json"), "w"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
