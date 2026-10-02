# experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_reference_replicates.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kemmeren2014_reference_replicates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_reference_replicates
"""Measure the Kemmeren 2014 reference ``n_replicates`` fix (#484).

Builds ``MicroarrayKemmeren2014Dataset`` into a scratch root (``--build``; the raw GEO
files and Table S1 are symlinked from the dev tree's ``raw/``, nothing in the dev tree is
written), then reads the scratch store and the dev-tree store
``$DATA_ROOT/data/torchcell/microarray_kemmeren2014`` read-only and reports, per store:

- the record count and the strain split of ``genome_reference``;
- the distribution of the reference ``phenotype_reference.n_replicates`` over every
  (record, gene) entry, overall and by strain;
- how many records have a reference ``n_replicates`` equal to the experiment's own
  ``n_replicates`` (same keys, same values);
- the own-gene sign check (the deleted gene's stored ``expression_log2_ratio``).

Writes ``results/kemmeren2014_reference_replicates.json``.

Run from the repo root (the build loads about 1.4 GB of GEO pickles; run it in the
background)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_reference_replicates.py \
        --scratch-root <dir> [--build --workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
SLUG = "microarray_kemmeren2014"
ACCESSIONS = ("GSE42527", "GSE42526", "GSE42241", "GSE42240", "GSE42217", "GSE42215")
RAW_FILES = (
    *(f"{a}_family.soft.gz" for a in ACCESSIONS),
    *(f"{a}.pkl" for a in ACCESSIONS),
    "kemmeren2014_table_s1.xlsx",
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


def build(scratch_root: str, data_root: str, workers: int) -> None:
    """Build the Kemmeren store under ``scratch_root`` from the dev tree's raw files."""
    from torchcell.datasets.scerevisiae.kemmeren2014 import (
        MicroarrayKemmeren2014Dataset,
    )
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(scratch_root, SLUG)
    raw = osp.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    for name in RAW_FILES:
        target = osp.join(raw, name)
        if not osp.lexists(target):
            os.symlink(osp.join(data_root, "data/torchcell", SLUG, "raw", name), target)
    dataset = MicroarrayKemmeren2014Dataset(
        root=root, genome=genome, process_workers=workers
    )
    print(f"built {SLUG}: {len(dataset)} records at {root}")


def summarize(root: str) -> dict[str, Any]:
    """Reference n_replicates distribution and consistency for one store."""
    records = read_store(root)
    overall: Counter[int] = Counter()
    by_strain: dict[str, Counter[int]] = {}
    per_record_max: Counter[int] = Counter()
    equal_to_experiment = 0
    own_gene = []
    for record in records:
        strain = record["reference"]["genome_reference"]["strain"]
        ref_n = record["reference"]["phenotype_reference"]["n_replicates"]
        exp_n = record["experiment"]["phenotype"]["n_replicates"]
        overall.update(ref_n.values())
        by_strain.setdefault(strain, Counter()).update(ref_n.values())
        per_record_max[max(ref_n.values())] += 1
        equal_to_experiment += int(ref_n == exp_n)
        log2 = record["experiment"]["phenotype"]["expression_log2_ratio"]
        (perturbation,) = record["experiment"]["genotype"]["perturbations"]
        gene = perturbation["systematic_gene_name"]
        if gene in log2 and np.isfinite(log2[gene]):
            own_gene.append(float(log2[gene]))
    own = np.array(own_gene)
    return {
        "root": root,
        "n_records": len(records),
        "strain_records": dict(
            Counter(r["reference"]["genome_reference"]["strain"] for r in records)
        ),
        "reference_n_entries": {str(k): v for k, v in sorted(overall.items())},
        "reference_n_entries_by_strain": {
            strain: {str(k): v for k, v in sorted(counts.items())}
            for strain, counts in sorted(by_strain.items())
        },
        "reference_n_record_max": {
            str(k): v for k, v in sorted(per_record_max.items())
        },
        "records_reference_n_equal_experiment_n": equal_to_experiment,
        "own_gene_sign": {
            "n_records": int(own.size),
            "fraction_negative": float(np.mean(own < 0)),
            "median": float(np.median(own)),
        },
    }


def main() -> None:
    """Optionally build the scratch store, then compare it with the dev tree."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.build:
        build(args.scratch_root, data_root, args.workers)

    result: dict[str, Any] = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/kemmeren2014_reference_replicates.py",
        "dev": summarize(osp.join(data_root, "data/torchcell", SLUG)),
        "scratch": summarize(osp.join(args.scratch_root, SLUG)),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(
        osp.join(RESULTS, "kemmeren2014_reference_replicates.json"), "w"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
