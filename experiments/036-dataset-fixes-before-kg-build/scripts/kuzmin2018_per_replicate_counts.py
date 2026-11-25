# experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2018_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kuzmin2018_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2018_per_replicate_counts
"""Prove the per-strain ``SmfKuzmin2018Dataset`` counts against the release.

Reads two already-built stores, record by record with interned ``$ref`` pointers
resolved: ``--before-root`` (origin/main, one query record per allele NAME, 1,539) and
``--after-root`` (one query record per query STRAIN). Both are built into scratch roots,
never the shared dev tree. It reports:

- the release side, from Data S1 itself: array and digenic query strains, how many
  state a fitness, and how many allele names carry two strains;
- records per kind (array / query) and the total, before and after;
- that every before-record (experiment, reference, publication) is an after-record,
  byte for byte, and which after-records are new;
- the mean-reproduction check: after-records grouped by (genotype, environment), which
  keeps a strain's own ``strain_id``, averaged and compared with the before-record of
  that group; and the same grouping with ``strain_id`` set aside (one group per
  genotype as written without the strain, so a gene's array and query strains share a
  group), where the three two-strain genes differ from the before value because the old
  store held the FIRST strain's value, not a mean.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/kuzmin2018_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2018_per_replicate_counts.py \
        --before-root <scratch>/before/smf_kuzmin2018 \
        --after-root <scratch>/after/smf_kuzmin2018
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

from torchcell.data import verify_raw_files
from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.kuzmin2018 import RAW_TABLE_NAME, RAW_TABLE_SHA256

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
QUERY_FITNESS = "Query single/double mutant fitness"
ARRAY_FITNESS = "Array single mutant fitness"


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


def _dump(record: dict[str, Any]) -> str:
    """A canonical JSON string of one whole record."""
    return json.dumps(record, sort_keys=True, default=str)


def _key(experiment: dict[str, Any], keep_strain: bool) -> str:
    """The (genotype, environment) group key, optionally without each ``strain_id``."""
    perturbations = [
        {k: v for k, v in p.items() if keep_strain or k != "strain_id"}
        for p in experiment["genotype"]["perturbations"]
    ]
    return json.dumps([perturbations, experiment["environment"]], sort_keys=True)


def _release_side(raw: str) -> dict[str, Any]:
    """Strain counts read straight off Data S1 (sha256 verified first)."""
    verify_raw_files(osp.dirname(raw), {RAW_TABLE_NAME: RAW_TABLE_SHA256})
    df = pd.read_csv(raw, sep="\t")
    digenic = df[df["Combined mutant type"] == "digenic"]
    query = digenic.groupby("Query strain ID")[QUERY_FITNESS]
    array = df.groupby("Array strain ID")[ARRAY_FITNESS]
    names = digenic.groupby("Query allele name")["Query strain ID"].unique()
    two = {n: sorted(s) for n, s in names.items() if len(s) > 1}
    return {
        "data_s1_rows": len(df),
        "array_strains": int(array.ngroups),
        "array_strains_with_fitness": int(array.count().gt(0).sum()),
        "array_strains_with_varying_fitness": int(
            array.nunique(dropna=False).gt(1).sum()
        ),
        "digenic_query_strains": int(query.ngroups),
        "digenic_query_strains_with_fitness": int(query.count().gt(0).sum()),
        "digenic_query_strains_with_varying_fitness": int(
            query.nunique(dropna=False).gt(1).sum()
        ),
        "digenic_query_allele_names": len(names),
        "allele_names_with_two_strains": {
            n: {
                s: float(
                    digenic.loc[digenic["Query strain ID"] == s, QUERY_FITNESS].iloc[0]
                )
                for s in strains
            }
            for n, strains in two.items()
        },
    }


def _value(record: dict[str, Any]) -> float:
    return float(record["experiment"]["phenotype"]["fitness"])


def main() -> None:
    """Read both stores and Data S1, compute counts and the mean check, write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()

    release = _release_side(osp.join(args.after_root, "raw", RAW_TABLE_NAME))
    before = read_store(args.before_root)
    after = read_store(args.after_root)
    after_csv = pd.read_csv(osp.join(args.after_root, "preprocess", "data.csv"))
    before_csv = pd.read_csv(osp.join(args.before_root, "preprocess", "data.csv"))

    before_dumps = Counter(_dump(r) for r in before)
    after_dumps = Counter(_dump(r) for r in after)
    missing = before_dumps - after_dumps
    added = after_dumps - before_dumps
    added_records = [json.loads(d) for d in added.elements()]

    # Mean reproduction: grouped with and without each strain_id, against the before-store.
    checks: dict[str, Any] = {}
    for label, keep_strain in (
        ("by_genotype_with_strain_id", True),
        ("by_genotype_without_strain_id", False),
    ):
        groups_after: dict[str, list[float]] = defaultdict(list)
        groups_before: dict[str, list[float]] = defaultdict(list)
        for r in after:
            groups_after[_key(r["experiment"], keep_strain)].append(_value(r))
        for r in before:
            groups_before[_key(r["experiment"], keep_strain)].append(_value(r))
        if set(groups_before) - set(groups_after):
            raise RuntimeError(f"{label}: a before-group has no after-records")
        diffs = {
            k: abs(sum(v) / len(v) - sum(groups_before[k]) / len(groups_before[k]))
            for k, v in groups_after.items()
            if k in groups_before
        }
        differing = {k: d for k, d in diffs.items() if d > 0}
        checks[label] = {
            "groups_after": len(groups_after),
            "groups_after_not_in_before": len(set(groups_after) - set(groups_before)),
            "group_multiplicity_after": {
                str(n): c
                for n, c in sorted(
                    Counter(len(v) for v in groups_after.values()).items()
                )
            },
            "max_abs_diff": max(diffs.values()),
            "groups_differing": [
                {
                    "genes": [p["perturbed_gene_name"] for p in json.loads(k)[0]],
                    "after_values": groups_after[k],
                    "before_values": groups_before[k],
                    "abs_diff": d,
                }
                for k, d in sorted(differing.items(), key=lambda kv: -kv[1])
            ],
        }

    summary = {
        "release": release,
        "before_records": len(before),
        "after_records": len(after),
        "before_by_kind": before_csv["smf_type"].value_counts().to_dict(),
        "after_by_kind": after_csv["smf_type"].value_counts().to_dict(),
        "after_query_strains_distinct": int(
            after_csv.loc[
                after_csv["smf_type"] == "query_smf", "Query strain ID"
            ].nunique()
        ),
        "before_records_missing_from_after": sum(missing.values()),
        "after_records_not_in_before": [
            {
                "strain_id": r["experiment"]["genotype"]["perturbations"][0][
                    "strain_id"
                ],
                "gene": r["experiment"]["genotype"]["perturbations"][0][
                    "perturbed_gene_name"
                ],
                "fitness": r["experiment"]["phenotype"]["fitness"],
                "fitness_std": r["experiment"]["phenotype"]["fitness_std"],
                "replicate_id": r["experiment"]["phenotype"]["replicate_id"],
            }
            for r in added_records
        ],
        "references_distinct_after": len({_dump(r["reference"]) for r in after}),
        "references_equal_to_before": {_dump(r["reference"]) for r in after}
        == {_dump(r["reference"]) for r in before},
        "mean_reproduction": checks,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "kuzmin2018_per_replicate_counts.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
