# experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_gene_names.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.messner2023_gene_names]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_gene_names
"""Measure what issue #485 changes in ``ProteomeMessner2023Dataset`` records.

Builds the dataset with the current loader into a SCRATCH root (``--scratch-root``;
the raw files are symlinked from the dev tree, and ``process`` re-verifies their
sha256), then reads that store and the dev store
(``$DATA_ROOT/data/torchcell/proteome_messner2023``) record by record, resolving
interned ``$ref`` pointers, and compares them:

- ``perturbed_gene_name`` old vs new for every record, classified as
  ``numeric_token`` (old name all digits), ``lowercase_orf`` (old name is the ORF in
  another case), ``other`` (any other difference) or unchanged;
- whether anything else in the record (experiment with the name blanked, reference,
  publication) differs;
- how many KO ORFs are no SGD GFF feature ID (name = the ORF by the loader's rule).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/messner2023_gene_names.json``
(summary) and ``messner2023_gene_names_changed.csv`` (one row per changed record).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_gene_names.py \
        --scratch-root <scratch dir>/proteome_messner2023
"""

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.messner2023 import (
    MATRIX_FILENAME,
    METADATA_FILENAME,
    ProteomeMessner2023Dataset,
    build_orf_to_gene_name_map,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"


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


def build_scratch(dev_root: str, scratch_root: str) -> None:
    """Build the store at ``scratch_root`` from the dev tree's raw files (symlinked)."""
    raw = osp.join(scratch_root, "raw")
    os.makedirs(raw, exist_ok=True)
    for name in (MATRIX_FILENAME, METADATA_FILENAME):
        dest = osp.join(raw, name)
        if not osp.exists(dest):
            os.symlink(osp.join(dev_root, "raw", name), dest)
    ProteomeMessner2023Dataset(root=scratch_root)


def classify(old: str, new: str, orf: str) -> str:
    """Kind of change from the old (filename) name to the new (SGD) name."""
    if old == new:
        return "unchanged"
    if old.isdigit():
        return "numeric_token"
    if old.upper() == orf:
        return "lowercase_orf"
    return "other"


def main() -> None:
    """Build the scratch store, compare it with the dev store, write the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    load_dotenv()
    dev_root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/proteome_messner2023")
    build_scratch(dev_root, args.scratch_root)

    dev_csv = pd.read_csv(osp.join(dev_root, "preprocess", "data.csv"))
    new_csv = pd.read_csv(osp.join(args.scratch_root, "preprocess", "data.csv"))
    assert list(dev_csv["filename"]) == list(new_csv["filename"])
    dev = read_store(dev_root)
    new = read_store(args.scratch_root)
    assert len(dev) == len(new) == len(dev_csv)

    orf2name = build_orf_to_gene_name_map()
    changed: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    other_fields_differ = 0
    for i, (d, n) in enumerate(zip(dev, new, strict=True)):
        dp = d["experiment"]["genotype"]["perturbations"][0]
        np_ = n["experiment"]["genotype"]["perturbations"][0]
        assert dp["systematic_gene_name"] == np_["systematic_gene_name"]
        orf = dp["systematic_gene_name"]
        kind = classify(dp["perturbed_gene_name"], np_["perturbed_gene_name"], orf)
        kinds[kind] += 1
        if kind != "unchanged":
            changed.append(
                {
                    "index": i,
                    "filename": dev_csv["filename"][i],
                    "systematic_gene_name": orf,
                    "old_perturbed_gene_name": dp["perturbed_gene_name"],
                    "new_perturbed_gene_name": np_["perturbed_gene_name"],
                    "kind": kind,
                    "orf_in_sgd_gff": orf in orf2name,
                }
            )
        dp_rest = {**dp, "perturbed_gene_name": None}
        np_rest = {**np_, "perturbed_gene_name": None}
        d_exp = {**d["experiment"], "genotype": None}
        n_exp = {**n["experiment"], "genotype": None}
        if (
            dp_rest != np_rest
            or d_exp != n_exp
            or d["reference"] != n["reference"]
            or d["publication"] != n["publication"]
        ):
            other_fields_differ += 1

    orfs = sorted(set(new_csv["orf"]))
    not_in_gff = [o for o in orfs if o not in orf2name]
    changed_df = pd.DataFrame(changed)
    summary = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_gene_names.py",
        "dev_root": dev_root,
        "scratch_root": args.scratch_root,
        "n_records_dev": len(dev),
        "n_records_scratch": len(new),
        "perturbed_gene_name_change_kinds": dict(sorted(kinds.items())),
        "n_records_perturbed_gene_name_changed": len(changed),
        "n_distinct_orfs_numeric_token": int(
            changed_df[changed_df["kind"] == "numeric_token"][
                "systematic_gene_name"
            ].nunique()
        ),
        "lowercase_orf_now_a_standard_name": int(
            (
                changed_df[changed_df["kind"] == "lowercase_orf"][
                    "new_perturbed_gene_name"
                ]
                != changed_df[changed_df["kind"] == "lowercase_orf"][
                    "systematic_gene_name"
                ]
            ).sum()
        ),
        "other_breakdown": {
            "old_unknown": int(
                (
                    changed_df[changed_df["kind"] == "other"]["old_perturbed_gene_name"]
                    == "Unknown"
                ).sum()
            ),
            "case_only": int(
                (
                    changed_df[changed_df["kind"] == "other"][
                        "old_perturbed_gene_name"
                    ].str.upper()
                    == changed_df[changed_df["kind"] == "other"][
                        "new_perturbed_gene_name"
                    ]
                ).sum()
            ),
            "orf_not_sgd_gff_feature_id": int(
                (~changed_df[changed_df["kind"] == "other"]["orf_in_sgd_gff"]).sum()
            ),
        },
        "n_records_with_any_other_field_different": other_fields_differ,
        "n_ko_orfs": len(orfs),
        "n_ko_orfs_not_sgd_gff_feature_id": len(not_in_gff),
        "n_records_orf_not_sgd_gff_feature_id": int(
            new_csv["orf"].isin(not_in_gff).sum()
        ),
        "examples": {
            "10_9_hpr57_ko_YBL007C_2824_0.49": new_csv.set_index("filename").loc[
                "10_9_hpr57_ko_YBL007C_2824_0.49", "gene"
            ],
            "10_9_hpr10_ko_YBR174C_YBR174c_0.23": new_csv.set_index("filename").loc[
                "10_9_hpr10_ko_YBR174C_YBR174c_0.23", "gene"
            ],
        },
        "new_name_equals_orf": int((new_csv["gene"] == new_csv["orf"]).sum()),
        "new_name_has_lowercase": int(
            (new_csv["gene"] != new_csv["gene"].str.upper()).sum()
        ),
        "new_name_all_digits": int(new_csv["gene"].astype(str).str.isdigit().sum()),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "messner2023_gene_names.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    changed_df.to_csv(
        osp.join(RESULTS, "messner2023_gene_names_changed.csv"), index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
