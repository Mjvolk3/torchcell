# experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_released_columns_store_report.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.bacteria_released_columns_store_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_released_columns_store_report

r"""Record counts and L0-L4 tables of the 13 stores PR #1072 rebuilt (#879 #904 #905 #914 #915).

Reads each dev store's LMDB entry count and its ``preprocess/verification_report.json``
(written by ``build_dataset_lmdb --verify`` in slurm array 4475, or by the loader's own
``verify_build`` for a store the CLI cannot verify alone), and writes
``results/bacteria_released_columns_store_report.json`` plus one markdown L0-L4 table
per store to stdout, the tables the dataset notes carry. ``--before`` takes the counts
measured before the rebuild (a JSON of class -> count) and adds them to the output.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_released_columns_store_report.py \
        --before counts_before.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

import lmdb
from dotenv import load_dotenv

import torchcell.datasets  # noqa: F401  (registers every dataset class)
from torchcell.database.build_dataset_lmdb import dataset_default_root
from torchcell.datasets.dataset_registry import dataset_registry

#: The rebuilt stores, in array 4475's order.
STORES: tuple[str, ...] = (
    "CrispriGuideFitnessWang2018Dataset",
    "CrispriGeneFitnessWang2018Dataset",
    "CrispriScreenRousset2018Dataset",
    "ProteomeBrunk2016Dataset",
    "MetabolomeBrunk2016Dataset",
    "ExometaboliteBrunk2016Dataset",
    "BiofuelTiterBrunk2016Dataset",
    "IsoprenolSelectionMenasalvas2025Dataset",
    "ProteomeMenasalvas2025Dataset",
    "MetaboliteGrowthPhaseMenasalvas2025Dataset",
    "MetaboliteProductionPhaseMenasalvas2025Dataset",
    "IsopentenolTiterFoo2014Dataset",
    "ProteinFoldChangeCaglar2017Dataset",
)
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "bacteria_released_columns_store_report.json",
)


def store_count(processed: str) -> int:
    """Entries of the store's record LMDB under ``processed``."""
    path = osp.join(processed, "lmdb")
    env = lmdb.open(path, readonly=True, lock=False)
    count = int(env.stat()["entries"])
    env.close()
    return count


def table(report: dict[str, Any]) -> str:
    """One markdown L0-L4 table: level, rule, gate or report, verdict, message."""
    rows = ["| level | rule | kind | verdict | message |", "|---|---|---|---|---|"]
    for result in sorted(report["results"], key=lambda r: (r["level"], r["name"])):
        message = str(result["message"]).replace("|", "/").replace("\n", " ")
        rows.append(
            f"| L{result['level']} | `{result['name']}` | {result['finding_class']} | "
            f"{'pass' if result['passed'] else 'FAIL'} | {message[:220]} |"
        )
    return "\n".join(rows)


def main() -> None:
    """Measure every store, print its table, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="CLASS=PATH: read this store from PATH (a superseded sibling, when another "
        "branch has since rebuilt the shared dev store)",
    )
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.root)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    with open(args.before) as handle:
        before: dict[str, int | None] = json.load(handle)
    out: dict[str, Any] = {}
    for name in STORES:
        root = overrides.get(
            name, osp.join(data_root, dataset_default_root(dataset_registry[name]))
        )
        lmdb_root = osp.join(root, "processed")
        report_root = osp.join(root, "preprocess")
        if name in overrides:
            # a superseded store keeps processed.superseded.<ts> and
            # preprocess.superseded.<ts> as siblings of the live tree
            stamp = root.rsplit(".superseded.", 1)[1]
            base = root.rsplit("/", 1)[0]
            lmdb_root = osp.join(base, f"processed.superseded.{stamp}")
            report_root = osp.join(base, f"preprocess.superseded.{stamp}")
        with open(osp.join(report_root, "verification_report.json")) as handle:
            report = json.load(handle)
        gated = [r for r in report["results"] if r["finding_class"] == "gate"]
        out[name] = {
            "before": before[name],
            "after": store_count(lmdb_root),
            "store": lmdb_root,
            "gate_rules": len(gated),
            "gate_failures": [r["name"] for r in gated if not r["passed"]],
            "reported_findings": [
                r["name"]
                for r in report["results"]
                if r["finding_class"] == "report" and not r["passed"]
            ],
        }
        print(f"### {name}\n\n{table(report)}\n")
    with open(RESULTS, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
