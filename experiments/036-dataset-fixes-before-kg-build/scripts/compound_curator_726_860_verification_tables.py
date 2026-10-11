# experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_curator_726_860_verification_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables
r"""Render the L0-L4 verification of the stores the #726/#860 pass touched, per dataset.

The reports are the ones each dataset's own route wrote into its dev store's
``preprocess/verification_report*.json``: by the rebuild itself
(``build_dataset_lmdb --verify``, slurm arrays 3809 and 3836) for the datasets whose module
declares a per-dataset entry point, and by ``kg4_pre_build_verification_sweep.py`` (slurm
array 3872, ``gh_compound_curator_726_860_verify.slurm``) for the family-runner datasets
and the touched loaders whose stores were not rebuilt. This reads them back from disk, so
a table is regenerated from the reports rather than retyped, and prints, per dataset, the
store's record count, its build time (``preprocess/build_manifest.json``), the report time
(file mtime), a verdict, and one markdown table per report of every rule: level, name,
pass or fail, and the report's own message.

A report OLDER than its store's build manifest is flagged ``STALE REPORT``: it verified
different bytes.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables.py \
        --classes-file /scratch/projects/torchcell/database/slurm/classes_726_860_tables.txt \
        > experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860_verification_tables.txt
"""

import argparse
import glob
import json
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.database.build_dataset_lmdb import (
    dataset_default_root,
    resolve_dataset_class,
)


def report_rows(path: str) -> list[dict[str, Any]]:
    """The ``results`` rules of one verification report."""
    with open(path, encoding="utf-8") as handle:
        rules: list[dict[str, Any]] = json.load(handle)["results"]
    return rules


def record_count(root: str) -> int:
    """LMDB entries of a dev store."""
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        count = int(txn.stat()["entries"])
    env.close()
    return count


def built_at(root: str) -> datetime:
    """When the store's build manifest was written (UTC)."""
    with open(osp.join(root, "preprocess", "build_manifest.json")) as handle:
        stamp = str(json.load(handle)["built_at"])
    parsed = datetime.fromisoformat(stamp)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def render(class_name: str, data_root: str) -> tuple[list[str], str]:
    """One dataset's heading, verdict line and rule tables, and its verdict."""
    root = osp.join(data_root, dataset_default_root(resolve_dataset_class(class_name)))
    reports = sorted(
        glob.glob(osp.join(root, "preprocess", "verification_report*.json"))
    )
    built = built_at(root)
    verdict = "NO REPORT" if not reports else "PASS"
    body: list[str] = []
    for path in reports:
        written = datetime.fromtimestamp(osp.getmtime(path), tz=UTC)
        stale = written < built
        rules = report_rows(path)
        failed = [r for r in rules if not r["passed"]]
        if stale:
            verdict = "STALE REPORT"
        elif failed and verdict == "PASS":
            verdict = "FAIL"
        body.extend(
            [
                f"Report `{osp.basename(path)}` (written {written:%Y-%m-%d %H:%M} UTC"
                f"{', STALE REPORT' if stale else ''}):",
                "",
                "| level | rule | result | message |",
                "|---|---|---|---|",
            ]
        )
        for rule in rules:
            message = str(rule["message"]).replace("|", "\\|").replace("\n", " ")
            body.append(
                f"| L{rule['level']} | `{rule['name']}` | "
                f"{'pass' if rule['passed'] else 'FAIL'} | {message} |"
            )
        body.append("")
    head = [
        f"#### `{class_name}`",
        "",
        f"{record_count(root):,} records; store built {built:%Y-%m-%d %H:%M} UTC; "
        f"{len(reports)} report(s); verdict **{verdict}**.",
        "",
    ]
    return head + body, verdict


def main() -> None:
    """Print every dataset's tables, then a one-line-per-dataset summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes-file", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    summary: list[str] = ["| dataset | verdict |", "|---|---|"]
    for name in args.classes_file.read_text().split():
        lines, verdict = render(name, data_root)
        print("\n".join(lines))
        summary.append(f"| `{name}` | {verdict} |")
    print("### Summary\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
