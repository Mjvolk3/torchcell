# experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_curator_726_860_verification_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables
r"""Render the L0-L4 verification of the stores the #726/#860 pass rebuilt, per dataset.

Reads the results file ``kg4_pre_build_verification_sweep.py --out`` wrote for these
datasets (each row names the ``preprocess/verification_report*.json`` files its route
wrote) and prints, for every dataset, one markdown table of every rule: level, rule name,
pass or fail, and the report's own message. The dendron notes paste these tables, so a
note's L0-L4 table is regenerated from the reports rather than retyped.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860_verification_tables.py \
        --results experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860_verification.json \
        > experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860_verification_tables.txt
"""

import argparse
import json
import os.path as osp
from typing import Any


def report_rows(path: str) -> list[dict[str, Any]]:
    """The ``results`` rules of one verification report."""
    with open(path, encoding="utf-8") as handle:
        rules: list[dict[str, Any]] = json.load(handle)["results"]
    return rules


def render(row: dict[str, Any]) -> list[str]:
    """One dataset's heading, verdict line and rule table."""
    lines = [
        f"#### `{row['dataset_class']}`",
        "",
        f"{row['records']} records; route `{row['route_detail']}`; verdict "
        f"**{row['verdict']}** over {row['n_rules']} rules; store built "
        f"{row['manifest_built_at']}; verified at `{(row['git_head'] or '-')[:10]}`.",
        "",
    ]
    if row["error"]:
        lines.extend([f"Error: {row['error']}", ""])
    for path in row["reports"]:
        lines.extend(
            [
                f"Report `{osp.basename(path)}`:",
                "",
                "| level | rule | result | message |",
                "|---|---|---|---|",
            ]
        )
        for rule in report_rows(path):
            message = str(rule["message"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| L{rule['level']} | `{rule['name']}` | "
                f"{'pass' if rule['passed'] else 'FAIL'} | {message} |"
            )
        lines.append("")
    return lines


def main() -> None:
    """Print every dataset's rule tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)
    for row in results["datasets"].values():
        print("\n".join(render(row)))


if __name__ == "__main__":
    main()
