# experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.no_verifier_store_report_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables
"""Render the L0-L4 table of each #889 store from the report its verifier wrote.

Reads ``$DATA_ROOT/data/torchcell/<store>/preprocess/verification_report.json`` (written
by the sweep's registry route, see ``kg4_pre_build_verification_sweep.py``) and prints
one markdown table per store, rule by rule in report order, ready to paste under the
dataset's dendron note. Messages are cut at 220 characters; the report holds the rest.

    python experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py [store ...]
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from datetime import datetime

from dotenv import load_dotenv

from torchcell.verification.report import VerificationReport

STORES = [
    "smf_costanzo2016",
    "dmf_costanzo2016",
    "dmi_costanzo2016",
    "smf_kuzmin2018",
    "dmf_kuzmin2018",
    "tmf_kuzmin2018",
    "dmi_kuzmin2018",
    "tmi_kuzmin2018",
    "smf_kuzmin2020",
    "dmf_kuzmin2020",
    "tmf_kuzmin2020",
    "dmi_kuzmin2020",
    "tmi_kuzmin2020",
    "gene_essentiality_sgd",
    "syn_leth_db_yeast",
    "syn_rescue_db_yeast",
]


def render(store: str, data_root: str) -> str:
    """One store's report as a markdown table, headed by its verdict and write time."""
    path = osp.join(
        data_root, "data/torchcell", store, "preprocess", "verification_report.json"
    )
    with open(path, encoding="utf-8") as handle:
        report = VerificationReport.model_validate_json(handle.read())
    written = datetime.fromtimestamp(osp.getmtime(path)).isoformat(timespec="seconds")
    failed = [r for r in report.results if not r.passed]
    lines = [
        f"`{store}`: **{'PASS' if report.passed else 'FAIL'}**, "
        f"{len(report.results)} rules, {len(failed)} failing "
        f"(report written {written})",
        "",
        "| Level | Rule | Verdict | Message |",
        "|---|---|---|---|",
    ]
    for result in report.results:
        message = result.message.replace("|", "\\|")
        if len(message) > 220:
            message = message[:217] + "..."
        lines.append(
            f"| {result.level.name} | `{result.name}` | "
            f"{'PASS' if result.passed else '**FAIL**'} | {message} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Print the tables of the stores named on the command line (default: all 16)."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    stores = sys.argv[1:] or STORES
    print("\n\n".join(render(store, data_root) for store in stores))


if __name__ == "__main__":
    main()
