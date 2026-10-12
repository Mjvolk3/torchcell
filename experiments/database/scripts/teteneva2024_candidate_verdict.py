#!/usr/bin/env python
# experiments/database/scripts/teteneva2024_candidate_verdict.py
# [[experiments.database.scripts.teteneva2024_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/teteneva2024_candidate_verdict.py
"""Re-gate the Teteneva 2024 row with the loader's inputs and store the verdict (#910).

The row is on the bacteria schedule ("Teteneva 2024 lake water"), so this is
``candidate-gate gate`` on that row with the two inputs the loader decides: the
phenotype class its records use (``EnvironmentResponsePhenotype``, G5; Table S4 names no
compound) and the issue the row is tracked under (#910). G2 reads W3110 since its
genome class landed (``ecoli_K12_W3110_ASM1024v1`` resolves and reads); G3 inventories
the raw mirror's seven files (the SI deposit, the PMC paper text and the two derived
texts).

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``) after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/teteneva2024_candidate_verdict.py          # print
    python experiments/database/scripts/teteneva2024_candidate_verdict.py --write  # store
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from torchcell.candidates.cli import main as candidate_gate
from torchcell.datasets.ecoli import teteneva2024 as loader

REPO: Final = Path(__file__).resolve().parents[3]
ROW: Final = "Teteneva 2024 lake water"
PHENOTYPE_CLASS: Final = "EnvironmentResponsePhenotype"
ISSUE: Final = 910


def arguments(write: bool) -> list[str]:
    """The ``candidate-gate`` argument vector for this row."""
    out = [
        "--store-dir",
        str(REPO / "database" / "candidates"),
        "gate",
        "--row",
        ROW,
        "--table",
        "bacteria",
        "--phenotype-class",
        PHENOTYPE_CLASS,
        "--issue",
        str(ISSUE),
    ]
    return [*out, "--write"] if write else out


def main() -> None:
    """Gate the row; with ``--write`` store ``database/candidates/<key>.json``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    print(f"citation key {loader.CITATION_KEY}, row {ROW!r}")
    raise SystemExit(candidate_gate(arguments(args.write)))


if __name__ == "__main__":
    main()
