#!/usr/bin/env python
# experiments/database/scripts/potel2018_candidate_verdict.py
# [[experiments.database.scripts.potel2018_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/potel2018_candidate_verdict.py
"""Re-gate the Potel 2018 row with the loader's inputs and store the verdict (#921).

The row is on the bacteria schedule ("Potel 2018 phosphoproteome"), so this is
``candidate-gate gate`` on that row with the two inputs the loader decides: the
phenotype class its records use (``PhosphositeIntensityPhenotype``, G5; the release names
no compound) and the issue the row is tracked under (#921). G2 reads W3110 now that a
genome class reads ``ecoli_K12_W3110_ASM1024v1``; G3 inventories the raw mirror the
release inventory deposited
(``experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_release_inventory.py``).

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``) after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/potel2018_candidate_verdict.py          # print
    python experiments/database/scripts/potel2018_candidate_verdict.py --write  # store
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from torchcell.candidates.cli import main as candidate_gate
from torchcell.datasets.ecoli import potel2018 as loader

REPO: Final = Path(__file__).resolve().parents[3]
ROW: Final = "Potel 2018 phosphoproteome"
PHENOTYPE_CLASS: Final = "PhosphositeIntensityPhenotype"
ISSUE: Final = 921


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
