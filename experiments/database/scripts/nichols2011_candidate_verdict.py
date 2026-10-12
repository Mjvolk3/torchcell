#!/usr/bin/env python
# experiments/database/scripts/nichols2011_candidate_verdict.py
# [[experiments.database.scripts.nichols2011_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/nichols2011_candidate_verdict.py
"""Gate Nichols 2011 (bacteria row "Nichols 2011", #691) with the loader's evidence.

``candidate-gate gate --citation-key nicholsPhenotypicLandscapeBacterial2011`` composes
the same gates; this script adds what the loader measured and sourced:

- G3: the deposit is the owner's manual download (#1046); the gate's inventory counts
  worksheet rows only in files named ``.xlsx``, so it counts 0 here. The counts are
  ``nichols2011_release_checks.py``'s (``experiments/database/results/
  nichols2011_release_checks.json``), and the Table S2 shape quote is attached.
- G4: the key half. The value half is not run through the gate: the one store that could
  hold these values, Shiver 2016's, serves only Shiver's own 57 conditions by
  construction and its dev store was built from a dirty tree, which
  ``read_store_values`` refuses. The overlap with Shiver's unloaded batch 0 is measured
  by the release-checks script instead (235 conditions, 831,939 overlapping cells, 0
  identical, pooled r 0.800).
- G5: ``EnvironmentResponsePhenotype`` exists; the compound names are Table S1's own,
  and the ones the curated table does not hold are issue #1063.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/nichols2011_candidate_verdict.py          # print
    python experiments/database/scripts/nichols2011_candidate_verdict.py --write  # store
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from torchcell.candidates import gates, store
from torchcell.candidates.cli import load_doi_to_pmid, load_table_module, table_rows
from torchcell.candidates.findings import finding_for
from torchcell.candidates.gates import GateRun
from torchcell.candidates.verdict import CandidateVerdict, GateName
from torchcell.datasets.ecoli import nichols2011 as loader

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
ROW: Final = "Nichols 2011"
CITATION_KEY: Final = loader.CITATION_KEY
PHENOTYPE_CLASS: Final = "EnvironmentResponsePhenotype"
#: The open curation issue for the Table S1 names the compound table lacks.
G5_ISSUE: Final = 1063
#: Every compound name the loader stores, as Table S1 writes it.
COMPOUNDS: Final = tuple(sorted({c for s in loader.SCREENS for c in s.compounds}))
#: The loader's sourced statements attached to the gate each supports.
EVIDENCE: Final = (
    ("G3", loader.TABLE_S2),
    ("G5", loader.RESCALED),
    ("G5", loader.UNAVERAGED_S_SCORES),
    ("G5", loader.REPLICATE_DESIGN),
)


def git_commit() -> str:
    """The checkout's HEAD commit."""
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def with_evidence(verdict: CandidateVerdict) -> CandidateVerdict:
    """The verdict with the loader's quotes added to the measured gates they support."""
    data = verdict.model_dump()
    for gate in data["gates"]:
        if gate["outcome"] == "unmeasured":
            continue
        extra = [v.model_dump() for g, v in EVIDENCE if g == gate["gate"]]
        gate["evidence"] = [*gate["evidence"], *extra]
    return CandidateVerdict.model_validate(data)


def main() -> None:
    """Gate the row with the loader's inputs; print, or with --write store it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO / ".env")
    root = os.environ["DATA_ROOT"]
    row = gates.find_row(table_rows("bacteria", load_table_module("bacteria")), ROW)
    index = gates.module_index()
    g1 = gates.evaluate_g1(row, finding_for(row.name))
    steps: dict[GateName, Callable[[], GateRun]] = {
        "G2": lambda: gates.evaluate_g2(
            row, root, source_kind=g1.source_kind, verify=False
        ),
        "G3": lambda: gates.evaluate_g3(gates.inventory(CITATION_KEY, root)),
        "G4": lambda: gates.evaluate_g4_key(
            row,
            CITATION_KEY,
            index,
            doi_to_pmid=load_doi_to_pmid(),
            aggregator_pmids=gates.aggregator_pmid_sets(root),
        ),
        "G5": lambda: gates.evaluate_g5(PHENOTYPE_CLASS, COMPOUNDS, issue=G5_ISSUE),
    }
    runs: dict[GateName, GateRun] = {}
    stopped = g1.result.outcome in ("fail", "blocked")
    for name, step in steps.items():
        if stopped:
            break
        runs[name] = step()
        stopped = runs[name].result.outcome in ("fail", "blocked")
    verdict = with_evidence(
        gates.compose_verdict(
            row=row,
            citation_key=CITATION_KEY,
            g1=g1,
            runs=runs,
            zotero=gates.zotero_item(CITATION_KEY, root),
            decided_at=date.today().isoformat(),
            torchcell_commit=git_commit(),
        )
    )
    print(f"{CITATION_KEY}: {verdict.outcome}")
    for g in verdict.gates:
        print(f"  {g.gate} {g.outcome}: {g.reason}")
    if args.write:
        print(f"wrote {store.write_verdict(verdict)}")


if __name__ == "__main__":
    main()
