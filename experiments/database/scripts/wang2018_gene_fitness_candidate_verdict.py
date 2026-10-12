#!/usr/bin/env python
# experiments/database/scripts/wang2018_gene_fitness_candidate_verdict.py
# [[experiments.database.scripts.wang2018_gene_fitness_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/wang2018_gene_fitness_candidate_verdict.py
"""Gate Wang 2018 (bacteria row "Wang 2018") for its gene-level class (#879).

``candidate-gate gate --row "Wang 2018"`` composes the same gates; this script adds the
loader's sourced statements for the Supplementary Data 5 columns:

- G3: the deposit now includes Supplementary Data 5 (MOESM8), retrieved by the same
  ``pmc_cloud`` route as the other seven files and pinned by sha256.
- G4: the key half; the guide-level class is the same paper's other table, held by the
  same module, so no other module holds the key.
- G5: ``EnvironmentResponsePhenotype`` carries every SD5 column (the Z score on the
  field #879 adds); the compound the curated table lacks (ampicillin) is issue #1063.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/wang2018_gene_fitness_candidate_verdict.py
    python experiments/database/scripts/wang2018_gene_fitness_candidate_verdict.py --write
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
from torchcell.datasets.ecoli import wang2018 as loader

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
ROW: Final = "Wang 2018"
CITATION_KEY: Final = loader.CITATION_KEY
PHENOTYPE_CLASS: Final = "EnvironmentResponsePhenotype"
#: The open curation issue whose list holds ampicillin.
G5_ISSUE: Final = 1063
#: Every compound name either Wang 2018 class stores on an environment.
COMPOUNDS: Final = ("D-glucose", "ampicillin", "furfural", "isobutanol", "kanamycin")
#: The loader's sourced statements attached to the gate each supports.
EVIDENCE: Final = (
    ("G5", loader.GENE_FITNESS_STATISTIC),
    ("G5", loader.GENE_FPR_IS_A_P_VALUE),
    ("G5", loader.GENE_FDR_IS_A_Q_VALUE),
    ("G5", loader.Z_SCALE),
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
