#!/usr/bin/env python
# experiments/database/scripts/essentiality_evidence_924_candidate_verdicts.py
# [[experiments.database.scripts.essentiality_evidence_924_candidate_verdicts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/essentiality_evidence_924_candidate_verdicts.py
"""Re-gate G5 of Baba 2006 and Gerdes 2003 now that #924 types the essentiality evidence.

Both verdicts recorded G5 as a gap on #924 (``admissible_with_gaps``): the phenotype
class existed but its one boolean carried no evidence type, and the PCR counts, the
assertion error and the calls' meaning had no field. Since #924 every record carries a
``GeneEssentialityEvidence``. G1 to G4 read nothing this change touched (the rows, the
mirrors, the assemblies, the module index and the content comparison), so they are kept
as stored, with their hand-written reasons; G5 is re-run (``gates.evaluate_g5``) and its
evidence is the loader's own sourced statements of the rule, the definition and the
caveat, so the verdict quotes the same bytes the records do.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/essentiality_evidence_924_candidate_verdicts.py
    python experiments/database/scripts/essentiality_evidence_924_candidate_verdicts.py --write
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from types import ModuleType
from typing import Final

from torchcell.candidates import gates, store
from torchcell.candidates.verdict import CandidateVerdict, GateResult
from torchcell.datamodels.schema import EssentialityStatementRole
from torchcell.datasets.ecoli import baba2006, gerdes2003

#: The loaders whose verdicts carried the #924 gap, and what their evidence is.
LOADERS: Final[tuple[ModuleType, ...]] = (baba2006, gerdes2003)
PHENOTYPE_CLASS: Final = "GeneEssentialityPhenotype"


def git_commit() -> str:
    """The 40-hex commit the gates read."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def regated(verdict: CandidateVerdict, loader: ModuleType) -> CandidateVerdict:
    """``verdict`` with G5 re-run and its reason and evidence stating the descriptor."""
    run = gates.evaluate_g5(PHENOTYPE_CLASS, (), issue=None)
    evidence = loader.evidence()
    statements = [
        *evidence.statements_of(EssentialityStatementRole.scoring_rule),
        *evidence.statements_of(EssentialityStatementRole.call_definition),
        *evidence.statements_of(EssentialityStatementRole.caveat),
    ]
    g5 = GateResult(
        gate="G5",
        outcome=run.result.outcome,
        reason=f"{run.result.reason}; since #924 every record carries a "
        f"GeneEssentialityEvidence of type {evidence.evidence_type.value} (class "
        f"{evidence.evidence_class.value}): the paper's scoring rule, its definition of "
        "the call and its caveat, quoted with sha256, plus the per-record statistics "
        "the release carries",
        evidence=statements,
        issue=None,
        blocked_kind=run.result.blocked_kind,
    )
    data = verdict.model_dump()
    data.update(
        gates=[
            g5.model_dump() if g.gate == "G5" else g.model_dump() for g in verdict.gates
        ],
        g5=run.record.model_dump(),
        decided_at=date.today().isoformat(),
        torchcell_commit=git_commit(),
    )
    return CandidateVerdict.model_validate(data)


def main() -> None:
    """Re-gate G5 of both verdicts; print, or with --write store them."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for loader in LOADERS:
        verdict = regated(store.read_verdict(loader.CITATION_KEY), loader)
        print(f"{verdict.citation_key}: {verdict.outcome}")
        for g in verdict.gates:
            print(f"  {g.gate} {g.outcome}: {g.reason[:160]}")
        if args.write:
            print(f"wrote {store.write_verdict(verdict)}")


if __name__ == "__main__":
    main()
