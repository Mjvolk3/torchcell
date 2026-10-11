#!/usr/bin/env python
# experiments/database/scripts/messner2021_candidate_verdict.py
# [[experiments.database.scripts.messner2021_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/messner2021_candidate_verdict.py
"""Re-gate Messner 2021 (off-schedule, so ``candidate-gate gate`` has no row for it).

The yeast Zotero audit (``yeast_zotero_collection_audit.py``) built this key's row from
its ``RowSpec`` and gated it with no compounds named and no raw mirror. Once the loader
lands, three things the gates read have changed: the raw mirror holds the DIA-NN report
(G3 inventories it), ``messner2021.py`` declares the DOI (G4 reads it as the key's own
module), and the 18 compounds plus DMSO can be named (G5 checks that every one resolves).
This script re-runs the audit's own ``gate_entry`` for this one entry with those inputs
and the loader's measured evidence, and with ``--write`` replaces the stored verdict.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/messner2021_candidate_verdict.py          # print
    python experiments/database/scripts/messner2021_candidate_verdict.py --write  # store
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Final

from dotenv import load_dotenv

from torchcell.candidates import gates, store
from torchcell.candidates.cli import load_table_module, table_rows
from torchcell.candidates.gates import GateRun
from torchcell.datasets.scerevisiae import messner2021 as loader

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
AUDIT_SCRIPT: Final = SCRIPT.with_name("yeast_zotero_collection_audit.py")
CITATION_KEY: Final = loader.CITATION_KEY
#: Every compound label the loader passes to the resolver.
COMPOUNDS: Final = tuple(
    sorted({spec.label for spec in loader.AGENTS.values()} | {"DMSO"})
)
#: The loader's measured statements that support G3 (a processed per-run release) and G5.
EXTRA_EVIDENCE: Final = (
    ("G3", loader.SOURCED["diann"]),
    ("G3", loader.SOURCED["filters"]),
    ("G5", loader.SOURCED["contrast"]),
)


def audit_module() -> ModuleType:
    """The audit script, loaded by path (``experiments/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("yeast_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Gate the one entry with the loader's inputs; print, or with --write store it.

    The stop rule and the composition are the audit's ``gate_entry``; the only
    difference is G5, which is given the loader's compound labels.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO / ".env")
    root = os.environ["DATA_ROOT"]
    audit = audit_module()
    entry = next(e for e in audit.AUDIT if e.citation_key == CITATION_KEY)
    entry = entry.model_copy(update={"evidence": (*entry.evidence, *EXTRA_EVIDENCE)})
    rows = {r.name: r for r in table_rows("yeast", load_table_module("yeast"))}
    row = audit.candidate_row(entry, rows)
    index = gates.module_index()
    g1 = gates.evaluate_g1(row, entry.finding)
    steps: dict[str, Callable[[], GateRun]] = {
        "G2": lambda: gates.evaluate_g2(
            row, root, source_kind=g1.source_kind, verify=False
        ),
        "G3": lambda: gates.evaluate_g3(gates.inventory(CITATION_KEY, root)),
        "G4": lambda: gates.evaluate_g4_key(
            row, CITATION_KEY, index, doi_to_pmid=None, aggregator_pmids=None
        ),
        "G5": lambda: gates.evaluate_g5(entry.phenotype_class, COMPOUNDS, issue=None),
    }
    runs: dict[str, GateRun] = {}
    stopped = g1.result.outcome in ("fail", "blocked")
    for name, step in steps.items():
        if stopped:
            break
        runs[name] = step()
        stopped = runs[name].result.outcome in ("fail", "blocked")
    verdict = audit.with_evidence(
        gates.compose_verdict(
            row=row,
            citation_key=CITATION_KEY,
            g1=g1,
            runs=runs,  # type: ignore[arg-type]
            zotero=gates.zotero_item(CITATION_KEY, root),
            decided_at=date.today().isoformat(),
            torchcell_commit=audit.git_commit(),
        ),
        entry,
    )
    print(f"{CITATION_KEY}: {verdict.outcome}")
    for g in verdict.gates:
        print(f"  {g.gate} {g.outcome}: {g.reason}")
    if args.write:
        print(f"wrote {store.write_verdict(verdict)}")


if __name__ == "__main__":
    main()
