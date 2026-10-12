#!/usr/bin/env python
# experiments/database/scripts/yoshikawa2009_candidate_verdict.py
# [[experiments.database.scripts.yoshikawa2009_candidate_verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/yoshikawa2009_candidate_verdict.py
"""Re-gate Yoshikawa 2009 (off-schedule, so ``candidate-gate gate`` has no row for it).

The yeast Zotero audit (``yeast_zotero_collection_audit.py``) built this key's row from
its ``RowSpec`` and left G3 ``unmeasured``: no manifest listed anything under ``si/`` or
``data/``, because the publisher's Table S1 is walled to scripts. The owner has since
deposited it by hand, so three things the gates read have changed: the raw mirror's
``manifest.json`` lists the publisher ``.xls`` and the YeastPhenome ``.xlsx`` (G3
inventories them off the bytes), ``yoshikawa2009.py`` declares the DOI (G4 reads it as
the key's own module), and the two stress compounds can be named (G5 checks that ethanol
and sodium chloride resolve). This script re-runs the audit's gate composition for this
one entry with those inputs, the CLI's PMID inputs, and the loader's measured evidence,
and with ``--write`` replaces the stored verdict.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), after committing, so the
verdict records a clean commit::

    python experiments/database/scripts/yoshikawa2009_candidate_verdict.py          # print
    python experiments/database/scripts/yoshikawa2009_candidate_verdict.py --write  # store
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
from torchcell.candidates.cli import load_doi_to_pmid, load_table_module, table_rows
from torchcell.candidates.gates import GateRun
from torchcell.datasets.scerevisiae import yoshikawa2009 as loader

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
AUDIT_SCRIPT: Final = SCRIPT.with_name("yeast_zotero_collection_audit.py")
CITATION_KEY: Final = loader.CITATION_KEY
#: Every compound label the loader passes to the resolver.
COMPOUNDS: Final = tuple(
    sorted(
        spec.compound_label
        for spec in loader.CONDITIONS
        if spec.compound_label is not None
    )
)
#: The loader's sourced statements that support G3 (the per-strain release) and G5.
EXTRA_EVIDENCE: Final = (
    ("G3", loader.TABLE_S1_IS_NORMALIZED),
    ("G5", loader.TWO_CULTURES),
    ("G5", loader.TABLE1_IS_SAMPLE_SD),
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

    The stop rule and the composition are the audit's (``gate_entry``); the differences
    are G4's PMID inputs, which are the CLI's, and G5, which is given the loader's
    compound labels.
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
            row,
            CITATION_KEY,
            index,
            doi_to_pmid=load_doi_to_pmid(),
            aggregator_pmids=gates.aggregator_pmid_sets(root),
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
