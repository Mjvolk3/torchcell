# experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.d2cell2026_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory
"""Settle schedule row 60, D2Cell 2026, by measuring what its deposit holds.

``deposit`` runs each file's recorded retriever, verifies the sha256 and writes the raw
mirror ``$DATA_ROOT/torchcell-raw/liLeveragingLargeLanguage2024/``. ``inventory``
(default) reads the pinned bytes and measures:

1. the LLM-extracted database workbook (E. coli rows, source DOIs, titer units, the
   field-misalignment signatures),
2. the D2Cell-pred E. coli split files (real/simulated flag, labels, DOIs),
3. the overlap of the E. coli source DOIs with every served bacterial dataset DOI and
   every row of the bacteria candidate table,
4. that every quote in ``torchcell.datasets.ecoli.li2024`` is a substring of the pinned
   ``paper.md``.

Writes ``results/d2cell2026_release_inventory.json`` and
``results/d2cell2026_ecoli_source_dois.csv`` (the per-DOI lead list).

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory.py deposit
    python experiments/036-dataset-fixes-before-kg-build/scripts/d2cell2026_release_inventory.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pkgutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from torchcell.data import verify_sha256, write_verified
from torchcell.datasets.ecoli import li2024
from torchcell.literature.provenance import run_retriever
from torchcell.verification.sourced import SourcedValue

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "experiments/036-dataset-fixes-before-kg-build/results"
MCF2CHEM_SCRIPT = (
    REPO / "experiments/036-dataset-fixes-before-kg-build/scripts/"
    "mcf2chem2023_release_inventory.py"
)


def _mcf2chem_helpers() -> Any:
    """The candidate-table DOI reader the row-46 script already defines."""
    spec = importlib.util.spec_from_file_location("mcf2chem_inventory", MCF2CHEM_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _citation_key(module: ModuleType) -> str:
    """A served module's key: its own, its ``PAPER``'s, or its paper module's."""
    if hasattr(module, "CITATION_KEY"):
        return str(module.CITATION_KEY)
    if hasattr(module, "PAPER"):
        return str(module.PAPER.citation_key)
    parents = [
        value
        for value in vars(module).values()
        if isinstance(value, ModuleType)
        and value.__name__.startswith("torchcell.datasets.")
        and hasattr(value, "CITATION_KEY")
    ]
    if len(parents) != 1:
        raise ValueError(f"{module.__name__}: cannot resolve one citation key")
    return str(parents[0].CITATION_KEY)


def served_bacterial_dois(library_root: Path) -> dict[str, str | None]:
    """Mirrored DOI of every served bacterial dataset class, by class name."""
    import torchcell.datasets.ecoli as ecoli
    import torchcell.datasets.pputida as pputida

    for pkg in (ecoli, pputida):
        for found in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{found.name}")

    from tests.torchcell.knowledge_graphs.test_build_time_projection import (
        BACTERIAL_DATASETS,
    )
    from torchcell.datasets.dataset_registry import dataset_registry

    out: dict[str, str | None] = {}
    for name in sorted(BACTERIAL_DATASETS):
        key = _citation_key(sys.modules[dataset_registry[name].__module__])
        # A paper in neither Zotero library has only a raw-mirror manifest.
        library_manifest = library_root / key / "manifest.json"
        raw_manifest = library_root.parent / "torchcell-raw" / key / "manifest.json"
        path = library_manifest if library_manifest.exists() else raw_manifest
        manifest = json.loads(path.read_text(encoding="utf-8"))
        out[name] = (manifest.get("doi") or "").lower() or None
    return out


def deposit() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        sources: dict[str, Path] = {}
        for raw in li2024.RAW_FILES:
            path = Path(tmp) / raw.name
            write_verified(
                run_retriever(raw.retrieval), path, raw.sha256, raw.source_url
            )
            sources[raw.name] = path
        return li2024.deposit_raw_mirror(sources=sources)


def _norm_doi(doi: object) -> str:
    return str(doi).strip().lower()


def audit_quotes() -> dict[str, bool]:
    paper = li2024.library_dir() / li2024.PAPER_MD
    verify_sha256(paper, li2024.PAPER_MD_SHA256)
    text = paper.read_text(encoding="utf-8")
    return {
        name: value.quote in text
        for name, value in vars(li2024).items()
        if isinstance(value, SourcedValue) and value.quote is not None
    }


def inventory() -> dict[str, Any]:
    database = li2024.read_database()
    splits = li2024.read_splits()
    db = li2024.measure_database(database)
    split = li2024.measure_training_split(splits)
    leads = li2024.ecoli_source_dois(database)

    served = served_bacterial_dois(li2024.library_dir().parent)
    served_dois = {doi for doi in served.values() if doi is not None}
    candidate_dois = _mcf2chem_helpers().candidate_table_dois(REPO)
    lead_dois = {_norm_doi(d) for d in leads["doi"].dropna()}

    quotes = audit_quotes()
    if not all(quotes.values()):
        raise ValueError(
            f"quotes not in paper.md: {[k for k, v in quotes.items() if not v]}"
        )

    results: dict[str, Any] = {
        "citation_key": li2024.CITATION_KEY,
        "measured_at": datetime.now(UTC).isoformat(),
        "raw_files": [
            {
                "name": r.name,
                "sha256": r.sha256,
                "bytes": r.bytes,
                "source_url": r.source_url,
            }
            for r in li2024.RAW_FILES
        ],
        "database": db.model_dump(mode="json"),
        "training_split": split.model_dump(mode="json"),
        "source_doi_overlap": {
            "n_ecoli_source_dois": len(lead_dois),
            "n_served_bacterial_classes": len(served),
            "n_served_bacterial_dois": len(served_dois),
            "n_candidate_table_dois": len(candidate_dois),
            "in_served": sorted(lead_dois & served_dois),
            "in_candidate_table": sorted(lead_dois & candidate_dois),
        },
        "quote_audit": quotes,
        "schema_blockers": [b.model_dump(mode="json") for b in li2024.SCHEMA_BLOCKERS],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "d2cell2026_release_inventory.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    leads.sort_values(["n_entries", "doi"], ascending=[False, True]).to_csv(
        RESULTS / "d2cell2026_ecoli_source_dois.csv", index=False
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="inventory", choices=("inventory", "deposit")
    )
    if parser.parse_args().action == "deposit":
        print(f"deposited {deposit()}")
        return
    results = inventory()
    print(
        json.dumps({k: results[k] for k in ("database", "training_split")}, indent=1)[
            :4000
        ]
    )
    overlap = results["source_doi_overlap"]
    print(
        "served overlap",
        overlap["in_served"],
        "candidate overlap",
        overlap["in_candidate_table"],
    )


if __name__ == "__main__":
    main()
