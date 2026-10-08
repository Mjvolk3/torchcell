# experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.mcf2chem2023_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory
"""Settle schedule row 46, MCF2Chem 2023, by measuring what the release contains.

The row is ``status="aggregation"``: a knowledge base whose 8,888 production records are
values COLLECTED FROM OTHER PAPERS. Three questions decide it, and each is measured on
the sha256-pinned mirror bytes rather than read off the abstract.

1. **Is there a per-record artifact at all?** The row's own status vocabulary says a row
   whose per-record values are not released is ``blocked``, so the first measurement is
   an inventory: every table of every pinned SI file, its header cells and its row count,
   and an explicit search for a production table (one whose header names titer, yield,
   productivity, content, strain or compound). The paper's Availability of data and
   materials names one channel, the web server, so ``probe`` records that channel's live
   HTTP status beside the inventory.

2. **Does the row duplicate what we already serve?** The duplication surface is the
   phenotype MCF2Chem carries, a product titer, so this counts the served bacterial
   dataset classes whose ``experiment_class`` is ``ProductTiterExperiment`` and tests
   each against MCF2Chem's extraction window (reviews published 2017-08-01 to
   2022-07-31): a primary paper published after that window cannot be in the release.
   It also intersects the 268 review DOIs of Table S1 with the DOI of every served
   bacterial dataset and every row of the bacteria candidate table, which is the direct
   test of whether MCF2Chem's source corpus and ours are the same papers.

3. **Can ``ProductTiterPhenotype`` hold a row of it?** Each required field of the
   phenotype and of ``ProductTiterExperimentReference`` is checked against a verbatim
   statement of the release's own limits, read from the pinned ``paper.md``.

Writes, under ``results/``: ``mcf2chem2023_release_inventory.json``,
``mcf2chem2023_table_s1_reviews.csv`` (the 268 review DOIs),
``mcf2chem2023_titer_surface.csv`` and, from ``probe``,
``mcf2chem2023_accession_probe.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory.py probe
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import os.path as osp
import pkgutil
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from torchcell.data import verify_sha256
from torchcell.datasets.ecoli.cai2023 import (
    ACCESSION_URL,
    CITATION_KEY,
    PAPER_MD,
    PAPER_MD_SHA256,
    PROBE_PATHS,
    PRODUCTION_HEADER_TERMS,
    REVIEW_WINDOW_END,
    REVIEW_WINDOW_START,
    SI_FILES,
    docx_tables,
    library_dir,
    release_inventory,
)

RESULTS_REL = "experiments/036-dataset-fixes-before-kg-build/results"
#: Every row of the bacteria candidate table carries its paper's DOI in ``url``.
CANDIDATE_TABLE = (
    "experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)
DOI_IN_URL = re.compile(r"doi\.org/(10\.[^\s\"']+)")


def _data_root() -> str:
    load_dotenv()
    root = os.getenv("DATA_ROOT")
    if root is None:
        raise RuntimeError("DATA_ROOT is not set")
    return root


def _results_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / RESULTS_REL
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_s1_reviews(library: Path) -> list[dict[str, str]]:
    """Every (title, DOI) row of Additional file 1: Table S1."""
    tables = docx_tables(library / SI_FILES["si1"].relpath)
    if len(tables) != 1:
        raise ValueError(f"Table S1 holds {len(tables)} tables, expected 1")
    rows = tables[0]
    header = [cell.lower() for cell in rows[0]]
    if header != ["review_title", "review_doi"]:
        raise ValueError(f"unexpected Table S1 header: {rows[0]}")
    return [{"review_title": row[0], "review_doi": row[1]} for row in rows[1:]]


def candidate_table_dois(repo_root: Path) -> set[str]:
    """Every DOI the bacteria candidate table names in a row's ``url``."""
    source = (repo_root / CANDIDATE_TABLE).read_text(encoding="utf-8")
    return {doi.rstrip(").,").lower() for doi in DOI_IN_URL.findall(source)}


def served_bacterial_papers(library_root: Path) -> dict[str, dict[str, str | None]]:
    """Citation key and mirrored DOI of every served bacterial dataset class."""
    import torchcell.datasets.ecoli as ecoli
    import torchcell.datasets.pputida as pputida

    for pkg in (ecoli, pputida):
        for found in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg.__name__}.{found.name}")

    from tests.torchcell.knowledge_graphs.test_build_time_projection import (
        BACTERIAL_DATASETS,
    )
    from torchcell.datasets.dataset_registry import dataset_registry

    served: dict[str, dict[str, str | None]] = {}
    for name in sorted(BACTERIAL_DATASETS):
        cls = dataset_registry[name]
        module = sys.modules[cls.__module__]
        # The three Schmidt 2016 arm modules take the key from the paper module they
        # import, so they name it only on their ``PAPER`` provenance.
        key: str = getattr(module, "CITATION_KEY", None) or str(
            getattr(module, "PAPER").citation_key
        )
        manifest = json.loads(
            (library_root / key / "manifest.json").read_text(encoding="utf-8")
        )
        experiment = inspect.getsource(
            inspect.getattr_static(cls, "experiment_class").fget
        )
        returned = [
            line.strip().removeprefix("return ").strip()
            for line in experiment.splitlines()
            if line.strip().startswith("return ")
        ]
        served[name] = {
            "module": cls.__module__.split(".")[-1],
            "citation_key": key,
            "doi": (manifest.get("doi") or "").lower() or None,
            "experiment_class": returned[0],
        }
    return served


def titer_surface(served: dict[str, dict[str, str | None]]) -> list[dict[str, Any]]:
    """The served titer classes, each tested against MCF2Chem's extraction window.

    A citation key ends in its paper's year, and the window closes on 2022-07-31, so a
    key whose year is 2023 or later names a paper MCF2Chem's reviews could not have
    summarized. A key of 2022 or earlier is in-window and is a genuine duplication
    candidate, which is what the measurement is for.
    """
    cutoff_year = int(REVIEW_WINDOW_END[:4])
    rows: list[dict[str, Any]] = []
    for name, record in sorted(served.items()):
        if record["experiment_class"] != "ProductTiterExperiment":
            continue
        key = str(record["citation_key"])
        year_match = re.search(r"(\d{4})$", key)
        if year_match is None:
            raise ValueError(f"citation key does not end in a year: {key}")
        year = int(year_match.group(1))
        rows.append(
            {
                "dataset_class": name,
                "citation_key": key,
                "doi": record["doi"],
                "paper_year_from_key": year,
                "in_mcf2chem_window": year <= cutoff_year,
            }
        )
    return rows


def production_tables(library: Path) -> list[dict[str, Any]]:
    """Every table of every pinned SI file, flagged for production-record headers."""
    found: list[dict[str, Any]] = []
    for name, record in SI_FILES.items():
        for index, table in enumerate(docx_tables(library / record.relpath)):
            header = [cell.strip().lower() for cell in table[0]]
            hits = sorted(
                term
                for term in PRODUCTION_HEADER_TERMS
                if any(term in cell for cell in header)
            )
            found.append(
                {
                    "si_file": name,
                    "relpath": record.relpath,
                    "table_index": index,
                    "n_rows": len(table) - 1,
                    "n_columns": len(header),
                    "header": table[0],
                    "production_header_terms": hits,
                }
            )
    return found


def probe_accession() -> dict[str, Any]:
    """HTTP status of the accession root and of every candidate data path.

    A status of ``None`` with a ``reason`` is a transport failure, not a server answer,
    and is recorded as such: the point of the probe is what the one named distribution
    channel answers today, so a timeout is a finding rather than something to retry past.
    """
    probed: list[dict[str, Any]] = []
    for path in PROBE_PATHS:
        url = ACCESSION_URL.rstrip("/") + path
        request = urllib.request.Request(url, method="GET")
        status: int | None = None
        reason: str | None = None
        length: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                length = len(response.read())
        except urllib.error.HTTPError as error:
            status = error.code
            length = len(error.read())
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        probed.append(
            {
                "path": path or "/",
                "url": url,
                "status": status,
                "bytes": length,
                "reason": reason,
            }
        )
    return {
        "accession_url": ACCESSION_URL,
        "probed_at": datetime.now(UTC).isoformat(),
        "paths": probed,
    }


def run_inventory() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    data_root = _data_root()
    library = library_dir(data_root)
    library_root = library.parent

    verify_sha256(library / PAPER_MD, PAPER_MD_SHA256)
    for record in SI_FILES.values():
        verify_sha256(library / record.relpath, record.sha256)

    inventory = release_inventory(library)
    reviews = table_s1_reviews(library)
    served = served_bacterial_papers(library_root)
    surface = titer_surface(served)

    review_dois = {row["review_doi"].strip().lower() for row in reviews}
    served_dois = {
        str(record["doi"]) for record in served.values() if record["doi"] is not None
    }
    candidate_dois = candidate_table_dois(repo_root)

    results: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "measured_at": datetime.now(UTC).isoformat(),
        "release": inventory.model_dump(mode="json"),
        "si_tables": production_tables(library),
        "table_s1": {"n_reviews": len(reviews), "n_distinct_dois": len(review_dois)},
        "source_corpus_overlap": {
            "n_review_dois": len(review_dois),
            "n_served_bacterial_dois": len(served_dois),
            "n_candidate_table_dois": len(candidate_dois),
            "review_dois_in_served": sorted(review_dois & served_dois),
            "review_dois_in_candidate_table": sorted(review_dois & candidate_dois),
        },
        "titer_surface": {
            "n_served_bacterial_classes": len(served),
            "n_product_titer_classes": len(surface),
            "window_start": REVIEW_WINDOW_START,
            "window_end": REVIEW_WINDOW_END,
            "n_in_window": sum(1 for row in surface if row["in_mcf2chem_window"]),
            "in_window": [
                row["dataset_class"] for row in surface if row["in_mcf2chem_window"]
            ],
            "rows": surface,
        },
    }

    out = _results_dir()
    (out / "mcf2chem2023_release_inventory.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    with (out / "mcf2chem2023_table_s1_reviews.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_title", "review_doi"])
        writer.writeheader()
        writer.writerows(reviews)
    with (out / "mcf2chem2023_titer_surface.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(surface[0]))
        writer.writeheader()
        writer.writerows(surface)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="inventory",
        choices=("inventory", "probe"),
        help="inventory measures the pinned mirror; probe reads the live accession",
    )
    action = parser.parse_args().action

    if action == "probe":
        results = probe_accession()
        path = _results_dir() / "mcf2chem2023_accession_probe.json"
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        for row in results["paths"]:
            print(f"  {row['path']:24s} -> {row['status']} {row['reason'] or ''}")
        print(f"wrote {path}")
        return

    results = run_inventory()
    release = results["release"]
    print(f"MCF2Chem 2023 ({CITATION_KEY})")
    print(f"  SI files pinned          : {len(release['si_files'])}")
    print(f"  production record tables : {release['n_production_tables']}")
    print(f"  Table S1 reviews         : {results['table_s1']['n_reviews']}")
    overlap = results["source_corpus_overlap"]
    print(
        "  review DOIs in served    : "
        f"{len(overlap['review_dois_in_served'])}/{overlap['n_review_dois']}"
    )
    print(
        "  review DOIs in schedule  : "
        f"{len(overlap['review_dois_in_candidate_table'])}/{overlap['n_review_dois']}"
    )
    surface = results["titer_surface"]
    print(
        f"  served titer classes     : {surface['n_product_titer_classes']}"
        f" of {surface['n_served_bacterial_classes']}"
    )
    print(
        f"  in MCF2Chem window       : {surface['n_in_window']} {surface['in_window']}"
    )
    print(f"wrote {osp.join(RESULTS_REL, 'mcf2chem2023_release_inventory.json')}")


if __name__ == "__main__":
    main()
