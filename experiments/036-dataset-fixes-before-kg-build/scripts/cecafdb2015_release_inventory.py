# experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.cecafdb2015_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory
"""Settle schedule row 54, the CeCaFDB flux compendium, by measuring its release.

The row is ``status="aggregation"``: 581 published 13C flux maps re-served by a
database. Four questions decide it, each measured on the sha256-pinned deposit in
``$DATA_ROOT/torchcell-raw/zhangCeCaFDBCuratedDatabase2015/``:

1. **Per-source attribution.** Does every in-scope workbook name the reference the
   Download page files it under? (``all_attributed``)
2. **Per-reaction values with intervals.** Does any workbook carry a column beyond the
   per-case fluxes, or any uncertainty term? Does the paper describe one?
3. **Hosts.** How many E. coli and P. putida references and cases, how many cases name
   a genomes-tier strain, how many references hold a single case (no in-release parent
   map for ``FluxExperimentReference``).
4. **Duplication.** The 33 source DOIs against every DOI named anywhere in a bacterial
   loader module (torchcell/datasets/ecoli and pputida, a superset of the served
   papers' DOIs, so an empty intersection is conclusive) and every DOI in the bacteria
   candidate table.

``probe`` records what the accession answers over http and https today, with the
workbook server's ``Last-Modified``.

Writes, under ``results/``: ``cecafdb2015_release_inventory.json``,
``cecafdb2015_workbooks.csv`` and, from ``probe``, ``cecafdb2015_accession_probe.json``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory.py probe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torchcell.datasets.ecoli.zhang2015 import (
    ACCESSION_URL,
    CITATION_KEY,
    ECOLI,
    PAPER_TEXT_REL,
    PAPER_TEXT_SHA256,
    WORKBOOKS,
    raw_mirror_dir,
    release_inventory,
)

RESULTS_REL = "experiments/036-dataset-fixes-before-kg-build/results"
#: Words that would describe an uncertainty in the paper's text.
PAPER_UNCERTAINTY_TERMS = re.compile(
    r"standard deviation|confidence|uncertaint", re.IGNORECASE
)
#: The reference the largest E. coli share comes from, and its own schedule row.
HAVERKORN_DOI = "10.1038/msb.2011.9"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _results_dir() -> Path:
    path = _repo_root() / RESULTS_REL
    path.mkdir(parents=True, exist_ok=True)
    return path


DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)")
CANDIDATE_TABLE = (
    "experiments/database/scripts/build_bacteria_candidate_datasets_table.py"
)
LOADER_DIRS = ("torchcell/datasets/ecoli", "torchcell/datasets/pputida")


def _dois_in(text: str) -> set[str]:
    return {doi.rstrip(").,;:").lower() for doi in DOI.findall(text)}


def loader_dois(repo_root: Path) -> dict[str, set[str]]:
    """Every DOI each bacterial loader module names, this record's module excluded."""
    found: dict[str, set[str]] = {}
    for rel in LOADER_DIRS:
        for path in sorted((repo_root / rel).glob("*.py")):
            if path.name == "zhang2015.py":
                continue
            dois = _dois_in(path.read_text(encoding="utf-8"))
            if dois:
                found[f"{rel}/{path.name}"] = dois
    return found


def candidate_table_dois(repo_root: Path) -> set[str]:
    """Every DOI the bacteria candidate table generator names."""
    return _dois_in((repo_root / CANDIDATE_TABLE).read_text(encoding="utf-8"))


def paper_uncertainty_mentions(raw: Path) -> int:
    """Occurrences of an uncertainty word in the pinned PMC full text."""
    path = raw / PAPER_TEXT_REL
    from torchcell.data import verify_sha256

    verify_sha256(path, PAPER_TEXT_SHA256)
    return len(PAPER_UNCERTAINTY_TERMS.findall(path.read_text(encoding="utf-8")))


def run_inventory() -> dict[str, Any]:
    raw = raw_mirror_dir()
    inventory = release_inventory(raw)
    by_module = loader_dois(_repo_root())
    served_dois = set().union(*by_module.values())
    candidate_dois = candidate_table_dois(_repo_root())
    source_dois = {w.doi.lower() for w in WORKBOOKS}
    haverkorn = [w for w in inventory.workbooks if w.doi.lower() == HAVERKORN_DOI]
    ecoli_cases = inventory.totals[ECOLI].cases
    results: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "measured_at": datetime.now(UTC).isoformat(),
        "release": inventory.model_dump(mode="json"),
        "paper_uncertainty_mentions": paper_uncertainty_mentions(raw),
        "duplication": {
            "n_source_dois": len(source_dois),
            "n_loader_modules_naming_a_doi": len(by_module),
            "n_loader_dois": len(served_dois),
            "source_dois_in_loaders": sorted(source_dois & served_dois),
            "source_dois_in_candidate_table": sorted(source_dois & candidate_dois),
            "haverkorn_cases": haverkorn[0].n_cases,
            "haverkorn_fraction_of_ecoli_cases": haverkorn[0].n_cases / ecoli_cases,
            "haverkorn_strain_label_run": haverkorn[0].longest_strain_label_run,
        },
    }
    out = _results_dir()
    (out / "cecafdb2015_release_inventory.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    rows = [w.model_dump(mode="json") for w in inventory.workbooks]
    with (out / "cecafdb2015_workbooks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            row["units"] = ";".join(row["units"])
            writer.writerow(row)
    return results


def _fetch(url: str, *, method: str = "GET") -> dict[str, Any]:
    """Status, size, title and validators of one URL; a transport failure is recorded."""
    context = ssl.create_default_context()
    request = urllib.request.Request(url, method=method)
    record: dict[str, Any] = {"url": url, "method": method}
    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            body = response.read()
            record["status"] = response.status
            record["bytes"] = len(body)
            record["last_modified"] = response.headers.get("Last-Modified")
            title = re.search(rb"<title>(.*?)</title>", body, re.S)
            record["title"] = (
                title.group(1).decode("utf-8", "replace").strip() if title else None
            )
    except urllib.error.HTTPError as error:
        record["status"] = error.code
    except (urllib.error.URLError, TimeoutError) as error:
        record["status"] = None
        record["reason"] = str(error)
    return record


def probe_accession() -> dict[str, Any]:
    https_root = ACCESSION_URL.replace("http://", "https://")
    return {
        "accession_url": ACCESSION_URL,
        "probed_at": datetime.now(UTC).isoformat(),
        "probes": [
            _fetch(ACCESSION_URL + "/"),
            _fetch(https_root + "/"),
            _fetch(ACCESSION_URL + "/download_action"),
            _fetch(WORKBOOKS[0].source_url, method="HEAD"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="inventory", choices=("inventory", "probe")
    )
    if parser.parse_args().action == "probe":
        results = probe_accession()
        path = _results_dir() / "cecafdb2015_accession_probe.json"
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        for row in results["probes"]:
            print(f"  {row['url']:72s} {row.get('status')} {row.get('title') or ''}")
        print(f"wrote {path}")
        return
    results = run_inventory()
    release = results["release"]
    print(f"CeCaFDB 2015 ({CITATION_KEY})")
    print(f"  Download page references : {release['n_index_references']}")
    for species, totals in release["totals"].items():
        print(f"  {species:24s} : {totals}")
    print(f"  workbooks with interval  : {release['n_workbooks_with_interval']}")
    print(f"  units                    : {release['units']}")
    print(f"  all attributed           : {release['all_attributed']}")
    print(f"  paper uncertainty words  : {results['paper_uncertainty_mentions']}")
    print(f"  duplication              : {results['duplication']}")


if __name__ == "__main__":
    main()
