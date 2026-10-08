# experiments/035-benchmark-bundles/scripts/sgd_phenotype_mirror.py
# [[experiments.035-benchmark-bundles.scripts.sgd_phenotype_mirror]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-benchmark-bundles/scripts/sgd_phenotype_mirror
r"""Mirror the SGD source files the gene essentiality benchmark is labeled from.

Two kinds of file, both from the Saccharomyces Genome Database, both kept verbatim with
a sha256 and the exact retrieval in ``provenance.json`` beside them, so the bundle
script needs no network and a rebuild can verify every byte it consumed:

* ``SGD_features.tab``: the chromosomal feature table. Its rows with feature type
  ``ORF`` (qualifiers Verified, Uncharacterized, Dubious) are the gene universe.
* ``phenotype_details/<ORF>.json``: the locus phenotype annotations from the SGD
  backend (``/backend/locus/<ORF>/phenotype_details``), one file per ORF. The label
  rule reads the ``null`` mutant-type rows in strain ``S288C``: ``inviable`` is the
  essential call ``GeneEssentialitySgdDataset`` already stores, ``viable`` is the
  non-essential call this benchmark adds. The same endpoint is what
  ``torchcell.graph.sgd`` fetched for the served dataset.

A file already in the mirror is kept, never re-fetched (SGD is live and changes; the
bytes the bundle was built from are the record). ``--refresh`` moves the mirror aside
first. Failures are retried, then listed; the run exits non-zero if any ORF is missing,
so a partial mirror is never silently used.

Run from the repo root::

    python experiments/035-benchmark-bundles/scripts/sgd_phenotype_mirror.py \\
        --mirror <DATA_ROOT>/data/torchcell/torchcell-raw/cherrySGDSaccharomycesGenome1998
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

FEATURES_URL = (
    "http://sgd-archive.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab"
)
LOCUS_URL = "https://www.yeastgenome.org/backend/locus/{orf}/phenotype_details"
ORF_QUALIFIERS = ("Verified", "Uncharacterized", "Dubious")
USER_AGENT = "torchcell-benchmark-mirror/1 (+https://github.com/Mjvolk3/torchcell)"


class SourceFile(BaseModel):
    """One mirrored file: where it came from, how, and what bytes arrived."""

    path: str
    role: Literal["feature_table", "phenotype_details"]
    source_url: str
    retrieval_method: Literal["direct_url"] = "direct_url"
    retrieval_command: str
    retrieved_at: datetime
    sha256: str
    bytes: int


class MirrorProvenance(BaseModel):
    """``provenance.json``: every file in the mirror and the ORF universe it covers."""

    script: str = "experiments/035-benchmark-bundles/scripts/sgd_phenotype_mirror.py"
    features: SourceFile
    n_orfs: int
    orf_qualifiers: dict[str, int]
    phenotype_details: list[SourceFile]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, accept: str, attempts: int = 4) -> bytes:
    """GET ``url``; retried with backoff on network errors and 5xx."""
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, headers={"accept": accept, "user-agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data: bytes = response.read()
                return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def orfs_from_features(table: bytes) -> dict[str, str]:
    """``{systematic name: qualifier}`` for the ORF rows of ``SGD_features.tab``."""
    out: dict[str, str] = {}
    text = table.decode("utf-8", errors="strict")
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(row) < 4 or row[1] != "ORF":
            continue
        if row[2] not in ORF_QUALIFIERS:
            raise ValueError(f"unexpected ORF qualifier {row[2]!r} on {row[3]}")
        out[row[3]] = row[2]
    return out


def source_file(
    path: Path, mirror: Path, role: str, url: str, accept: str, retrieved_at: datetime
) -> SourceFile:
    data = path.read_bytes()
    return SourceFile(
        path=str(path.relative_to(mirror)),
        role=role,  # type: ignore[arg-type]
        source_url=url,
        retrieval_command=f"curl -sS -H 'accept: {accept}' '{url}' -o '{path.name}'",
        retrieved_at=retrieved_at,
        sha256=sha256_bytes(data),
        bytes=len(data),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    mirror: Path = args.mirror
    if args.refresh and mirror.exists():
        aside = mirror.with_name(f"{mirror.name}.{datetime.now(UTC):%Y%m%dT%H%M%S}")
        mirror.rename(aside)
        print(f"previous mirror moved to {aside}")
    details_dir = mirror / "phenotype_details"
    details_dir.mkdir(parents=True, exist_ok=True)

    features_path = mirror / "SGD_features.tab"
    features_at = datetime.now(UTC)
    if not features_path.exists():
        features_path.write_bytes(fetch(FEATURES_URL, "text/tab-separated-values"))
    features = source_file(
        features_path,
        mirror,
        "feature_table",
        FEATURES_URL,
        "text/tab-separated-values",
        features_at,
    )
    orfs = orfs_from_features(features_path.read_bytes())
    qualifiers: dict[str, int] = {}
    for q in orfs.values():
        qualifiers[q] = qualifiers.get(q, 0) + 1
    print(f"{len(orfs)} ORFs in {features_path.name}: {qualifiers}")

    todo = [orf for orf in sorted(orfs) if not (details_dir / f"{orf}.json").exists()]
    print(f"{len(orfs) - len(todo)} already mirrored, fetching {len(todo)}")
    failed: dict[str, str] = {}

    def one(orf: str) -> str:
        data = fetch(LOCUS_URL.format(orf=orf), "application/json")
        json.loads(data)  # refuse a non-JSON body (an HTML error page)
        (details_dir / f"{orf}.json").write_bytes(data)
        return orf

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, orf): orf for orf in todo}
        for n, future in enumerate(as_completed(futures), 1):
            orf = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001  # every failure is listed and fails the run
                failed[orf] = str(exc)[:200]
            if n % 500 == 0:
                print(f"  {n}/{len(todo)} after {time.monotonic() - started:.0f}s")
    retrieved_at = datetime.now(UTC)

    details = [
        source_file(
            details_dir / f"{orf}.json",
            mirror,
            "phenotype_details",
            LOCUS_URL.format(orf=orf),
            "application/json",
            retrieved_at,
        )
        for orf in sorted(orfs)
        if (details_dir / f"{orf}.json").exists()
    ]
    provenance = MirrorProvenance(
        features=features,
        n_orfs=len(orfs),
        orf_qualifiers=qualifiers,
        phenotype_details=details,
    )
    (mirror / "provenance.json").write_text(
        provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(details)} phenotype files recorded in {mirror / 'provenance.json'}")
    if failed:
        print(f"{len(failed)} ORFs failed:", file=sys.stderr)
        for orf, why in sorted(failed.items()):
            print(f"  {orf}: {why}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
