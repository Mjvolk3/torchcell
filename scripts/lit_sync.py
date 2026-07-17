#!/usr/bin/env python
# scripts/lit_sync.py
# [[scripts.lit_sync]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_sync.py

"""Nightly sync: mirror + OCR new papers in the Zotero collections we track.

Diffs each named Zotero collection (default: ``database`` + ``paper``) against
``<DATA_ROOT>/torchcell-library/`` and captures (download PDF -> MinerU OCR ->
manifest) any paper present in Zotero but missing from the mirror. Idempotent:
already-mirrored papers are skipped, so re-running only picks up what is new. A
key that appears in both collections is captured once (the second pass sees it as
``present``), so ordering the collections never double-downloads a paper.

Supersedes ``scripts/lit_sync_database.py`` (database-only). Designed to run from
cron on GilaHyper (GPU host for MinerU). Self-flocks so a nightly run never
overlaps an on-demand run. Writes one timestamped JSON report per collection under
``<DATA_ROOT>/torchcell-library/_sync_reports/`` and appends a one-line summary to
the log.

Usage::

    python scripts/lit_sync.py                       # sync database + paper (+ OCR)
    python scripts/lit_sync.py --collection paper     # just the paper collection
    python scripts/lit_sync.py --dry-run              # report the gap, capture nothing
    python scripts/lit_sync.py --limit 5              # cap captures per collection
    python scripts/lit_sync.py --no-ocr               # download only, skip OCR
"""

import argparse
import fcntl
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root from this file so cron's cwd does not matter.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from torchcell.literature.backfill import library_root  # noqa: E402
from torchcell.literature.sync import (  # noqa: E402
    DEFAULT_COLLECTIONS,
    SyncMode,
    SyncReport,
    sync_collection,
)
from torchcell.literature.zotero import ZoteroLibrary  # noqa: E402

log = logging.getLogger("lit_sync")

LOCK_PATH = Path("/tmp/torchcell-lit-sync.lock")


def _acquire_lock() -> "os.PathLike[str] | int":
    """Take an exclusive non-blocking flock; exit 0 if another run holds it."""
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("another lit-sync run holds the lock; exiting")
        sys.exit(0)
    return fd


def _write_report(report: SyncReport, dry_run: bool) -> None:
    """Log the per-collection outcome and persist a timestamped JSON report."""
    log.info(report.summary())
    for r in report.by_mode(SyncMode.CAPTURED):
        log.info("  captured: %s (%s)", r.citation_key, r.doi)
    for r in report.by_mode(SyncMode.FAILED):
        log.error("  FAILED: %s (%s) -- %s", r.citation_key, r.doi, r.error)
    for r in report.by_mode(SyncMode.UNSUPPORTED):
        log.info("  unsupported (needs hand-run): %s (doi=%s)", r.citation_key, r.doi)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = library_root(os.environ["DATA_ROOT"]) / "_sync_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = "dryrun_" if dry_run else ""
    report_path = reports_dir / f"sync_{prefix}{report.collection}_{stamp}.json"
    report_path.write_text(report.model_dump_json(indent=2))
    log.info("wrote report -> %s", report_path)


def main() -> None:
    """Parse args, sync each collection, write per-collection reports, exit non-zero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help=(
            "Zotero collection to sync; repeatable. "
            f"Default: {' '.join(DEFAULT_COLLECTIONS)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify the collection and report the gap; capture nothing.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Download PDFs but skip MinerU OCR (fast; markdown produced later).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap papers captured per collection this pass (bounds nightly GPU time).",
    )
    args = parser.parse_args()
    collections = args.collection or list(DEFAULT_COLLECTIONS)

    _acquire_lock()  # held until process exit

    lib = ZoteroLibrary.from_env()
    any_failed = False
    for collection in collections:
        report = sync_collection(
            lib,
            collection,
            do_ocr=not args.no_ocr,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        _write_report(report, args.dry_run)
        any_failed = any_failed or bool(report.by_mode(SyncMode.FAILED))

    # Non-zero exit if any capture failed, so cron mail / logs surface it.
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
