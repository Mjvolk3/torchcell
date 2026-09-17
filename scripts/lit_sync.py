#!/usr/bin/env python
# scripts/lit_sync.py
# [[scripts.lit_sync]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_sync.py

"""Nightly sync: mirror + OCR new papers from both Zotero libraries we track.

Covers two sources, matching the union the bibliography is built from
(``scripts/lit_bib.py``) so a citable ``@key`` and its OCR markdown stay in step:

1. named **group** collections (default ``database`` + ``paper`` +
   ``microbe-perturb-seq``), and
2. the **personal** collection trees, each walked recursively: ``torchcell``
   (where new reading is filed first) and ``thesis`` (the dissertation's
   primary-sources, biofoundry-ai and publications collections). The list is
   ``--personal-root`` (repeatable), else the comma-separated
   ``ZOTERO_USER_ROOT_COLLECTION``, else ``DEFAULT_PERSONAL_ROOTS``.

Each is diffed against ``<DATA_ROOT>/torchcell-library/`` and any paper present in
Zotero but missing from the mirror is captured (download PDF -> MinerU OCR ->
manifest). Idempotent: already-mirrored papers are skipped, so re-running only picks
up what is new. A key that appears in several collections, or in both libraries, is
captured once and reads ``present`` afterward, so ordering never double-downloads.

Capture needs a DOI **and** a PDF attachment; anything else is reported
``unsupported`` and needs a fix in Zotero rather than another run.

Supersedes ``scripts/lit_sync_database.py`` (database-only). Designed to run from
cron on GilaHyper (GPU host for MinerU). Self-flocks so a nightly run never
overlaps an on-demand run. Writes one timestamped JSON report per collection under
``<DATA_ROOT>/torchcell-library/_sync_reports/`` and appends a one-line summary to
the log.

Usage::

    python scripts/lit_sync.py                        # group collections + personal tree
    python scripts/lit_sync.py --collection paper     # just the paper collection (+ tree)
    python scripts/lit_sync.py --no-personal          # group only, the pre-2026.08 behavior
    python scripts/lit_sync.py --no-group --personal-root thesis   # one personal tree
    python scripts/lit_sync.py --dry-run              # report the gap, capture nothing
    python scripts/lit_sync.py --limit 5              # cap captures (see below)
    python scripts/lit_sync.py --no-ocr               # download only, skip OCR

``--limit`` is a per-collection cap for the group pass, but a single budget shared
across each personal tree: a 25-collection tree under a per-collection cap
would authorize far more MinerU time than a nightly run should take.

A personal root that does not exist in the library is logged as an error for that
root and the run exits non-zero, after the other roots have been synced: one
misnamed (or not yet created) tree must not hide the night's captures elsewhere.
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

from pydantic import SecretStr  # noqa: E402

from torchcell.literature.backfill import library_root  # noqa: E402
from torchcell.literature.sync import (  # noqa: E402
    DEFAULT_COLLECTIONS,
    DEFAULT_PERSONAL_ROOTS,
    SyncMode,
    SyncReport,
    parse_root_list,
    sync_collection,
    sync_collection_tree,
)
from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary  # noqa: E402

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
    # A personal-tree label is a slash-joined collection path, which would otherwise
    # be read as directories; ':' separates the library prefix and is dropped too.
    slug = report.collection.replace("/", "__").replace(":", "_")
    report_path = reports_dir / f"sync_{prefix}{slug}_{stamp}.json"
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
    parser.add_argument(
        "--no-personal",
        action="store_true",
        help="Skip the personal collection trees; sync the group collections only.",
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Skip the group collections; sync the personal collection trees only.",
    )
    parser.add_argument(
        "--personal-root",
        action="append",
        metavar="NAME",
        help=(
            "Personal collection tree to sync recursively; repeatable. Default: "
            "$ZOTERO_USER_ROOT_COLLECTION (comma-separated), else "
            f"{','.join(DEFAULT_PERSONAL_ROOTS)}."
        ),
    )
    args = parser.parse_args()
    collections = args.collection or list(DEFAULT_COLLECTIONS)
    personal_roots = args.personal_root or parse_root_list(
        os.environ.get("ZOTERO_USER_ROOT_COLLECTION", ",".join(DEFAULT_PERSONAL_ROOTS))
    )

    _acquire_lock()  # held until process exit

    lib = ZoteroLibrary.from_env()
    any_failed = False
    for collection in [] if args.no_group else collections:
        report = sync_collection(
            lib,
            f"group:{collection}",
            do_ocr=not args.no_ocr,
            dry_run=args.dry_run,
            limit=args.limit,
            collection_key=lib.collection_key(collection),
        )
        _write_report(report, args.dry_run)
        any_failed = any_failed or bool(report.by_mode(SyncMode.FAILED))

    # The personal trees are where new reading lands first, so a group-only sync
    # mirrors them late or never. Same mirror, same keys: a paper filed in several
    # trees or in both libraries is captured once and reads `present` after that.
    if not args.no_personal:
        user = ZoteroLibrary(
            ZoteroConfig(
                library_id=os.environ["ZOTERO_USER_ID"],
                library_type="user",
                api_key=SecretStr(os.environ["ZOTERO_API_KEY"]),
            )
        )
        for root in personal_roots:
            try:
                reports = sync_collection_tree(
                    user,
                    root,
                    do_ocr=not args.no_ocr,
                    dry_run=args.dry_run,
                    limit=args.limit,
                    label_prefix="personal:",
                )
            except ValueError as exc:
                # collection_key() raises when the root is not in the library,
                # naming what IS there. Report it and keep going: a tree not yet
                # created in Zotero must not hide the other trees' captures.
                log.error("personal root %r not synced: %s", root, exc)
                any_failed = True
                continue
            for report in reports:
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
