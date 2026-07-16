# torchcell/literature/sync.py
# [[torchcell.literature.sync]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/sync.py
# Test file: tests/torchcell/literature/test_sync.py

"""Sync the on-disk library mirror against the Zotero ``database`` collection.

The Zotero ``database`` collection is the authoritative set of papers backing
the knowledge graph. This module diffs that collection against the citation-key
directories already present under ``<DATA_ROOT>/torchcell-library/`` and captures
any paper that is present in Zotero but missing from the mirror -- download the
PDF(s), OCR to markdown, write the provenance manifest.

It is the engine behind the nightly ``scripts/lit_sync_database.py`` job: new
papers dropped into the Zotero ``database`` collection are mirrored + OCR'd
automatically, without hand-running :func:`capture_by_doi` per paper.

A paper is captured only when it carries a DOI (the join key
:func:`capture_by_doi` relies on) and a PDF attachment. Papers that need special
retrieval (e.g. Costanzo's SOM.zip from the Boone-lab mirror) or that lack a DOI
are reported as ``unsupported`` and left for a hand-run -- never silently faked.
"""

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from torchcell.literature.backfill import library_root
from torchcell.literature.capture import capture_by_doi
from torchcell.literature.zotero import ZoteroLibrary, _resolve_citation_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATABASE_COLLECTION = "database"
MANIFEST_FILENAME = "manifest.json"


class SyncMode(StrEnum):
    """Outcome for one database-collection paper on a sync pass."""

    PRESENT = "present"  # already mirrored (dir + manifest) -- nothing to do
    CAPTURED = "captured"  # newly downloaded + OCR'd this run
    WOULD_CAPTURE = "would_capture"  # dry-run: eligible, not executed
    UNSUPPORTED = "unsupported"  # no DOI or no PDF attachment -- needs a hand-run
    FAILED = "failed"  # capture raised -- see ``error``


class KeySyncResult(BaseModel):
    """Per-paper outcome of a sync pass."""

    citation_key: str
    mode: SyncMode
    doi: str | None = None
    error: str | None = None


class SyncReport(BaseModel):
    """Aggregate outcome of one :func:`sync_database` pass."""

    collection: str
    n_collection_items: int
    results: list[KeySyncResult]

    def by_mode(self, mode: SyncMode) -> list[KeySyncResult]:
        """Results with a given mode."""
        return [r for r in self.results if r.mode == mode]

    def summary(self) -> str:
        """One-line ``mode=count`` tally for logs."""
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.mode] = counts.get(r.mode, 0) + 1
        tally = " ".join(f"{m}={counts[m]}" for m in sorted(counts))
        return f"{self.collection}: {self.n_collection_items} items | {tally}"


def _database_items(lib: ZoteroLibrary) -> list[dict[str, Any]]:
    """Top-level (non-attachment/note) items of the Zotero ``database`` collection."""
    coll_key = lib.collection_key(DATABASE_COLLECTION)
    items: list[dict[str, Any]] = lib.zot.everything(lib.zot.collection_items(coll_key))
    return [
        it for it in items if it["data"].get("itemType") not in ("attachment", "note")
    ]


def _is_mirrored(root: Path, citation_key: str) -> bool:
    """True when the mirror already holds a captured directory for this key.

    Captured directories always carry a ``manifest.json`` (written last by the
    capture pipeline); its absence means an incomplete/aborted capture that the
    sync should redo.
    """
    return (root / citation_key / MANIFEST_FILENAME).exists()


def plan_database_sync(
    lib: ZoteroLibrary, data_root: str | Path | None = None
) -> SyncReport:
    """Diff the Zotero ``database`` collection against the mirror, capturing nothing.

    Every collection paper is classified: ``present`` (already mirrored),
    ``unsupported`` (no DOI or no PDF attachment), or ``would_capture`` (eligible
    and missing). This is the read-only view the ``--dry-run`` job prints.
    """
    root = library_root(data_root or os.environ["DATA_ROOT"])
    items = _database_items(lib)
    results: list[KeySyncResult] = []
    for item in items:
        key = _resolve_citation_key(item)
        doi = (item["data"].get("DOI") or "").strip() or None
        if _is_mirrored(root, key):
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.PRESENT, doi=doi)
            )
        elif doi is None or not lib.pdf_attachments(item["key"]):
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.UNSUPPORTED, doi=doi)
            )
        else:
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.WOULD_CAPTURE, doi=doi)
            )
    return SyncReport(
        collection=DATABASE_COLLECTION, n_collection_items=len(items), results=results
    )


def sync_database(
    lib: ZoteroLibrary,
    *,
    data_root: str | Path | None = None,
    do_ocr: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
) -> SyncReport:
    """Mirror + OCR every ``database``-collection paper missing from the mirror.

    Args:
        lib: Connected Zotero library.
        data_root: Mirror root; defaults to ``$DATA_ROOT``.
        do_ocr: Run MinerU OCR on captured PDFs (the "minor OCR processing").
        dry_run: Classify only; capture nothing (equivalent to
            :func:`plan_database_sync`).
        limit: Cap the number of papers captured this pass (``None`` = no cap).
            Bounds a nightly run's GPU time when a large backlog appears at once.

    Returns:
        A :class:`SyncReport` with a per-paper outcome for the whole collection.
    """
    if dry_run:
        return plan_database_sync(lib, data_root=data_root)

    root = library_root(data_root or os.environ["DATA_ROOT"])
    items = _database_items(lib)
    results: list[KeySyncResult] = []
    captured = 0
    for item in items:
        key = _resolve_citation_key(item)
        doi = (item["data"].get("DOI") or "").strip() or None
        if _is_mirrored(root, key):
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.PRESENT, doi=doi)
            )
            continue
        if doi is None or not lib.pdf_attachments(item["key"]):
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.UNSUPPORTED, doi=doi)
            )
            continue
        if limit is not None and captured >= limit:
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.WOULD_CAPTURE, doi=doi)
            )
            continue
        try:
            capture_by_doi(lib, doi, do_ocr=do_ocr, data_root=data_root)
            captured += 1
            results.append(
                KeySyncResult(citation_key=key, mode=SyncMode.CAPTURED, doi=doi)
            )
        except Exception as exc:  # noqa: BLE001 -- report, do not abort the batch
            log.exception("sync: capture failed for %s (%s)", key, doi)
            results.append(
                KeySyncResult(
                    citation_key=key, mode=SyncMode.FAILED, doi=doi, error=str(exc)
                )
            )
    return SyncReport(
        collection=DATABASE_COLLECTION, n_collection_items=len(items), results=results
    )
