# torchcell/literature/backfill.py
# [[torchcell.literature.backfill]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/backfill.py
# Test file: tests/torchcell/literature/test_backfill.py

"""Backfill ``manifest.json`` for OCR'd artifact directories captured before the
provenance layer existed.

Every artifact file is sha256-pinned and role-tagged (offline, deterministic). When
a Zotero library is supplied, each directory is matched to its item by citation key
(the directory name) and enriched with doi/title/library_id/zotero_item_key/
collections plus per-attachment md5 + source pointers -- the same metadata a fresh
``capture_by_doi`` records. Directories with no Zotero match get an offline manifest
with ``provenance_complete=False``: the bytes are still hashed, only the upstream
retrieval chain is marked unknown (never fabricated).
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.literature.capture import _collection_names, _pdf_sources_and_md5
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    build_manifest,
    write_manifest,
)
from torchcell.literature.zotero import (
    ZoteroLibrary,
    _resolve_citation_key,
    with_zotero_retry,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LIBRARY_SUBDIR = "torchcell-library"

# Top-level Manifest metadata fields whose absence marks incomplete provenance.
_METADATA_FIELDS = ("doi", "title", "zotero_item_key")


class KeyBackfillResult(BaseModel):
    """Outcome of backfilling one citation-key directory."""

    citation_key: str
    mode: str = Field(description="enriched | offline | skipped")
    n_files: int = 0
    provenance_complete: bool = True
    null_metadata: list[str] = Field(
        default_factory=list,
        description="Top-level metadata fields left null (doi/title/zotero_item_key).",
    )
    doi: str | None = None


class BackfillReport(BaseModel):
    """Summary of a mirror-wide backfill run."""

    root: str
    used_zotero: bool
    results: list[KeyBackfillResult] = Field(default_factory=list)

    @property
    def enriched(self) -> int:
        """Count of directories enriched from Zotero."""
        return sum(1 for r in self.results if r.mode == "enriched")

    @property
    def offline(self) -> int:
        """Count of directories written offline (provenance incomplete)."""
        return sum(1 for r in self.results if r.mode == "offline")

    @property
    def skipped(self) -> int:
        """Count of directories skipped (manifest already present, no --force)."""
        return sum(1 for r in self.results if r.mode == "skipped")


def library_root(data_root: str | Path) -> Path:
    """Resolve the ``<data_root>/torchcell-library`` mirror directory."""
    return Path(data_root) / LIBRARY_SUBDIR


def build_citation_index(lib: ZoteroLibrary) -> dict[str, dict[str, Any]]:
    """Map every library item's citation key to the item (one paginated scan)."""
    items: list[dict[str, Any]] = with_zotero_retry(
        lambda: lib.zot.everything(lib.zot.items())
    )
    return {_resolve_citation_key(item): item for item in items}


def _enriched_manifest(
    artifact_dir: Path, citation_key: str, lib: ZoteroLibrary, item: dict[str, Any]
) -> Manifest:
    """Build a manifest enriched with the item's Zotero metadata."""
    data = item["data"]
    sources, zotero_md5 = _pdf_sources_and_md5(lib, item)
    return build_manifest(
        artifact_dir,
        citation_key=citation_key,
        doi=data.get("DOI") or None,
        title=data.get("title") or None,
        library_id=lib.config.library_id,
        zotero_item_key=item["key"],
        collections=_collection_names(lib, data.get("collections", [])),
        sources=sources,
        zotero_md5=zotero_md5,
        provenance_complete=True,
    )


def backfill_key(
    artifact_dir: Path,
    *,
    citation_index: dict[str, dict[str, Any]] | None = None,
    lib: ZoteroLibrary | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> KeyBackfillResult:
    """Regularize one citation-key directory to a ``manifest.json``.

    Args:
        artifact_dir: The ``<...>/torchcell-library/<citation_key>/`` directory.
        citation_index: Optional citation_key -> Zotero item map for enrichment.
        lib: Connected Zotero library (required to enrich when a match is found).
        force: Rewrite an existing manifest instead of skipping it.
        dry_run: Build the manifest but do not write it.

    Returns:
        The per-key outcome (mode, file count, provenance completeness).
    """
    citation_key = artifact_dir.name
    if (artifact_dir / MANIFEST_FILENAME).exists() and not force:
        return KeyBackfillResult(citation_key=citation_key, mode="skipped")

    item = (citation_index or {}).get(citation_key)
    if item is not None and lib is not None:
        manifest = _enriched_manifest(artifact_dir, citation_key, lib, item)
        mode = "enriched"
    else:
        manifest = build_manifest(
            artifact_dir, citation_key=citation_key, provenance_complete=False
        )
        mode = "offline"

    if not dry_run:
        write_manifest(artifact_dir, manifest)

    null_metadata = [
        name for name in _METADATA_FIELDS if getattr(manifest, name) is None
    ]
    return KeyBackfillResult(
        citation_key=citation_key,
        mode=mode,
        n_files=len(manifest.files),
        provenance_complete=manifest.provenance_complete,
        null_metadata=null_metadata,
        doi=manifest.doi,
    )


def backfill_mirror(
    root: str | Path,
    *,
    use_zotero: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> BackfillReport:
    """Backfill manifests for every citation-key directory under ``root``.

    Dynamically scans for directories lacking (or, with ``force``, already having) a
    manifest -- the set of keys is never hardcoded, so the mirror can keep growing.

    Args:
        root: The ``torchcell-library`` directory (or a fixture mirror root).
        use_zotero: Enrich from the TorchCell Zotero group (needs ``ZOTERO_*`` env);
            when False, every directory is written offline.
        force: Rewrite manifests that already exist.
        dry_run: Build + report but write nothing.

    Returns:
        A :class:`BackfillReport` with one :class:`KeyBackfillResult` per directory.
    """
    root = Path(root)
    lib: ZoteroLibrary | None = None
    citation_index: dict[str, dict[str, Any]] | None = None
    if use_zotero:
        lib = ZoteroLibrary.from_env()
        log.info("Backfill: scanning Zotero library %s ...", lib.config.library_id)
        citation_index = build_citation_index(lib)
        log.info("Backfill: indexed %d Zotero items", len(citation_index))

    results: list[KeyBackfillResult] = []
    for artifact_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        result = backfill_key(
            artifact_dir,
            citation_index=citation_index,
            lib=lib,
            force=force,
            dry_run=dry_run,
        )
        results.append(result)
        log.info(
            "Backfill: %-48s %-9s files=%d complete=%s",
            result.citation_key,
            result.mode,
            result.n_files,
            result.provenance_complete,
        )
    return BackfillReport(root=str(root), used_zotero=use_zotero, results=results)


def main() -> None:
    """CLI entry point: backfill manifests over the literature mirror."""
    import os

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="torchcell-library dir (default: $DATA_ROOT/torchcell-library).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rewrite existing manifests."
    )
    parser.add_argument(
        "--no-zotero",
        action="store_true",
        help="Skip Zotero enrichment; write every manifest offline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build + report but write nothing."
    )
    args = parser.parse_args()

    root = args.root or library_root(os.environ["DATA_ROOT"])
    report = backfill_mirror(
        root, use_zotero=not args.no_zotero, force=args.force, dry_run=args.dry_run
    )
    total = len(report.results)
    with_manifest = (
        total - report.skipped if args.force else report.enriched + report.offline
    )
    log.info(
        "Backfill done: %d dirs (enriched=%d offline=%d skipped=%d); "
        "%d manifests written this run",
        total,
        report.enriched,
        report.offline,
        report.skipped,
        0 if args.dry_run else with_manifest,
    )


if __name__ == "__main__":
    main()
