# torchcell/literature/manifest.py
# [[torchcell.literature.manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/manifest.py
# Test file: tests/torchcell/literature/test_manifest.py

"""Provenance and integrity manifest for captured paper artifacts."""

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Roles a file in an artifact directory can play.
ROLE_PAPER_PDF = "paper_pdf"
ROLE_PAPER_OCR = "paper_ocr"
ROLE_SI_PDF = "si_pdf"
ROLE_SI_OCR = "si_ocr"
ROLE_SI_DATA = "si_data"
ROLE_OCR_IMAGE = "ocr_image"
ROLE_OCR_LAYOUT = "ocr_layout"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class FileRecord(BaseModel):
    """One captured file in an artifact directory."""

    path: str = Field(description="Path relative to the artifact directory.")
    role: str = Field(description="paper_pdf | paper_ocr | si_pdf | si_ocr | si_data")
    bytes: int
    sha256: str
    source: str | None = Field(
        default=None,
        description="Where the bytes came from (zotero attachment key, URL, OCR).",
    )
    zotero_md5: str | None = Field(
        default=None, description="MD5 reported by Zotero for the synced attachment."
    )


class Manifest(BaseModel):
    """Provenance + integrity record for one paper's captured artifact.

    Written as ``manifest.json`` at the root of
    ``<data_root>/torchcell-library/<citation_key>/``. The DOI is the join key
    back to the Zotero item and to the TorchCell dataset object; the per-file
    sha256 lets a later run verify the mirror is intact, and ``si_data_sources``
    records the external repositories (DRYAD/GEO/GitHub/Zenodo) the SI data was
    pulled from so reproduction does not depend on the publisher.
    """

    version: int = Field(default=MANIFEST_VERSION)
    citation_key: str
    doi: str | None = None
    title: str | None = None
    library_id: str | None = None
    zotero_item_key: str | None = None
    collections: list[str] = Field(default_factory=list)
    files: list[FileRecord] = Field(default_factory=list)
    si_data_sources: list[str] = Field(
        default_factory=list,
        description="External URLs the SI data files were fetched from.",
    )
    si_expected: list[str] = Field(
        default_factory=list,
        description="SI items the paper says should exist (e.g. 'Data S1'..'S7'), "
        "for a completeness check against captured si_data files.",
    )
    created_at: str | None = None


def _role_for(rel_path: str) -> str:
    """Infer a file's role from its relative path within the artifact dir."""
    if rel_path == "paper.pdf":
        return ROLE_PAPER_PDF
    if rel_path == "paper.md":
        return ROLE_PAPER_OCR
    if rel_path.startswith("si/si_data/"):
        return ROLE_SI_DATA
    if rel_path.startswith("si/") and rel_path.endswith(".pdf"):
        return ROLE_SI_PDF
    if rel_path.startswith("si/") and rel_path.endswith(".md"):
        return ROLE_SI_OCR
    # MinerU byproducts: extracted figures and layout JSON, for paper and SI.
    if "images/" in rel_path and rel_path.endswith((".jpg", ".jpeg", ".png")):
        return ROLE_OCR_IMAGE
    if rel_path.endswith(("_content_list.json", "_middle.json")):
        return ROLE_OCR_LAYOUT
    return "other"


def build_manifest(
    artifact_dir: Path,
    *,
    citation_key: str,
    doi: str | None = None,
    title: str | None = None,
    library_id: str | None = None,
    zotero_item_key: str | None = None,
    collections: list[str] | None = None,
    sources: dict[str, str] | None = None,
    zotero_md5: dict[str, str] | None = None,
    si_data_sources: list[str] | None = None,
    si_expected: list[str] | None = None,
    created_at: str | None = None,
) -> Manifest:
    """Scan an artifact directory and build its manifest.

    Walks every file except ``manifest.json`` itself, computes sha256 and size,
    and tags each with a role and (when known) a source URL / Zotero md5.

    Args:
        artifact_dir: The ``<...>/<citation_key>/`` directory.
        citation_key: Zotero/library identity for the paper.
        doi: DOI join key, when known.
        title: Paper title, when known.
        library_id: Zotero library id the item came from.
        zotero_item_key: Zotero item key for the paper.
        collections: Zotero collection names the item belongs to.
        sources: Optional map of relative-path -> source string.
        zotero_md5: Optional map of relative-path -> Zotero-reported md5.
        si_data_sources: External URLs the SI data came from.
        si_expected: SI items the paper says exist (completeness checklist).
        created_at: ISO timestamp; defaults to now (UTC).
    """
    sources = sources or {}
    zotero_md5 = zotero_md5 or {}
    files: list[FileRecord] = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        rel = str(path.relative_to(artifact_dir))
        role = _role_for(rel)
        # OCR markdown has no external source; tag it as produced by MinerU.
        default_source = "mineru-ocr" if role in (ROLE_PAPER_OCR, ROLE_SI_OCR) else None
        files.append(
            FileRecord(
                path=rel,
                role=role,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source=sources.get(rel, default_source),
                zotero_md5=zotero_md5.get(rel),
            )
        )
    return Manifest(
        citation_key=citation_key,
        doi=doi,
        title=title,
        library_id=library_id,
        zotero_item_key=zotero_item_key,
        collections=collections or [],
        files=files,
        si_data_sources=si_data_sources or [],
        si_expected=si_expected or [],
        created_at=created_at or datetime.now(UTC).isoformat(),
    )


def write_manifest(artifact_dir: Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` into the artifact directory; return its path."""
    dest = artifact_dir / MANIFEST_FILENAME
    dest.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest: wrote %s (%d files)", dest, len(manifest.files))
    return dest
