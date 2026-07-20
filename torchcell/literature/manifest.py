# torchcell/literature/manifest.py
# [[torchcell.literature.manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/manifest.py
# Test file: tests/torchcell/literature/test_manifest.py

"""Provenance and integrity manifest for captured paper artifacts."""

import hashlib
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
ROLE_RAW_DATA = "raw_data"
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


class RetrievalMethod(StrEnum):
    """How an artifact is fetched. ``radiant_endpoint`` is a future slot (issue
    #20) for un-scriptable sources served from the Radiant VM; there is no
    ``manual_browser`` -- manual-once artifacts are deposited then served.
    """

    springer_esm = "springer_esm"
    zotero_attachment = "zotero_attachment"
    pmc_oa_api = "pmc_oa_api"
    direct_url = "direct_url"
    zenodo = "zenodo"
    radiant_endpoint = "radiant_endpoint"


class SourceCheck(BaseModel):
    """Result of re-running the retriever to test the source still yields our bytes."""

    model_config = ConfigDict(extra="forbid")

    checked_at: str = Field(description="ISO timestamp of the check (caller-supplied).")
    produced_sha256: str = Field(description="sha256 the retriever produced this run.")
    matches: bool = Field(description="Did produced_sha256 equal the record's sha256?")


class RetrievalRecord(BaseModel):
    """How an artifact was obtained, referencing a versioned retriever function."""

    model_config = ConfigDict(extra="forbid")

    method: RetrievalMethod
    source_url: str | None = Field(
        default=None, description="Canonical URL (historical retrieval metadata)."
    )
    retriever: str = Field(
        description="Registry key into torchcell.literature.retrieve (the source code)."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="kwargs passed to the retriever function."
    )
    sha256: str = Field(description="sha256 of the retrieved bytes (canonical anchor).")
    retrieved_at: str = Field(description="ISO date the artifact was retrieved.")
    last_check: SourceCheck | None = Field(
        default=None, description="Last time the retriever was re-run + verified."
    )


class ProcessingRecord(BaseModel):
    """How a derived artifact (OCR markdown, extracted data) was produced."""

    model_config = ConfigDict(extra="forbid")

    processor: str = Field(description="Dotted path to the processing function.")
    tool: str = Field(description="mineru | pdftotext | llm-extract | ...")
    version: str = Field(description="Tool/library version used.")
    params: dict[str, Any] = Field(default_factory=dict)
    input_sha256: list[str] = Field(
        default_factory=list, description="sha256 of the inputs this was derived from."
    )


class ArtifactRecord(FileRecord):
    """A ``FileRecord`` plus how it was retrieved and/or processed.

    This is the general per-file record: it serves papers, supplements, software,
    and dataset raw files. ``Manifest.files`` is typed as ``list[ArtifactRecord]``
    so ``retrieval``/``processing`` survive serialize/reload (a bare ``FileRecord``
    field would silently drop them by declared-type serialization).
    """

    retrieval: RetrievalRecord | None = None
    processing: ProcessingRecord | None = None


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
    files: list[ArtifactRecord] = Field(default_factory=list)
    si_data_sources: list[str] = Field(
        default_factory=list,
        description="External URLs the SI data files were fetched from.",
    )
    si_expected: list[str] = Field(
        default_factory=list,
        description="SI items the paper says should exist (e.g. 'Data S1'..'S7'), "
        "for a completeness check against captured si_data files.",
    )
    provenance_complete: bool = Field(
        default=True,
        description="False when the manifest was backfilled without being able to "
        "source retrieval/processing provenance (e.g. the paper was OCR'd before "
        "provenance was formalized and is not resolvable in Zotero). Files are still "
        "sha256-pinned; only the upstream retrieval chain is unknown, not fabricated.",
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
    # Loose SI files (not a PDF/markdown and not already under si/si_data/) are
    # released supplementary data tables, e.g. hoepfner's ``si/Table_S5.xls``.
    if rel_path.startswith("si/"):
        return ROLE_SI_DATA
    # MinerU byproducts: extracted figures and layout JSON, for paper and SI.
    if "images/" in rel_path and rel_path.endswith((".jpg", ".jpeg", ".png")):
        return ROLE_OCR_IMAGE
    if rel_path.endswith(("_content_list.json", "_middle.json")):
        return ROLE_OCR_LAYOUT
    # Large released quantitative tables kept beside the paper (data-only keys
    # such as xue2025/lopez, and the ``data/`` companion of captured papers).
    if rel_path.startswith("data/"):
        return ROLE_RAW_DATA
    # Data-only sources whose "paper" is a born-digital document (e.g. lopez's
    # ``thesis.pdf`` + ``thesis.txt``): a top-level PDF is the source document;
    # a top-level ``.txt``/``.md`` is its born-digital text extraction.
    if "/" not in rel_path:
        if rel_path.endswith(".pdf"):
            return ROLE_PAPER_PDF
        if rel_path.endswith((".txt", ".md")):
            return ROLE_PAPER_OCR
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
    provenance_complete: bool = True,
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
        provenance_complete: False when retrieval/processing provenance could not
            be sourced (backfill of a pre-formalization key); files stay hashed.
        created_at: ISO timestamp; defaults to now (UTC).
    """
    sources = sources or {}
    zotero_md5 = zotero_md5 or {}
    files: list[ArtifactRecord] = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        rel = str(path.relative_to(artifact_dir))
        role = _role_for(rel)
        # OCR markdown has no external source; tag it as produced by MinerU.
        default_source = "mineru-ocr" if role in (ROLE_PAPER_OCR, ROLE_SI_OCR) else None
        files.append(
            ArtifactRecord(
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
        provenance_complete=provenance_complete,
        created_at=created_at or datetime.now(UTC).isoformat(),
    )


def write_manifest(artifact_dir: Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` into the artifact directory; return its path."""
    dest = artifact_dir / MANIFEST_FILENAME
    dest.write_text(manifest.model_dump_json(indent=2))
    log.info("Manifest: wrote %s (%d files)", dest, len(manifest.files))
    return dest
