# torchcell/literature/manifest.py
# [[torchcell.literature.manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/manifest.py
# Test file: tests/torchcell/literature/test_manifest.py

"""Provenance and integrity manifest for captured paper artifacts."""

import argparse
import hashlib
import logging
import re
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

# Roles a file in an artifact directory can play.
ROLE_PAPER_PDF = "paper_pdf"
ROLE_PAPER_OCR = "paper_ocr"
# The publisher's own plain text of an article (PMC's .txt rendering), which is NOT
# OCR: it is the source of record for a verbatim quote when a paper has no Zotero
# item, and therefore no OCR'd literature mirror, to quote from.
ROLE_PAPER_TEXT = "paper_text"
# A deterministic TEXT rendering of a supplementary file that is not itself text (a
# workbook's legend rows, a table dumped to TSV). Its ProcessingRecord names the
# reader and pins the input's sha256, so a quote in it is auditable although the
# artifact it came from is binary.
ROLE_SI_TEXT = "si_text"
ROLE_SI_PDF = "si_pdf"
ROLE_SI_OCR = "si_ocr"
ROLE_SI_DATA = "si_data"
ROLE_RAW_DATA = "raw_data"
ROLE_OCR_IMAGE = "ocr_image"
ROLE_OCR_LAYOUT = "ocr_layout"
# ``ocr.ocr_pdf``'s ProcessingRecord for one OCR'd PDF, written beside its markdown
# as ``<stem>_ocr_provenance.json`` and attached to that markdown's record.
ROLE_OCR_PROVENANCE = "ocr_provenance"
OCR_PROVENANCE_SUFFIX = "_ocr_provenance.json"
# Our own reading notes pulled from Zotero -- the only role not derived from the
# paper's own bytes, so it is sourced to Zotero rather than to a retrieval URL.
ROLE_ANNOTATIONS = "annotations"


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
    role: str = Field(
        description="paper_pdf | paper_ocr | paper_text | si_pdf | si_ocr | "
        "si_text | si_data | raw_data | ocr_image | ocr_layout | ocr_provenance | "
        "annotations"
    )
    bytes: int
    sha256: str
    source: str | None = Field(
        default=None,
        description="Where the bytes came from (zotero attachment key, URL, OCR).",
    )
    zotero_md5: str | None = Field(
        default=None, description="MD5 reported by Zotero for the synced attachment."
    )
    original_filename: str | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="The file's name at its source when it is stored under another "
        "name (``capture_si`` writes ``si/si<N>.<ext>`` and keeps the publisher's "
        "name here). Omitted from the JSON when unset, so manifests written before "
        "the field existed serialize byte-identically.",
    )


class RetrievalMethod(StrEnum):
    """How an artifact is fetched. ``radiant_endpoint`` is a future slot (issue
    #20) for un-scriptable sources served from the Radiant VM. ``manual_browser``
    is a manual-once retrieval (a publisher behind a browser challenge): the record
    carries the human recipe as its command and the deposited bytes are what every
    rebuild reads; re-checking the source is not scriptable for it. ``pmc_cloud`` is
    one object of the PMC Article Datasets bucket on AWS, which replaced the retired
    ``pmc_oa_api`` package service. ``local_archive`` is a sha256-verified copy of a
    file out of a local archive whose own manifest records where the file originally
    lived (the thesis archive's ``MANIFEST.tsv``); the record's ``source_url`` names
    that original location and the retriever reads the archive copy.
    ``git_repository`` is a file taken from a git repository at a pinned commit: the
    retriever clones the repository with ``gh repo clone <owner>/<repo> <dir> -- --depth
    1``, fetches and checks out the pinned commit, and reads the file at
    ``path_in_repo``. A private repository needs the owner's ``gh`` authentication, so
    for one the deposited bytes are the rebuild source and the clone re-checks them
    only where that authentication is available.
    """

    springer_esm = "springer_esm"
    zotero_attachment = "zotero_attachment"
    pmc_oa_api = "pmc_oa_api"
    pmc_cloud = "pmc_cloud"
    direct_url = "direct_url"
    zenodo = "zenodo"
    radiant_endpoint = "radiant_endpoint"
    pubchem_api = "pubchem_api"
    manual_browser = "manual_browser"
    local_archive = "local_archive"
    globus = "globus"
    git_repository = "git_repository"


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


# What ``_run_mineru.py`` writes beside each PDF ``ocr.ocr_artifact`` OCRs
# (``paper.pdf`` and ``si/si*.pdf``): its figures, ``<stem>_content_list.json`` /
# ``<stem>_middle.json`` with ``<stem>`` the PDF stem, and ``ocr.ocr_pdf`` adds
# ``<stem>_ocr_provenance.json``. Figures: ``images/<file>`` for the paper,
# ``images/<stem>/<file>`` for another PDF at the key root (``SOM.pdf``) and
# ``si/images/<si stem>/<file>`` per SI PDF (``ocr.images_dir_for``); the flat
# ``si/images/<file>`` is the layout written before issue #579 and is still an OCR
# figure, so keys with one SI PDF stay valid without a re-OCR.
_MINERU_IMAGE = re.compile(
    r"(images|images/[^/]+|si/images|si/images/si[^/]*)/[^/]+\.(jpg|jpeg|png)"
)
_MINERU_LAYOUT = re.compile(r"(paper|si/si[^/]*)_(content_list|middle)\.json")
_MINERU_PROVENANCE = re.compile(r"(paper|si/si[^/]*)_ocr_provenance\.json")


def _role_for(rel_path: str) -> str:
    """Infer a file's role from its relative path within the artifact dir."""
    if rel_path == "paper.pdf":
        return ROLE_PAPER_PDF
    if rel_path == "paper.md":
        return ROLE_PAPER_OCR
    if rel_path in ("annotations.json", "annotations.md"):
        return ROLE_ANNOTATIONS
    if rel_path.startswith("si/si_data/"):
        return ROLE_SI_DATA
    # MinerU byproducts, for paper and SI alike: ``_run_mineru.py`` copies
    # ``<stem>_content_list.json``, ``<stem>_middle.json`` and ``images/`` next
    # to every PDF it OCRs, so ``si/si1_middle.json`` and ``si/images/*.jpg`` are
    # OCR sidecars and must win over the loose-SI rule below.
    if _MINERU_IMAGE.fullmatch(rel_path):
        return ROLE_OCR_IMAGE
    if _MINERU_LAYOUT.fullmatch(rel_path):
        return ROLE_OCR_LAYOUT
    if _MINERU_PROVENANCE.fullmatch(rel_path):
        return ROLE_OCR_PROVENANCE
    if rel_path.startswith("si/") and rel_path.endswith(".pdf"):
        return ROLE_SI_PDF
    if rel_path.startswith("si/") and rel_path.endswith(".md"):
        return ROLE_SI_OCR
    # Loose SI files (not a PDF/markdown, not an OCR sidecar and not already
    # under si/si_data/) are released supplementary data tables, e.g.
    # hoepfner's ``si/Table_S5.xls``.
    if rel_path.startswith("si/"):
        return ROLE_SI_DATA
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


def _is_mineru_output(rel_path: str) -> bool:
    """Whether a file is markdown MinerU writes, by its location.

    ``ocr.ocr_artifact`` OCRs ``paper.pdf`` and each ``si/si*.pdf`` and MinerU writes
    the markdown beside its PDF, so its outputs are exactly ``paper.md`` and the
    ``.md`` files directly under ``si/``. Any other text file (a top-level born-digital
    ``thesis.txt``, say) carries the ``paper_ocr`` role but no OCR step ran on it.
    """
    if rel_path == "paper.md":
        return True
    parts = rel_path.split("/")
    return len(parts) == 2 and parts[0] == "si" and parts[1].endswith(".md")


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
    and tags each with a role and (when known) a source URL / Zotero md5. A MinerU
    markdown whose ``<stem>_ocr_provenance.json`` sits beside it (written by
    ``ocr.ocr_pdf``) carries that ProcessingRecord as ``processing``; markdown
    OCR'd before the record existed has none.

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
        mineru_output = _is_mineru_output(rel)
        default_source = "mineru-ocr" if mineru_output else None
        record_path = path.with_name(f"{path.stem}{OCR_PROVENANCE_SUFFIX}")
        processing = (
            ProcessingRecord.model_validate_json(record_path.read_text())
            if mineru_output and record_path.is_file()
            else None
        )
        files.append(
            ArtifactRecord(
                path=rel,
                role=role,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source=sources.get(rel, default_source),
                zotero_md5=zotero_md5.get(rel),
                processing=processing,
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


class ManifestFailure(BaseModel):
    """One ``<key>/manifest.json`` that does not validate as a :class:`Manifest`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_key: str = Field(description="The key directory the manifest sits in.")
    path: str = Field(description="Absolute path of the failing manifest.")
    n_errors: int = Field(description="Number of pydantic validation errors.")
    error: str = Field(description="The pydantic ValidationError, verbatim.")


class MirrorScan(BaseModel):
    """Every paper-key manifest under one mirror root, split into valid and failing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str
    manifests: dict[str, Manifest] = Field(
        description="Key directory name -> its validated manifest, in sorted order."
    )
    failures: tuple[ManifestFailure, ...]


def scan_mirror(root: Path) -> MirrorScan:
    """Validate every ``<root>/<key>/manifest.json``; report, never drop, the failures.

    ``root`` is a mirror tree (``$DATA_ROOT/torchcell-library`` or
    ``$DATA_ROOT/torchcell-raw``). Directories whose name starts with ``_`` (``_bib``,
    ``_sync_reports``) are service stores with their own manifest models, not paper
    keys, and are not read. A manifest that fails validation (bad JSON or the wrong
    shape) is returned as a :class:`ManifestFailure` carrying the verbatim error, so a
    caller that has to keep going over the other keys still names the bad one.
    """
    manifests: dict[str, Manifest] = {}
    failures: list[ManifestFailure] = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
        key = manifest_path.parent.name
        if key.startswith("_"):
            continue
        try:
            manifests[key] = Manifest.model_validate_json(manifest_path.read_text())
        except ValidationError as exc:
            failures.append(
                ManifestFailure(
                    citation_key=key,
                    path=str(manifest_path),
                    n_errors=exc.error_count(),
                    error=str(exc),
                )
            )
    return MirrorScan(root=str(root), manifests=manifests, failures=tuple(failures))


MIRROR_TREES: tuple[str, ...] = ("torchcell-library", "torchcell-raw")


def main(argv: list[str] | None = None) -> int:
    """``validate``: check every manifest in both mirrors; exit 1 naming each bad key.

    The command-line counterpart of ``tc-lit``'s HTTP 500 on a corrupt manifest: a
    failing manifest is reported with its validation error and fails the run, it is
    never skipped silently.
    """
    parser = argparse.ArgumentParser(prog="python -m torchcell.literature.manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate every mirror manifest")
    validate.add_argument("--data-root", required=True, type=Path)
    validate.add_argument("--tree", choices=MIRROR_TREES, action="append", default=None)
    args = parser.parse_args(argv)
    n_bad = 0
    for tree in args.tree or MIRROR_TREES:
        base = args.data_root / tree
        if not base.is_dir():
            print(f"{tree}: no such directory {base}", file=sys.stderr)
            return 2
        scan = scan_mirror(base)
        print(f"{tree}: {len(scan.manifests)} valid, {len(scan.failures)} failing")
        for failure in scan.failures:
            print(f"FAIL {tree}/{failure.citation_key}: {failure.error}")
        n_bad += len(scan.failures)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
