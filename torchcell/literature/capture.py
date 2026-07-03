# torchcell/literature/capture.py
# [[torchcell.literature.capture]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/capture.py
# Test file: tests/torchcell/literature/test_capture.py

"""End-to-end paper capture: Zotero fetch, OCR, SI data, then manifest."""

import logging
from pathlib import Path
from typing import Any

from torchcell.literature.manifest import build_manifest, write_manifest
from torchcell.literature.ocr import ocr_artifact
from torchcell.literature.si_data import fetch_si_data
from torchcell.literature.zotero import ZoteroLibrary, _resolve_citation_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _collection_names(lib: ZoteroLibrary, keys: list[str]) -> list[str]:
    """Map collection keys to names; fall back to the key if unresolved."""
    if not keys:
        return []
    by_key = {c["key"]: c["data"]["name"] for c in lib.list_collections()}
    return [by_key.get(k, k) for k in keys]


def _pdf_sources_and_md5(
    lib: ZoteroLibrary, item: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Build relative-path -> source and -> zotero md5 maps for the PDFs.

    Mirrors :meth:`ZoteroLibrary.download_artifact`'s naming: main article ->
    ``paper.pdf``, SI PDFs -> ``si/si1.pdf``, ``si/si2.pdf``, ...
    """
    sources: dict[str, str] = {}
    md5: dict[str, str] = {}
    attachments = lib.pdf_attachments(item["key"])
    si_index = 0
    for i, att in enumerate(attachments):
        rel = "paper.pdf" if i == 0 else f"si/si{(si_index := si_index + 1)}.pdf"
        sources[rel] = f"zotero:attachment:{att['key']}"
        if att["data"].get("md5"):
            md5[rel] = att["data"]["md5"]
    return sources, md5


def capture_by_doi(
    lib: ZoteroLibrary,
    doi: str,
    *,
    dryad_doi: str | None = None,
    extra_si_urls: list[str] | None = None,
    si_expected: list[str] | None = None,
    do_ocr: bool = True,
    data_root: str | Path | None = None,
) -> Path:
    """Capture a paper's full artifact from Zotero + external SI data.

    Pipeline: resolve the Zotero item by DOI -> download main + SI PDFs ->
    fetch SI data files (DRYAD / extra URLs) -> OCR every PDF to markdown ->
    write ``manifest.json`` recording provenance + per-file sha256. The manifest
    is written last so it captures every produced file.

    Args:
        lib: Connected Zotero library.
        doi: Paper DOI (the join key); must exist in the library.
        dryad_doi: Optional DRYAD dataset DOI for the SI data files.
        extra_si_urls: Additional direct URLs to pull into ``si/si_data/``.
        si_expected: SI items the paper says should exist (completeness check).
        do_ocr: Run MinerU OCR (slow on first run -- downloads models).
        data_root: Mirror root; defaults to ``$DATA_ROOT``.

    Returns:
        The artifact directory path.
    """
    item = lib.find_item_by_doi(doi)
    if item is None:
        raise ValueError(
            f"No Zotero item with DOI {doi} in library {lib.config.library_id}"
        )
    data = item["data"]
    citation_key = _resolve_citation_key(item)
    log.info("Capture: %s (%s)", citation_key, doi)

    # 1. PDFs (main + SI) from Zotero.
    artifact_dir = lib.download_artifact(item, data_root=data_root)

    # 2. SI data files from external repositories.
    si_results: list[tuple[Path, str]] = []
    if dryad_doi or extra_si_urls:
        si_results = fetch_si_data(
            artifact_dir, dryad_doi=dryad_doi, extra_urls=extra_si_urls
        )

    # 3. OCR every PDF.
    if do_ocr:
        ocr_artifact(artifact_dir)

    # 4. Manifest (last, so it sees every file).
    sources, zotero_md5 = _pdf_sources_and_md5(lib, item)
    for path, url in si_results:
        sources[str(path.relative_to(artifact_dir))] = url
    manifest = build_manifest(
        artifact_dir,
        citation_key=citation_key,
        doi=data.get("DOI"),
        title=data.get("title"),
        library_id=lib.config.library_id,
        zotero_item_key=item["key"],
        collections=_collection_names(lib, data.get("collections", [])),
        sources=sources,
        zotero_md5=zotero_md5,
        si_data_sources=[url for _, url in si_results],
        si_expected=si_expected,
    )
    write_manifest(artifact_dir, manifest)
    log.info("Capture: done -> %s", artifact_dir)
    return artifact_dir
