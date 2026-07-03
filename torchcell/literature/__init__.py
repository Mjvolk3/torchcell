# torchcell/literature/__init__.py
# [[torchcell.literature.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/__init__.py

"""Literature capture and provenance subsystem (Zotero, OCR, manifest)."""

from torchcell.literature.capture import capture_by_doi
from torchcell.literature.citation_keys import generate_citation_key
from torchcell.literature.manifest import Manifest, build_manifest, write_manifest
from torchcell.literature.ocr import ocr_artifact, ocr_pdf
from torchcell.literature.si_data import dryad_files, fetch_si_data
from torchcell.literature.zotero import (
    ZoteroConfig,
    ZoteroLibrary,
    make_zotero_client,
    with_zotero_retry,
)

__all__ = [
    "capture_by_doi",
    "generate_citation_key",
    "Manifest",
    "build_manifest",
    "write_manifest",
    "ocr_artifact",
    "ocr_pdf",
    "dryad_files",
    "fetch_si_data",
    "ZoteroConfig",
    "ZoteroLibrary",
    "make_zotero_client",
    "with_zotero_retry",
]
