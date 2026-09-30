torchcell.literature
====================

.. module:: torchcell.literature

.. currentmodule:: torchcell.literature

The literature capture subsystem. It reads papers and supplementary files from Zotero (``zotero``), OCRs or extracts them into markdown (``ocr``, ``extract``, ``scanned``), fetches supplementary data (``si_data``), and records a sha256 provenance manifest for every stored artifact (``manifest``, ``provenance``) in the on-disk mirror. ``server`` is the read-only HTTP endpoint (``tc-lit-server``) that serves the mirrored artifacts, and ``sync`` reconciles the mirror against a Zotero collection.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   BackfillReport
   Manifest
   LiteratureServerConfig
   LiteratureKeys
   ZoteroConfig
   ZoteroLibrary

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   backfill_key
   backfill_mirror
   capture_by_doi
   generate_citation_key
   build_manifest
   write_manifest
   ocr_artifact
   ocr_pdf
   create_app
   create_app_from_env
   dryad_files
   fetch_si_data
   make_zotero_client
   with_zotero_retry
