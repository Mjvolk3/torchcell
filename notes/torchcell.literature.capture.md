---
id: fuw1w2r8kcvbddy83gfx5jb
title: Capture
desc: ''
updated: 1783563706706
created: 1783563706706
---

## 2026.07.08 - One DOI in, one fully-provenanced artifact directory out

This module exists as the top-level orchestration that ties the whole subsystem into a single call: give it a paper's DOI and it produces the complete, self-documenting artifact directory the provenance principle demands. It sequences the pieces so nothing is lost -- fetch the main + SI PDFs from Zotero, pull the actual SI data files from external repositories, OCR every PDF to markdown, and write the manifest LAST so it inventories every produced file with per-file sha256.

- `capture_by_doi`: resolves the Zotero item by DOI (the join key; must exist in the library), then drives [[torchcell.literature.zotero]] -> [[torchcell.literature.si_data]] -> [[torchcell.literature.ocr]] -> [[torchcell.literature.manifest]] in order.
- Threads provenance through: each PDF is tagged with its Zotero attachment key + reported md5, and each SI data file with its source URL, so the manifest answers "where did this byte come from?" for everything in the directory.
- Fails loudly on a missing DOI rather than silently producing an empty capture.
