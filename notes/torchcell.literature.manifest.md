---
id: s0siejlx1mf64fc6astuypo
title: Manifest
desc: ''
updated: 1783563727901
created: 1783563727901
---

## 2026.07.08 - The pydantic ground truth: per-file sha256 + provenance for one paper's mirror

This module exists to give every captured paper a single, serializable record of what it contains and where each byte came from -- written as `manifest.json` at the root of its artifact directory. It is the bottom of the import graph on purpose: the record types (`RetrievalMethod`, `RetrievalRecord`, `ProcessingRecord`, `ArtifactRecord`, `Manifest`) are pure pydantic data with no I/O, so [[torchcell.literature.provenance]] can layer verify/re-run BEHAVIOR on top without a circular import. The per-file sha256 is what lets any later run prove the mirror is intact, and the DOI is the join key back to Zotero and to the TorchCell dataset object.

- `build_manifest` scans an artifact directory, hashes and sizes every file, and tags each with a role (paper PDF, SI PDF, OCR markdown, SI data, MinerU byproducts) so the manifest is a complete inventory, not a curated subset.
- `ArtifactRecord` carries optional retrieval + processing sub-records so provenance survives serialize/reload -- one general per-file record serving papers, supplements, and dataset raw files alike.
- `si_expected` vs captured `si_data` gives a completeness check (what the paper says should exist vs what we actually mirrored); `si_data_sources` records the external repos so reproduction never needs the publisher.
