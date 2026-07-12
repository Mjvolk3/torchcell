---
id: 0ez2qv7hqiwmpm6qcl5p590
title: Si_data
desc: ''
updated: 1783563770213
created: 1783563770213
---

## 2026.07.08 - Mirror the actual SI data files so reproduction never needs the publisher

This module exists to capture the supplementary DATA files themselves -- not just the SI PDF that lists what should exist -- from the durable external repositories that actually host them (DRYAD, GEO, Zenodo, GitHub). The SI PDF is a checklist; the real data lives elsewhere, and mirroring it into `<artifact_dir>/si/si_data/` is what makes a dataset rebuildable without ever hitting the publisher's paywalled or JS-gated site.

- `dryad_files`: resolves a DRYAD dataset DOI to its concrete file download URLs via the v2 API (dataset -> latest version -> files), so a whole deposit is captured by DOI rather than by hand-copied links.
- `fetch_si_data`: combines a DRYAD dataset with explicit extra URLs, streams each into `si/si_data/`, and returns `(local_path, source_url)` pairs so [[torchcell.literature.capture]] can record each file's origin in the [[torchcell.literature.manifest]].
- Idempotent by default (keeps a non-empty existing file) so re-running capture does not re-download.
