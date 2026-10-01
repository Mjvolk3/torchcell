---
id: 3o85bzl0bys0hzgy6kyvof4
title: Incremental_import
desc: ''
updated: 1790816086704
created: 1790816086704
---

## 2026.09.30 - filter_existing_edges is idempotent (issue #538)

- A rerun re-read the `unfiltered/` backup but looked up only the pairs in the current (already filtered) part file, so a row the first run found in the store was written back into the increment.
- Every run now filters the CURRENT part file. The `unfiltered/` copy is written once (a byte copy, on the first run that drops a row of that file) and never overwritten, and the summary counts this run's drops only. Reading the current output was chosen over refusing a rerun so that a retried increment job still works; it is safe because an incremental import never deletes a relationship, so a row once found in the store stays there.
- Evidence: `tests/torchcell/knowledge_graphs/test_incremental_import.py::test_filter_existing_edges_rerun_is_idempotent`.
