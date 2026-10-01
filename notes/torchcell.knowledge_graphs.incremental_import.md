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

## 2026.09.30 - Crash-atomic part-file rewrite in filter_existing_edges (issue #570)

- Previous behavior: after reading an edge part file, `filter_existing_edges` rewrote it in place with `write_text`. A crash mid-write left a truncated part file, which a rerun would filter again silently, since every run filters the CURRENT part file.
- Fix: the kept rows are written to `unfiltered/<Type>-partNNN.csv.filtering` (`FILTERING_SUFFIX`) and renamed over the part file with `os.replace`. The staged file lives in the `unfiltered/` subdirectory, so it never matches the `<directory>/<Type>-part.*` pattern the neo4j-admin call script passes. A crash before the rename leaves the part file whole; the next run overwrites the stale staged file and completes the rename.
- Evidence: `tests/torchcell/knowledge_graphs/test_incremental_import.py::test_filter_existing_edges_crash_before_rename_leaves_the_part_intact` (`os.replace` raises once; the part file keeps both original rows, the staged file holds the one kept row, `discover_csv_groups` still sees only the part file, and the rerun consumes the staged file).
- Ownership and durability: the rewritten part file is a new inode owned by the job user with a umask-derived mode, where the previous in-place write kept the uid 7474 owner and the `a+rw` mode set by the slurm script. Under the shell's umask 022, uid 7474 can still read it, which is all the import needs. "Crash-atomic" covers a process crash, not power loss: there is no `fsync` before the rename.
- The staged file never matches the call script's pattern: the test asserts that `re.fullmatch("ExperimentMemberOf-part.*", name)` over the entries of the output directory matches only `ExperimentMemberOf-part000.csv`. The slurm comment in `database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm` now describes the staged-then-replaced rewrite.
