---
id: lz7oqmyp8z406skvx03y7pf
title: Test_releases
desc: ''
updated: 1790549796071
created: 1790549796071
---

## 2026.09.27 - Releases on tmp_path trees, a scripted driver and a stubbed git

Twenty-six tests added to the nine existing: `release_id` and `next_version` for the full and incremental kinds, `content_sha256` on a hand-built id list against a digest computed independently with hashlib, `content_hashes_from_csv` on a two-dataset BioCypher output directory, `KgRelease` to and from node properties, manifest reading and stamping as exact JSON, `diff` (unchanged, changed, added, removed), `compatibility` against a worktree surface, `resolve_database`, `datasets`, `read_release` and `write_release` through an in-memory scripted driver installed at `neo4j.GraphDatabase.driver` (no connection is made), the status report with `git` stubbed through PATH by a script that records its argv (the pattern of [[tests.torchcell.scripts.test_ops_sh]]), and every error path with its exact message. Module coverage from this file 99%; the one missed line is an unreachable `raise AssertionError` (releases.py line 871). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.10.13 - The restore check

Seven tests for the restore check, all on the literal plain output of a two-dataset store (`RESTORE_ROWS`):

- The query returns the parser's columns.
- `parse_restore_check` reads a bare `NULL` as None and refuses empty output, a foreign header, no rows, two release nodes, a repeated dataset and a short row, each with its message.
- `expected_total_nodes` gives `n_nodes + 1` for `3.0`, `n_nodes` for `3.1`, and None when nothing was recorded.
- `restore_mismatches` is empty for the archived release and lists each drift as one exact line.
- `pairing_statement` returns None when paired, the snapshot's pair otherwise, and `null` for a missing tag.
- Through `main`: `restore-check-query`; `verify-restore` exiting 0 or 1 with exact output; `pairing-statement` printing, printing nothing, or refusing a snapshot of another release.
