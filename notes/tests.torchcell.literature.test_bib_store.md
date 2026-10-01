---
id: 5gtc4xy320o1umzlob5d6t2
title: Test_bib_store
desc: ''
updated: 1790765161553
created: 1790765161553
---

## 2026.09.30 - Phase 14: dispatch, exact .bib bytes, manifest rows, refusals, Makefile parsing

Twelve to twenty-four tests, 84 to 100 percent. Dispatch by scope shape, the exact bytes of three `.bib` files and every manifest row, the refusals with exact messages, Makefile parsing.

Findings: any 8-character upper-case name is sent as a collection key (line 267); a failed export leaves earlier specs' `.part` files behind; a spec removed from the repo stops being listed and served (404) but its `.bib` file stays on disk; an inline `# comment` after a Makefile value silently drops the document (82-84).

## 2026.10.01 - Findings retired: keys, staging, retired files, Makefile comments (issue #529)

Retired the four Findings. Now asserted: scope collections are keys by declaration (`RNASEQ01` sent as a key; `microbe-perturb-seq`, `ABCDEFG`, `w46ats7b` refused with the exact pydantic message and field); the paired pull gets `as_keys=True`; a failed export leaves `_bib/` empty (and with a previous store, exactly the previous three files); a dropped spec's `.bib` and a stray `.bib.part` move with their bytes to `_bib/_retired/T1/` with one exact warning each; an inline `# comment` is cut off both Makefile variables; a two-word value is refused naming the Makefile. 24 to 25 tests.

## 2026.10.01 - Review of PR #589: subset carry-forward, stamp, Makefile path

Replaced `test_dropped_spec_is_unserved_and_its_file_is_moved_aside` (it modeled a fewer-spec run as removal, pinning a 404 after a subset) with `test_spec_removed_from_the_repo_is_unserved_and_moved_aside`. Added: a `--name` run over three served bibliographies keeps all three listed, the other two records byte-identical and their files and GET hashes unchanged; a subset run still retires a spec removed from the repo; a missing or edited carried file is refused by name (parametrized) with the store unchanged; an undeclared exported spec is refused; four bad `generated_at` stamps, including `../../escaped`, are refused with no file moved anywhere under the mirror; the exporter's default stamp passes. The two-word Makefile test is now parametrized with a collection name and asserts the message naming the Makefile. Stamps are real UTC ISO timestamps (`T0`, `T1` constants). 25 to 31 test functions (36 collected).

## 2026.10.01 - Delta review of PR #589

Added five tests: a full export over a truncated manifest succeeds and writes a valid one; a subset export over it is refused with the exact message naming the manifest; an `OSError` injected into the `manifest.json.tmp` write leaves the previous manifest byte-identical; `2026-13-99T99:99:99+00:00` is refused; two runs with one stamp retire `stray.bib` and `stray.bib.1` with both contents kept. 31 to 36 test functions (41 collected).
