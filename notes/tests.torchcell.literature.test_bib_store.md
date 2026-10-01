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
