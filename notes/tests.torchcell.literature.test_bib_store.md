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
