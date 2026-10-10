---
id: j8zd4t6twlf8k45oawt8c0h
title: Ledger
desc: ''
updated: 1791617793581
created: 1791617793581
---

## 2026.10.10 - What this ledger is

This note is the generated ledger AND the note of `torchcell/candidates/ledger.py`. `python -m torchcell.candidates ledger` renders every verdict in `database/candidates/` as one Markdown table and appends it as a `## YYYY.MM.DD - Candidate gate ledger` section; a second section on the same day is refused, so earlier states are never overwritten. Columns: citation key, table, row, one glyph per gate (pass, fail, blocked, gap with its issue, unmeasured), the derived outcome, the issues, the decision date, and the 9-character commit the gates read. Never edit a generated section by hand; re-run the gate and append the next day's section.

No ledger section exists yet: the store is empty until the re-audit (piece 3 of [[plan.dataset-admission-pipeline.2026.10.10]]).
