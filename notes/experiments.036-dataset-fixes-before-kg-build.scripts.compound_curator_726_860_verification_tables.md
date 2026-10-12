---
id: ue652t9kjo4xu4ufpw8l4zd
title: Compound_curator_726_860_verification_tables
desc: ''
updated: 1791698267432
created: 1791698267432
---

## 2026.10.11 - Rule tables for the stores the #726 / #860 pass rebuilt

Reads the results file `kg4_pre_build_verification_sweep.py --out` wrote for the rebuilt
stores and prints one markdown table per report (level, rule, pass or fail, the report's
message), which the dataset notes paste as their L0-L4 tables. Output:
`results/compound_curator_726_860_verification_tables.txt`.

## 2026.10.11 - What the dataset notes carry, and reports rewritten under them

The script reads each store's `preprocess/verification_report*.json` (not the sweep's
results file the section above names) and marks a report older than its store's build
manifest `STALE REPORT`. The combined output was not committed. Each dataset note carries
its store's tables as rendered before that note's commit (08:21 to 09:22 UTC on
2026-10-11), except Hoepfner 2014 and Borchert 2024, rendered after the rebase from the
reports this branch's slurm 3872 (task 1, 15:43 UTC) and 3836 (11:28 UTC) wrote.

A re-render of all 45 classes after the rebase found 12 reports rewritten
between 10:43 and 10:52 UTC on 2026-10-11, after those renders and by no slurm job of this
branch, with every store's build manifest unchanged: the three Yunus 2026 CRISPRi stores,
Niu 2019, Wang 2015, Goodall 2018, Foo 2014, De Siqueira 2025, both Thompson 2019 stores,
Schastnaya 2021 and Banerjee 2025. Seven still match their notes' tables rule for rule.
Goodall 2018's now FAILs `one_record_per_gene_and_condition` and
`calls_match_released_tables` on 4,098 records where the note's 08:04 UTC report passes
both. Of the four without a note table, Thompson 2019's lactam report now lists `inchikey`
as a deferred field and the Foo 2014, De Siqueira 2025 and Thompson 2019 valerolactam
reports have one rule fewer than at 09:22 UTC. Hypothesis
(untested): a verification run from a checkout without this branch's rows, against the dev
stores every worktree shares. The tables in the notes are the reports this branch's own
jobs wrote, and a re-render here does not reproduce them.
