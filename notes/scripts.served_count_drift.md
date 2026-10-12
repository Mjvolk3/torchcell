---
id: v8bv1pj93z31yhaj5ke8f31
title: Served_count_drift
desc: ''
updated: 1791814464907
created: 1791814464907
---

## 2026.10.12 - Why this script exists

On 2026.10.11 a loader change (PR #983) refused 317,234 Costanzo 2016 and Kuzmin 2020 double-mutant records because the R64-4-1 annotation no longer carries 35 of the released ORF names. CI was green and the adapter pin counts were right, so nothing caught it until a removals audit the next morning. The owner's rule: the data is the data; a released record is dropped only when its authors withdrew it; the genome is a separate layer that never gates a measurement.

This script is the mechanical half of that rule. It compares the newest committed served-counts table against the dev stores and exits 1 on any decrease, so a revised loader of a served dataset cannot land, and a KG build cannot be cut, with a silent loss. Run from the repo root:

```bash
~/miniconda3/envs/torchcell/bin/python scripts/served_count_drift.py
```

First run, 2026.10.12 09:30 CDT, against KG 3.0's 51 datasets: 48 unchanged, SGD essentiality +147 (the #985 typed media gap brought alleles back), the two Costanzo double-mutant stores decreased only because they were mid-rebuild under the reverted loader. The `add-dataset` skill (`.claude/skills/add-dataset/SKILL.md`) makes the run part of every dataset PR.
