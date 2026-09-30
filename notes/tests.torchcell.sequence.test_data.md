---
id: drwyri5andn2zt4503bmi6b
title: Test_data
desc: ''
updated: 1790773271173
created: 1790773271173
---

## 2026.09.30 - Phase 16: codon frequencies, window arithmetic, validator messages

Twenty-four to forty-seven tests, 85 to 97 percent. Exact codon frequencies and the repr's tie order; the CDS refusal message; the strand-dependent window arithmetic; exact validator messages; the abstract-method refusals of `Gene` and `Genome`; the lazy, cached gene set.

Findings: `GeneSet` repr of size exactly 3 returns None so `repr` raises `TypeError` (lines 130-133); `get_chr_from_description` returns None for an unmatched description against its `-> int` (321); an unknown strand in `calculate_window_undersized` raises `UnboundLocalError` (414-420) while `calculate_window_bounds` silently returns a 29 bp window for a 30 bp request (493-504); an empty CDS passes validation then divides by zero (694); a payload with no `start` raises a bare `TypeError` (58).
