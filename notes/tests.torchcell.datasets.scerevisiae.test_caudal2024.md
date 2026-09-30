---
id: 89lo1xt9so6tyi2w34q0sem
title: Test_caudal2024
desc: ''
updated: 1790777401208
created: 1790777401208
---

## 2026.09.30 - Phase 17: the four consumed columns, rounding, the reference

New paired file, eight tests, 94.6 to 100 percent with the two existing Caudal files. The loader reads only `Strain, systematic_name, count, tpm`, so the released header's extra columns change nothing and no replicate column is consumed; two allele rows summed; a count of 12.5 rounding half-to-even to 12; the reference as the mean over isolates carrying the gene (round(16.25) = 16); a table missing `tpm` refused with pandas' exact message; the SGD FASTA's last tagged record flushed; the stale raw matrix used; `main`'s exact stdout.

Findings: an existing raw file is never re-hashed (line 324); a blank `systematic_name` is dropped silently by groupby's `dropna` (463); a zip with no `.tab` member raises a bare `StopIteration` (448); the unextractable-member guard can never fire on a real archive and, forced, drops that gene's variants without a log (522-523).
