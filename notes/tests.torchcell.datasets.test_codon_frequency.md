---
id: 3kphr43ynt071ejy0zuzhtf
title: Test_codon_frequency
desc: ''
updated: 1790818979238
created: 1790818979238
---

## 2026.09.30 - Skip logging for a refused CDS (issue #570)

A three-gene stub genome (`ATGAAATAA`, empty, `ATGATG`) builds the dataset under `tmp_path`. Asserted: the exact two WARNING lines (`skipping YAA002W: Empty CDS string; a codon frequency needs at least one codon.` and `skipped 1 gene(s) whose CDS has no codon frequency`), the stored ids `YAA001W`, `YAA003W`, a `KeyError` for `YAA002W`, and the exact 64-codon vectors. A genome with only valid CDS logs nothing.
