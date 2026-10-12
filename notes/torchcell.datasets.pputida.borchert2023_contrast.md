---
id: wbr4850ybt5mbwq8zb67koo
title: Borchert2023_contrast
desc: ''
updated: 1791837915366
created: 1791837915366
---

## 2026.10.12 - The 64,853 released contrasts, on a new fitness-contrast leaf

Borchert, Bleem and Beckham 2023 (`borchertRBTnSeqIdentifiesGenetic2023`), issue #913. `FitnessContrastBorchert2023Dataset`, slug `fitness_contrast_borchert2023`, 64,853 records, one per released (gene, comparison sheet) row of the 13 comparison sheets of `si1.xlsx` (sha256 `b80c6866...`).

**The leaf.** `FitnessContrastPhenotype` is a sibling of `EnvironmentResponsePhenotype` under `Phenotype`, with its own `BacterialFitnessContrastExperiment` / `...Reference` pair. Measured against #880's p-value pair first, as the issue asked: a contrast record on `EnvironmentResponsePhenotype` would need a released contrast value (the release gives only each arm's mean), has room for one adjusted p-value where the release gives two q-values, and would turn a per-culture test into a two-arm test. The record's environment is the condition arm and the reference's the reference arm. The four statistics are verbatim: the Methods name the test ("Comparison of mean fitness values between enrichment and medium reference cultures ... was performed using a two-sample $t$ test, where the $p$ value was corrected for multiple testing via the positive false discovery rate (pFDR) method") and the adjustment ("pFDR $q$ values were then adjusted for monotonicity (Yekutieli and Benjamini, 1999), and both unadjusted and adjusted $q$ values are reported."). 582 released `q-value` cells exceed 1 and are stored as released. The reference's statistics are typed gaps.

**Identity.** The released `old_locus_tag` and `gene_name` (the tag where the release names no gene, 53,091 rows) as a transposon insertion of `Putida_ML5_JBEI`; no annotation lookup.

**Environment.** Each arm's environment from the sibling loader's medium and stressor, with `duration_hours` a typed gap: the contrast pools three replicate cultures whose Table S4 durations differ.

**Finding, measured and not acted on.** On all 64,853 rows the released t-statistic has the sign of (first-named arm mean minus second-named arm mean), so a positive t marks a gene whose fitness is lower in the condition arm. The Methods do not state the orientation.

**Candidate gate.** G4 refuses the verdict: Borchert 2024's module names this paper's BioProject (PRJNA856070), and the gate reads any shared accession as "already held", although the contrasts are in no store. Owner question in the PR.
