---
id: 616frtkpzgesxck5b7ex8ep
title: Expression_ceiling_all
desc: ''
updated: 1789451149203
created: 1789451149203
---

## 2026.09.15 - The ceiling of every gene-level panel from its own reproducibility

`experiments/019-simb-multimodal/scripts/expression_ceiling_all.py` writes `results/expression_ceiling_all.json` and the document table `tables/expression_ceiling_all.tex`. Definition (the strand's): `ceiling_g = sqrt(rel_g)`, averaged over genes; `rel_g` by test-retest (per-gene Pearson across strains between two independent measurements) or by decomposition (`1 - noise_var / total_var`, noise from replicates of one genotype, total across the panel), clipped to [0, 1].

| panel | test-retest r | ceiling (test-retest) | ceiling (decomposition) | reported |
|---|---|---|---|---|
| Kemmeren 2014 / Sameith 2015 | 0.61 (82 shared deletions) | 0.77 | 0.86 | design only, no r |
| Caudal 2024 (943 isolates) | 0.40 (29 re-cultured isolates) | 0.60 | 0.46 | 0.94 per sample (ours 0.947) |
| Messner 2023 (4,476 deletions) | 0.20 (146 duplicated-origin ORFs) | 0.40 | 0.70 (paper CVs give 0.72) | CV 8.1 / 11.3 / 16.2 percent, no replicates |
| Zelezniak 2018 (97 kinase deletions) | -- | -- | 0.92 | CVs as a figure |
| Nadal-Ribelles 2025 | 0.04 across batches (null 0.02) | 0.21 (null 0.15) | -- | DEG counts vs Kemmeren |

Sources read: `results/expression_ceiling_replicate.json` (Kemmeren), the mirrored Caudal replicate table `replicate_data_tpm_22042023.tab` (60 samples, 29 isolates, comma-separated despite the suffix) plus the 943-isolate LMDB, the converted `fig3_proteome` records (WT SE on the log2 scale, n 388), the Zelezniak LMDB (per-strain SE, 2 to 7 cultures), and the 028 results `proteome_messner_replicates.json` and `nadal_replication_coverage.json`. Each paper's reproducibility statement is carried verbatim with its `paper.md` line. Caudal's two estimators disagree (0.60 vs 0.46) because 29 pairs determine neither well; the Messner decomposition assumes knockout noise equals wild-type noise. Document: [[experiments.019-simb-multimodal.expression-strand-retrospective]], section "The ceiling of every panel".
