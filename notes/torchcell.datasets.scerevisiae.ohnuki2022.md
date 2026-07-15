---
id: jl6ylsecd9dpcx6ot4nedws
title: Ohnuki2022
desc: ''
updated: 1783884847422
created: 1783884847422
---

## 2026.07.12 - Ohnuki 2022 quadruple-mutant morphology build

`ScmdOhnuki2022Dataset` (`torchcell/datasets/scerevisiae/ohnuki2022.py`). Ohnuki et al.
2022, *npj Syst Biol Appl* 8:3, doi:10.1038/s41540-022-00212-1, PMID 35087094. CalMorph
morphology of gene-deletion strains in a drug-hypersensitive 3Δ background →
`CalMorphPhenotype`. **L0-L4 verified, 1,979 records.** (Compound side NOT built — 7
compounds, no released morphology vectors.)

- **Source (SCMD2, mirror + hash-pin):** `quad1982data.tsv` (1,982 quadruple mutants,
  sha256 `5a1d4500…`) + `wt749data.tsv` (749 3Δ reps, sha256 `4603aadf…`).
- **FULL-FIDELITY 3Δ genotype (reuses Vanacloig 2022's convention for the same background):**
  each strain = 4 deletions — target `KanMxDeletionPerturbation` + PDR1(YGL013C)
  `NatMxDeletion` + PDR3(YBL005W) `MarkerDeletion(KlURA3)` + SNQ2(YDR011W)
  `MarkerDeletion(KlLEU2)` (verified vs strain Y13206). **FLAG for review:**
  `ExperimentReference` can't hold the 3Δ background genotype, so ReferenceGenome=BY4741
  placeholder while the reference PHENOTYPE is the measured 749-rep 3Δ average — an
  asymmetry shared with Vanacloig 2022.
- 3 dropped: YGL141W (NaN in 2 CV traits → dropped whole, no imputation), PDR1/SNQ2 as
  targets (already in background). Env YPD 25 C liquid (sourced). 501 CalMorph traits.

## 2026.07.15 - Migrated to the shared layered gene-name resolver

Target ORF names now reconcile through `SCerevisiaeGenome.resolve_gene_name` (the layered
resolver on main, [[torchcell.sequence.genome.scerevisiae.s288c]], PR #98), the same
retain-all and collision-safe pattern as [[torchcell.datasets.scerevisiae.ohya2005]].
Previously the
loader did NO R64 reconciliation (bare uppercase), so stale target names were stored
verbatim.

- Added a `genome` ctor param (lazy-built if not injected) and `_reconcile_orf_names`;
  reconciliation runs AFTER the NaN-completeness drop and BEFORE the 3Delta-background
  collision drop (so a legacy target that maps onto a background gene is still caught).
- Rebuild result: **1979 records, unchanged** (1982 raw − 1 NaN `YGL141W` − 2 background
  `YGL013C`/`YDR011W`). Reconciliation over 1981 unique names: 1970 `current`, **7
  `renamed`** now stored under their current R64 id, 3 `non_gene_feature`, 1 `retired`.
  - **7 remapped:** `YAL058C-A→YAL056C-A`, `YGR272C→YGR271C-A`, `YIR020W-B→YIR020W-A`,
    `YLR391W→YLR390W-A`, `YML010C-B→YML009C-A`, `YML013C-A→YML012C-A`, `YML033W→YML034W`.
  - **3 non_gene_feature (retained):** `YDR134C` (blocked_reading_frame), `YFL056C`
    (pseudogene), `YIR043C` (blocked_reading_frame).
  - **1 retired (retained verbatim):** `YAR037W`.
  - 0 merge-collisions.
- L0-L4 verifier PASS (`run_ohnuki_morphology`): 1979 records; genotype
  (`kanmx_deletion` target + 3Delta `natmx/marker` background) and round-trip unchanged.
