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

## 2026.07.15 - Add shared-resolver ORF reconciliation (retain-all naming)

The target ORF name is now reconciled to current R64 via the shared
`reconcile_systematic_names` helper (see
[[torchcell.datasets.scerevisiae.gene_name_reconcile]]) before the NaN + 3Δ-background
drops. Previously the loader stored raw ORF names with no R64 reconciliation. Build unchanged
at **1979** (1982 − 1 NaN row YGL141W − 2 background-collision rows), but 7 target names are
now correctly remapped to current R64 ids and 1 (YAR037W) retained as a legacy name. The
3Δ-background collision check now runs on the reconciled name (catches a legacy alias of a
background gene too). Added an optional injectable `genome`.
