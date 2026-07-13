---
id: 5q5cofs8uz78xralky1tnv0
title: Lopez2024
desc: ''
updated: 1783923943375
created: 1783923943375
---

## 2026.07.13 - Lopez/Montaño-López 2024 isobutanol biosensor YKO screen (in-house)

Two `MetabolitePhenotype` datasets from the in-house isobutanol GFP-biosensor screen of the
BY4741 gene-knockout collection (José de Jesús Montaño López 2024 Princeton dissertation).

- **`IsobutanolScreenLopez2024Dataset`** (`isobutanol_screen_lopez2024`) -- first genome-wide
  screen, **4554 records** (one per current-R64 ORF; 186 ORFs aggregated from >=2 plate rows,
  mean FC, n=row count, SE=SD/sqrt-n when n>=2).
- **`IsobutanolValidatedLopez2024Dataset`** (`isobutanol_validated_lopez2024`) -- validated
  re-screen (FC>=2 or FC<=0.5) in **triplicate**, **224 records** (n=3, SE=STD/sqrt3).

**Phenotype**: `metabolite_level={"isobutanol": FC}`, `measurement_type=
"biosensor_gfp_fluorescence_fold_change"`. **FC (verbatim)** = median GFP fluor(deletion) /
median fluor(WT control, same plate); reference = WT = 1.0. Biosensor = GFP alpha-
ketoisovalerate/isobutanol-pathway sensor (Leu1 promoter), integrated in every strain
(constant reporter background -- documented, not a per-record perturbation). **Genotype** =
one `KanMxDeletionPerturbation` per ORF; **environment** = SC liquid, 30 C (standard), aerobic,
no inducer (biosensor reports endogenous pathway flux). L0-L4 all pass.

**Provenance**: sha256-pinned dissertation supplementary tables (`supplementary_tables.xlsx`
f97cf13c...) + `thesis.pdf` (525e03b4...) in the mirror; thesis retrieved from Zotero item
6DQAEFWZ. Methods sourced born-digital (pdftotext Ch 3.3).

**FLAGGED for review**: (1) the dissertation has NO DOI/PMID (only Princeton DataSpace handle
88435/dsp019s161956t); `Publication` requires doi|pubmed_id so it cites the same-lab biosensor
methods paper (Montaño-López et al. Nat Commun 2022, PMID 35022416) -- the Provenance record
points to the true dissertation source. Consider a schema affordance for DOI-less dissertation
sources. (2) `YBL071W-A` dropped from S3 (listed in both up FC=3.469 and down FC=0.0757 blocks
-- contradictory). (3) temperature 30 C = standard, not explicitly stated. See memory
`[[remaining-datasets-blocked-status]]`.
