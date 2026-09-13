---
id: 4vp0a9tm13r73od3gy8bpqw
title: Xue2025
desc: ''
updated: 1783966590843
created: 1783966590843
---

## 2026.07.13 - Xue 2025 in-house combinatorial FFA-titer dataset

`FattyAcidXue2025Dataset` -- FIRST COMBINATORIAL-genotype metabolite dataset (up to 6
deletions/strain). 176 records + 1 measured WT reference, from the in-house Xue 2025 FFA
overproduction screen.

- **Genotype**: every non-WT strain = the `POX1-FAA1-FAA4` FFA-overproduction baseline
  (3 deletions, implicit in ALL strains) + 0-3 TF deletions decoded from the genotype string
  (`<letters> <N>Δ`, letters via the Abbreviations sheet, `#TF + 3 == N`). `+ve Ctrl` =
  baseline only. One `KanMxDeletionPerturbation` per gene (**markers unknown/mixed for
  in-house combos -> KanMX is a documented REPRESENTATIVE**; state=absent is the asserted
  fact -- can't stack 6 KanMX). 13 genes all resolve to R64. Decode source =
  `experiments/008-xue-ffa` parser.
- **Phenotype**: `MetabolitePhenotype`, `metabolite_level={C14:0,C16:0,C18:0,C16:1,C18:1: mg/L}`,
  `measurement_type="titer_mg_per_l"`, `metabolite_level_se` = sample SD/sqrt(n) PER FFA (n
  computed from actual non-blank reps: mostly 3; 10 strains n=2, `G-O-T 6Δ` n=1 -> SE NaN,
  which the verifier skips). `reference_centered=False`. `target_metabolite_ids=None` (defer
  ChEBI/Yeast9). Env = SC/30C aerobic (in-house-assumed, flagged).
- **Reference**: measured wt BY4741 titers.

**Verifier change**: metabolite L1 `orf_uniqueness` -> `genotype_uniqueness` (keys on the full
deletion-set signature, not per-ORF) so combinatorial strains where an ORF recurs across many
strains pass; backward-compatible (single-deletion datasets reduce to per-gene; confirmed on
isobutanol). L0-L4 all pass (176 records).

**FLAGGED**: (1) in-house/unpublished -> `Publication` anchors to the FFA-chassis paper
Runguphan & Keasling 2014 (PMID 23899824); `Provenance` points to the in-house xlsx (sha256
023de80e). (2) KanMX marker is a representative (real markers mixed/unknown). (3) SC + 30C
assumed. See memory `[[remaining-datasets-blocked-status]]`.

## 2026.09.13 - Expression correlates for the ten TF deletions (idea, not run)

Overlap of the 13 FFA genes (POX1, FAA1, FAA4 chassis plus the ten TFs) with the
knockout-expression and proteome panels, counted from the LMDBs:

| panel | FFA genes present as single deletions |
|---|---|
| Kemmeren 2014 (mRNA) | FAA4, FKH1, GCN5, OPI1, RFX1, RPD3, SPT3, YAP6 (7 of the 10 TFs; MED4, RGR1, TFC7 absent) |
| O'Duibhir 2014 (growth rate, same lab) | the same eight |
| Messner 2023 (protein) | POX1, FAA1, FAA4, FKH1, GCN5, OPI1, RFX1, RPD3, SPT3, YAP6 |
| Sameith 2015 singles / doubles | FKH1, YAP6 / no double inside the set |
| Nadal-Ribelles 2025 | POX1, FAA1, FAA4, RFX1, SPT3, YAP6 |

Two uses, both unrun. (1) The seven single-TF profiles in Kemmeren against the single-TF
FFA effects in the chassis: which expression signature a TF deletion carries, and whether
the FFA-pathway genes (the identify_ffa_reactions set) move in it. (2) An expression
model trained on Kemmeren singles (and Sameith doubles) can predict a profile for each of
the 175 combinatorial strains, none of which has measured expression; the predicted
profile is then a readout that can be tested against the measured FFA titers, which is
the only way to see expression on the doubles and triples without a new experiment.
Caveat: the FFA strains sit in the POX1 FAA1 FAA4 chassis, which no expression panel
carries as a background, and the three absent TFs (MED4, RGR1, TFC7) are essential in
S288C, so the Xue strains carry something other than a clean deletion for those.
