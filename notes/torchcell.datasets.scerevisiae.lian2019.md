---
id: iqhtiqrvf77x9ojrqzbjk2g
title: Lian2019
desc: ''
updated: 1783919988175
created: 1783919988175
---

## 2026.07.13 - Lian 2019 MAGIC CRISPRa/i/d furfural screen (per-guide, SRA-reprocessed)

`CrisprMagicLian2019Dataset` -- the FIRST dataset on the AXIS-4 CRISPR expression-modulation
ontology (`[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]`). 266,415
(guide x round) `EnvironmentResponseExperiment` records.

### Why the phenotype had to be reprocessed from raw NGS

Lian released only the DESIGNED guide library (Supp Data 1-3, design scores) + the guide
reference (Supp Data 4). The furfural per-guide ENRICHMENT (Fig 2a/c/e) is **not in any
supplement** (excluded from the Source Data file), and the in-house Carl-Schultz re-analysis
we had was collapsed to per-GENE (modality lost). Modality lives at guide resolution
(orthogonal Cas), so a true CRISPRa/i/d dataset required per-GUIDE counts -> reprocess SRA.
Full trail: memory `[[lian2019-magic-data-availability]]`.

### Reprocessing pipeline (scripts in the library mirror `.../lianMulti.../data/`)

1. `download.sh` -- 21 SRA runs (PRJNA504483): 3 rounds x {before/after} x triplicate +
   ecLibA/I/D plasmid baselines (2 runs pulled direct from ENA after prefetch 404).
2. `count_guides.py` -- barcode = read[27:70] (43bp activation) | read[27:71] (44bp
   interference/deletion), forward, offset-27 (empirically scanned); exact-match to the
   100,493-guide reference (sha256 4e3f225a...). Mapping 74-78% furfural / 58-74% plasmid.
   Barcode collisions (6,666 guides) are ALL same-gene+same-modality -> safe to collapse.
3. `enrichment.py` -- CPM(+1)/library; per round per replicate log2(after/before); mean+-SD
   over 3 triplicates. **Validated vs paper hits: PDR1i r3 rank 1, SLX5i r1 rank 1, SAP30d
   r1 rank 2.** Round 2 is the paper's noisy/synergistic round (RCF1a rank137, NAT1a mid).
4. `guide_enrichment_final.tsv` (sha256 f9af849f...) = the loader's pinned input.

### Record model

- **genotype (per-guide)**: library member as `CrisprActivation/Interference/Deletion`
  (target gene + spacer + effector dLbCas12a-VP/dSpCas9-RD1152/SaCas9) PLUS the round's
  integrated background (R2 +SIZ1i=YDR409W; R3 +SIZ1i+NAT1a=YDL040C, guide unspecified).
  So R1 = 1-pert, R2 = 2-pert, R3 = 3-pert mixed-modality combos.
- **environment**: furfural 5/10/15 mM (by round), SED/G418 liquid, 30 C, aerobic.
- **phenotype**: `EnvironmentResponsePhenotype` log2_ratio, response = mean log2FC,
  uncertainty = SD (sample_sd, n=3 -> SE=SD/sqrt(3)); reference = no-enrichment (log2FC 0).

### Drops (documented, counted)

300 random controls + 16 source-corrupted-gene guides (Excel date/serial artifact identical
in reference AND library -> unrecoverable) + 2,633 unresolved-gene guides (165 ncRNA/rDNA
genes absent from the ORF genome); 26,169 (guide,round) undetected; 48 guides skipped in the
round where they target their own background.

### Verification note (shared code touched)

L1 strain identity (`_genotype_signature`) was extended to be GUIDE-AWARE: sibling guides of
one (gene, mode) are distinct strains (like a TS-allele series), keyed by
`crispr.guide_sequence`. `background_genes` is EMPTY for this dataset -- the per-round
background is genuine genotype content, not a constant to subtract.
