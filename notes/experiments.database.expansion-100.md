---
id: ze2zee24rnjw0puk5m21hs1
title: Expansion 100
desc: ''
updated: 1787708931127
created: 1787708931127
---

## 2026.08.25 - Candidate list for datasets 50-100

Working note for `notes-tex/database-expansion-100/`. The typeset document is the
deliverable; this note holds the decisions behind it and the follow-ups it generated.

- Generator: `experiments/database/scripts/build_candidate_datasets_table.py` (the curated
  list lives in the script as pydantic records, since a curation cannot be recomputed from
  a store).
- Document: `notes-tex/database-expansion-100/main.pdf`, published to Zotero under
  `torchcell / notes-tex / database-expansion-100`.
- Machine-readable dump: `experiments/database/results/candidates/candidate_datasets.json`
  (gitignored, regenerate with the script).

### What was decided

**Scope.** *S. cerevisiae* only. 74 candidates, none of them among the 49 already built.
The top 51 is the recommended set because 49 + 51 = 100.

**The hard gate is sequence reconstruction.** Every row names a `sequence_basis` saying how
the total genomic content of one strain would be rebuilt. Genome shuffling and ALE without
resequencing fail it and are dropped rather than ranked, since the progeny differ from
their parents at unmapped positions.

**Ranking is on measurements, not instances.** Instances times phenotype dimensionality.
Sorting on instances alone put Muenzner 2024 (796 isolate proteomes, so roughly 1.6e6
numbers) below mid-sized scalar fitness screens. Same normalization the supported-datasets
table applies through its gzip-signal column.

**Tier bars are applied to content, not reputation.** Piotrowski 2017 has the largest
compound library in yeast and fails tier 1 on 157 strains. Turco 2023 has the widest
environment axis and fails because it is a meta-aggregation whose net-new fraction is
unknown.

### Two bugs found while building the ranking

Both were caught by reading the emitted order rather than by a test:

1. Ranking on instances alone misordered every vector-valued omics row. Fixed by the `dim`
   field and `Candidate.measurements`.
2. The requested-row pin displaced by measurement count alone, which evicted **Puddu 2019**
   -- the whole-collection WGS that every `S288C-KO` sequence basis in the table rests on.
   Fixed by displacing on `(-tier, measurements)`.

### Corrections to the 2026.07 triage pass

- **Its top ten is stale.** Nine of ten have been built since (Vanacloig-Pedros, Messner
  2023, Mulleder 2016, MAGIC/Lian 2019, Mormino, Hoepfner, Nadal-Ribelles 2025, Cachera,
  Ozaydin). Only Lee 2014 is outstanding, and it is row 1 here.
- **Anglada-Girotto 2022 is *E. coli*.** Verified from the mirrored PDF
  (`$DATA_ROOT/torchcell-library/anglada-girottoCombiningCRISPRiMetabolomics2022/paper.md`).
  It had been floated as a CRISPRi-plus-metabolomics candidate.
- **The CABBI 13C-MFA repository is 16 strains and mostly K-FIT model output**, not a
  genome-scale measurement. Blank 2005 supersedes it on both scale and directness.

### New finds not in the triage note

- **Dutta 2026** (Schacherer), Nat Commun, 520 barcoded natural isolates x >600 compounds.
  Natural sequence diversity crossed with chemogenomics on isolates from the 1,011 panel,
  so it joins to Caudal 2024 transcriptomes and Muenzner 2024 proteomes on one genotype axis.
- **Hale 2024** (Kruglyak), Nat Commun, 8,046 CRISPRi guides x 1,721 genes x 169 sequenced
  segregants. SRA PRJNA986287. Measures perturbation-by-background interaction directly.
- **Dong 2021** (Zhao lab, Jiazhang Lian a co-author), Metab Eng. MAGIC read out through a
  SAM biosensor by FACS instead of a growth selection. This is the biosensor MAGIC screen.
- **Kuroda 2019** and **Liu 2021**: two independent genome-wide YKO isobutanol-tolerance
  screens. The earlier assumption that none existed was wrong.
- **Pereira 2014** (wheat-straw hydrolysate), **Endo 2008** (vanillin), **Xiao 2014**
  (furfural RNAi): the three named recalcitrant-biomass inhibitors that had no dedicated
  screen in the built set.

### Open items, in priority order

1. **Trikka 2015 is row 51 and may not be ingestible.** Recorded as figure-only. Confirm
   Additional file 1 carries per-strain scores before committing the slot; if it does not,
   Chang 2012 comes back off the reserve.
2. **Turco 2023 needs a de-duplication pass** against this table and the built set before
   its 3.8e7 instances mean anything.
3. **Accession confirmation.** Six were verified live (GEO GSE123118, SRA PRJNA986287,
   PRIDE PXD048219, Dryad for Chica 2026, ChemGRID, BioProject PRJNA379146). The rest are
   claims from data-availability statements.
4. **The synergy pairs are hypotheses, not results.** Five transfer tests are named in the
   document (Cooper vs Mulleder, Liu vs Kuroda, Hale and Galardini, McGlincy vs
   Momen-Roknabadi, Caudal vs Muenzner). None has been run.
5. **Malonyl-CoA still has no direct measurement** anywhere in the built set or this list.
   See [[metabolism.central-carbon-precursors]].

Related: [[paper.north-star]] · [[paper.north-star.dataset-triage]] ·
[[paper.supported-datasets-and-databases]] · [[metabolism.central-carbon-precursors]] ·
[[experiments.024-perturb-seq-costing.method-review-and-costing]]
