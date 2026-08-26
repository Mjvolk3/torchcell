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

**Scope.** *S. cerevisiae* only. 73 candidates, none of them among the 49 already built.
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

### 2026.08.25 later - two rows are not untouched candidates

Caught by asking whether Lee 2014 was already held. Checking a name against the built 49 is
NOT sufficient: a dataset can carry a loader, or a failed retrieval attempt, without showing
up there. Added a `status` field (`candidate` / `blocked` / `loader-in-flight`) so this
cannot recur, rendered as a superscript B or L in the table.

- **Lee 2014 (row 1) is BLOCKED, not available.** WS15 already attempted it;
  [[plan.schematization-ingestion-roadmap.2026.06.23]] line 42 records "Remaining: Lee 2014
  (awaiting author matrices)". The Science SI and the Nislow/Giaever portal did not yield
  per-strain values. The rank is still earned on scale, but row 1 is an **author request,
  not a loader**, and its 1.3e7 instances are not reachable work.
- **Turco 2023 (row 20) already has a loader.** `torchcell/datasets/scerevisiae/yeastphenome.py`
  (commits `cc72795e`, `80f51193`) consumes the curated YeastPhenome release and retains 49
  growth screens. Not in the built 49, so it belongs in this list, but as in-flight rather
  than net-new. Its de-dup problem is partly solved already: the loader excludes primaries
  built directly, plus heterozygous-diploid and expression screens.
- **YeastPhenome does not backfill Lee 2014** -- PMID 24723613 is not among its 49 screens.

Consequence for the headline number: of the 5.7e7 instances in the recommended 51, Turco
contributes 3.8e7 (not net new) and Lee 1.3e7 (not reachable). **Net new and in hand is
about 6.8e6** (figure updated after the Wildenhain removal below).

### 2026.08.26 - Wildenhain 2016 was already built, under the 2015 name

Third instance of the same failure mode, and the one that broke my de-dup check. Caught by
the user asking "pretty sure we have already listed wildenhain 2016".

**It is a data descriptor, not a second experiment.** Its own Data Citation 1 (paper.md
line 176 of the mirror) is **NCBI PubChem BioAssay AID 1159580** -- exactly what
`torchcell/datasets/scerevisiae/wildenhain2015.py` already ingests. Both report 242 strains
and 492,126 interaction tests. Row removed; list is now 73.

**Why my automated cross-check missed it:** I grepped candidate DOIs and accession tokens
against the built loaders. Wildenhain 2016's DOI is not cited in the loader, and I had
written its accession as "ChemGRID + PubChem BioAssay" WITHOUT the AID number, so there was
nothing to match on. The loader is named for a *different paper* than the deposit it loads.
A name-vs-built-list check cannot catch that; only comparing **accessions** can.

**Knock-on:** freeing the slot let Chang 2012 back in, so Trikka 2015 now reaches row 51 on
merit and NO pin is in force. The pin machinery stays for when one binds again.

**Separate defect to fix in the built dataset (not this branch).** `wildenhain2015.py`
ingests the EXTENDED matrix (242 strains, 5,518 compounds) but is filed under the 2015 paper,
which reports 195 sentinels and 4,915 compounds. The loader half-flags this itself. Per the
sourcing rule that a value carries the citation it was read from, the released matrix should
cite the 2016 Sci Data descriptor with 2015 as the original subset.

Also checked and NOT duplicates: **Lian 2017** (lian2019.py names CRISPR-AID only as the
MAGIC chassis, it does not load the 2017 beta-carotene data) and **Mulleder 2012**
(mulleder2016.py already cites it for the prototrophic background; kept in the reserve
because that loader records the prototrophy-restoring markers as an unmodeled GeneAddition).

### 2026.08.26 - row 47 vs SynLethDB, and the overlap checker

User asked whether row 47 (Sharifpoor 2012, kinome SDL) is already inside SynLethDB.
**It is not**, and the absence is structural.

- PMID 22282571 is in NEITHER SynLethDB layer (SL 13,999 rows / 1,740 source PMIDs; SR
  1,918 source PMIDs). Checked against the raw CSVs at
  `$DATA_ROOT/data/torchcell/syn_leth_db_yeast/raw/Yeast_SL.csv` and the SR sibling.
- **Why it cannot be there:** SynLethDB models synthetic LETHALITY and synthetic RESCUE,
  both two-loss-of-function. Sharifpoor's contribution is synthetic DOSAGE lethality --
  overexpression of one gene in a deletion background of another. That pairs a
  gain-of-function with a loss, which is outside the DB's scope by construction. Confirmed
  from the schema: our loader emits only `SyntheticLethalityPhenotype` and
  `SyntheticRescuePhenotype`, and the raw CSV has no interaction-type column.
- This strengthens rather than weakens the row: the overexpression x deletion axis is
  carried by neither SynLethDB nor Costanzo/Kuzmin. `why` updated to say so.
- Also checked and absent: Decourty 2021 (34358317), Braberg 2020 (33303586).

**Built the checker** promised after the Wildenhain miss:
`experiments/database/scripts/check_candidate_overlap.py`. Three keys of increasing
strength -- DOI in a built loader, accession token shared with one, PMID re-served by an
aggregator. DOI->PMID via the NCBI ID converter, cached to
`results/candidates/doi2pmid.json` so `--offline` re-runs cost nothing. Exits non-zero on
a finding.

Current run: 1 overlap (Turco, already known), 0 SynLethDB overlaps.
**Coverage is the weak point: 52/73 rows resolve a PMID and only 8/73 carry a parseable
accession.** Accession is the strongest key and the one that would have caught Wildenhain,
so it is running on ~11% of the table. Recording each row's deposit identifier is the
highest-value edit left.

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

0. **Record every candidate's ACCESSION, not just its DOI, and de-dup on that.** Three of
   three duplicate/blocked finds this pass were invisible to a name check and two were
   invisible to a DOI check. The only reliable key is the deposit a loader actually reads
   (PubChem AID, GEO, PRIDE, SRA). Worth a small checker over the built loaders before the
   next triage pass.
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
