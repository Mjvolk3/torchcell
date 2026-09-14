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
- Document: `notes-tex/database-expansion-100/database-expansion-100.pdf`, published to Zotero under
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

### 2026.08.26 - expanded to 155 candidates, target now 200

Scope changed: 49 built + 151 = 200, so CUT=151 and the list is 155 (4 in reserve).

**Perturb-seq criterion rewritten.** The old boolean flagged any library. The real criterion
is two axes: **input dimensionality** (combinatorial or background-crossed perturbation --
a big single-perturbation library is NOT this, it is one edit per cell sampled widely) and
**output dimensionality** (transcriptome or per-cell distribution, not a scalar). Field is
now `PertSeq = none|input|output|both`. Momen-Roknabadi de-flagged accordingly.

**The key finding: the both-axes quadrant is nearly empty.** Of 155 rows, 15 input, 15
output, **3 both** (Boocock 2025, Albert 2018, N'Guessan 2025) -- and all three get there via
recombinant natural genotypes x transcriptome, none by designed perturbation. The largest
genotype-indexed transcriptome with designed perturbations is the built Kemmeren 2014: 1,484
SINGLE deletions. Sameith 2015 adds 72 doubles. **That gap is the argument for running the
experiment, and also its spec: combinatorial perturbation per cell + transcriptome readout.**

**Three new classes for depth** (the "deep not shallow" ask): Regulatory DNA (8), Deep
mutational scan (9), Combinatorial genome (3). All sequence-exact per genotype.

**Promoter datasets** (the one half-remembered): de Boer 2020 (>1e8 random promoters,
largest sequence-to-phenotype dataset in yeast), Vaishnav 2022 (fitness landscape + natural
isolate variants), Renganaath 2020 (5,832 REAL promoter alleles from natural variation, 451
causal). Keren 2013 measures ~900 native promoters directly across conditions. The most
likely referent is Renganaath or Vaishnav; both are in.

**Acetic acid.** Tolerance is now covered by four perturbation types (Mira 2010 deletion,
Mukherjee 2021 CRISPRi-essential, Sousa 2013 overexpression, built Mormino 2022 biosensor).
Production: **Peeters 2021 measures acetate titer across 1,125 sequenced segregants** -- the
only quantitative genotype-to-acetate map at scale -- plus Chang 2012 (ordinal halo, whole
collection). **No genome-wide screen with a quantitative per-strain acetate titer was found.**

**Table 1 caption now shows the glyph** rather than the word "bullet".

**New `confidence` field.** 73 `sourced` / 82 `recall`. Recall rows are real datasets with
sound descriptions but unverified citation details, counts and accessions. Ranking puts
several recall rows at the very top (de Boer, Boocock) because they genuinely are the
largest -- correct ranking, wrong reading if the flag is ignored.

**The checker found its first duplicate before I did:** O'Duibhir 2014 expression. Its DOI is
cited in `oduibhir2014.py`, whose docstring says the paper's expression data IS the Kemmeren
compendium (already built) and only the growth readout was new. Removed. That is 4 duplicate/
blocked finds total, and the first one a script caught rather than the user.

**Overlap coverage got relatively worse:** 56/155 rows resolve a PMID, 9/155 a parseable
accession. Accession is the strongest key and now runs on ~1 row in 17.

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

## 2026.09.12 - Re-triage: bands before scale, waves of 50 and 70, and a synergy table

Driven by a request to bring the CRISPRi work to the top, to sweep
`jacksonGeneRegulatoryNetwork2020` and the `microbe-perturb-seq` collection for missing
candidates, to re-triage the top 50 with a 51-70 bench, and to list the synergies between
every candidate and either a supported dataset or another candidate.

### What changed in the generator

`experiments/database/scripts/build_candidate_datasets_table.py`:

- `BUILT_COUNT` 49 -> 50 (Bloom 2019 built as the 50th), so `CUT` is 150.
- New `WAVE_1 = 50` and `WAVE_2 = 70`. The candidates table now carries three divider
  rows: end of wave 1, end of wave 2, and the long-run 150 cut.
- New `Band` literal and `BANDS` table: `perturb-seq` (28 rows), `metabolism x expression`
  (18 rows), `scale` (115 rows). `sort_key` is now `(band, tier, -log10(measurements))`, so
  band is applied before tier and before scale. Every banded row carries a `band_why` and
  the script refuses to run without one.
- New `Synergy` model and `SYNERGIES` table: 101 named joins across 59 candidates, each
  with a join key and what the join yields. 56 of the 101 partners are already built.
  `_apply_curation()` validates every partner name against `CANDIDATES` or against
  `SUPPORTED_PARTNERS` (the names in `build_supported_datasets_table.py`), so a renamed
  partner is a startup failure rather than a silent claim.
- New `previous_ranked()` and `moves()`. Bloom 2019 stays in `CANDIDATES` with
  `status="built"` so the previous pass's ranking reproduces exactly and its departure
  shows up as one recorded move instead of shifting 147 ranks by one.
- New tables: `synergies.tex`, `swaps.tex` (now the 91-row move table), `pins.tex` (the old
  pin table, unused while no pin binds). `counts.tex` now splits by wave and adds a band
  block.

### Ranking outcome

Wave 1 is 28 Perturb-seq rows, 18 metabolism-and-expression rows and 4 scale rows. Top ten:
Boocock 2025, Hale 2024, Jackson 2020, Puddu 2019, N'Guessan 2025, Hackett 2020, Hu 2007,
Dong 2021, Momen-Roknabadi 2020, McGlincy 2021. The cost of banding is visible and stated in
the document: de Boer 2020 falls 2 -> 47 and Lee 2014 4 -> 49, both still inside wave 1.

### New candidates and new exclusions

Three rows added: Airoldi 2016 (from the Jackson 2020 citation list, nitrogen-limited
chemostat transcriptome on the same media as Jackson's NLIM conditions), Jariani 2020 and
Urbonaite 2021 (yeast single-cell platforms from the collection sweep). All three are
`recall` confidence with unfetched accessions.

Eight exclusion rows added: Bloom 2019 (already built), the four TF-target prior networks
Jackson 2020 uses, the Tchourine 2018 bulk compendium, Scholes 2019, Brandner 2025
(mapSPLiT, the only microbial both-axes Perturb-seq found, but a preprint with no deposited
accession), mammalian Perturb-seq, bacterial single-cell atlases, and the collection's
reviews and statistics papers.

### Caveats that carry forward

- `recall` is now 86 of 161 rows and three of the first twenty are `recall` (Hackett 2020,
  Hu 2007, Lenstra 2011). Verify before writing a loader.
- Jackson 2020's authoritative 72-strain genotype list is Supplementary file 1 Table S2, an
  Excel file not in the mirror. The released matrix carries 38,225 cells; the abstract says
  38,285 and the discussion 38,255.
- The overlap check still resolves a PMID for only 56 of 161 rows.

## 2026.09.13 - Transcript against protein: seven rows, and the band widened

Follow-up driven by a sister session assembling gene-level RNA-versus-protein correlates.
The ask was that a strain-level or gene-level pairing of transcript, protein, translation
and turnover be visible in the ranking.

### Citations verified before any row was written

None of the seven papers is in the mirror, so every citation was verified against PubMed
E-utilities on 2026-09-13 rather than written from memory. Author lists, venues, volumes,
pages and DOIs are from that lookup; deposits were NOT fetched.

| row | citation | PMID | key counts (from the abstract) |
|---|---|---|---|
| Grossbach 2022 | Mol Syst Biol 18:e10712 | 35574625 | 112 strains; transcriptome + proteome + phosphoproteome |
| Teyssonniere 2024 | PNAS 121:e2319211121 | 38696467 | 942 isolate proteomes; eQTL/pQTL overlap 3% |
| Foss 2007 | Nat Genet 39:1369-1375 | 17952072 | segregant count NOT stated in the abstract |
| McManus 2014 | Genome Res 24:422-430 | 24318730 | 5,474 orthologs; cerevisiae, paradoxus, F1 |
| Martin-Perez 2017 | Cell Syst 5:283-294.e5 | 28918244 | 3,160 protein turnover rates |
| Sun 2013 | Mol Cell 52:52-62 | 24119399 | 46 deletion strains; synthesis + decay rates |
| Hughes 2000 | Cell 102:109-126 | 10929718 | 300 mutations and chemical treatments |

Two premises in the request did not survive the check and the rows say so:

- Sun 2013's 46 strains are deletions of mRNA degradation and metabolism genes. The
  abstract does not say they came from the Kemmeren collection, so the row records the
  overlap as gene identity and flags the provenance as unconfirmed.
- Martin-Perez is correct as Martin-Perez M and Villen J, Cell Systems 2017.

### Teyssonniere 2024 PNAS is a NEW row, not the existing one

The table already had "Teyssonniere 2024 (species-wide trait survey)" citing PLoS Genet
2024 (Shichino, Mito, Iwasaki, Schacherer). The PNAS paper is a different work with a
different author set, and it is the 942-isolate proteome paired to the Caudal 2024
transcriptomes. Both rows now exist under distinct names.

DE-DUPLICATION OPEN: Teyssonniere 2024 PNAS (942 isolates) and Muenzner 2024 Nature (796)
share authors and the 1,011 panel. Whether they are independent acquisitions is NOT
established. Settle it before building both; the risk is ingesting one measurement twice.

### Albert 2014 excluded

Albert FW, Treusch S, Shockley AH, Bloom JS, Kruglyak L. Nature 2014;506:494-497. X-pQTL
reads protein level from a GFP fusion, one gene at a time, in a large unsequenced sorted
pool. There is no strain-by-protein matrix, so it cannot enter as a proteome row.

### Band renamed: "metabolism x expression" -> "molecular layers"

The band's members were already as much transcript-against-protein (Jakobson, Muenzner,
Albert, Skelly) as metabolism, and McManus and Martin-Perez had nowhere to sit. The band is
now: two or more of transcript, translation, protein, phosphosite, metabolite, flux and
turnover, joinable on one axis. Membership is unchanged apart from the five rows added into
it. The table distinguishes a STRAIN-level key (both layers on the same genotypes, residual
per strain) from a GENE-level key (one layer is a genome-wide coefficient such as a
half-life, so it explains why two genes differ and not why two strains do).

### Consequence to watch

The two bands now total 53 rows, so wave 1 is exactly the banded rows and NO scale row
reaches it. de Boer 2020 falls 2 -> 54 and Lee 2014 4 -> 56, both out of wave 1. That is
the deliberate cost of band-before-scale and is stated in Sec. 1.2 of the document rather
than smoothed over. Wave 1 also inverts on instance count (1.8e6 instances but 3.9e8
measurements) because its rows are vector-valued.

Albert 2018 and Muenzner 2024 were ALREADY in this band from the previous pass; no re-band
was needed. They moved 5 -> 31 and 7 -> 33 only because the perturb-seq band grew.

## 2026.09.13 - Final fifty with ten in reserve, one table, statistics last (v10)

The reviewer asked for the final top 50 in one full table with every stat and the reason
for choosing, ten extra rows in case a row proves unreachable, and the summary statistics
at the end. The document is restructured to that shape and the rest of the 168-row ranking
stays in the script and its JSON, unprinted.

- `render_final(rows)` in `build_candidate_datasets_table.py` writes `tables/final.tex`:
  rows 1 to 60 with rank, dataset (class and phenotype on a second line), band and tier,
  genotypes, environments, instances, measurements, sequence basis, time axis and the
  `why` text, with a divider at row 51. `FINAL = 60`.
- `render_summary(rows)` writes `tables/summary.tex`: counts split at the fifty line by
  band, tier, class, sequence basis, Perturb-seq axis and attributes (time axis, joins to
  built datasets, confidence, status, instance basis), with genotype, instance and
  measurement sums. It is the last section of the document.
- Sources and joins tables are restricted to the sixty; the sources table's `Why` column
  becomes `Why it is ordered here` (the band reason, or the scale rule with the
  measurement count), since the reason for choosing now sits in the final table.
- New `Candidate.time_axis` and a `TIME_AXES` curation dict (checked against the rows at
  import like `BANDS`): seven of the sixty carry a time dimension (Hackett 2020, Jariani
  2020, Jackson 2023, Wang 2022, Airoldi 2016, and the rate rows Sun 2013 and Martin-Perez
  2017). Steady-state chemostat rows vary dilution rate and are not counted.
- Review comments on v9 (`zotero_comments.py database-expansion-100`, keys `4E2MFASD`
  and `RGGC2TRF`) are answered in the new summary section: the time comment gets a
  `Time` column, the count above, and the schema position (`Environment.duration_hours`
  and `duration_generations` flatten a series into per-point environments; a first-class
  time field and a series identity are the open schema items for c(t) modeling). The Sun
  2013 comment gets a subsection and a rewritten `why`: overlap with Kemmeren 2014,
  Hughes 2000, Hu 2007 and Lenstra 2011 is on the strain axis only, since none of those
  stores a rate; the exact shared-strain count needs the unfetched GEO strain list.
- Sections dropped from the document (still generated, no longer input): the 168-row
  candidates table, counts by wave, swaps, pins, the classes and Perturb-seq sections.

Statistics of the sixty (from `tables/summary.tex`): 30 perturb-seq + 20 molecular
layers make the fifty; the extra ten are 3 molecular layers + 7 scale, and the scale seven
include de Boer 2020 and Vaishnav 2022, which carry most of the instance sum. Classes:
expression / single cell 20, natural variation 15, CRISPR library 8, metabolite 6,
modality 5, tolerance 4, regulatory DNA 2. Bases: reference-only 16, S288C-KO 13,
segregant-WGS 11, S288C+guide 7, isolate-WGS 6, designed-edit 3, reporter-locus 2, tag 1,
engineered-chassis 1. 110 named joins, 58 to a built dataset, on 42 of the sixty rows.
Confidence: 34 sourced, 26 recall. One blocked row (Lee 2014, 56).
