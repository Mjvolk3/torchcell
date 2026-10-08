---
id: qafwzbq1qfft4v4f32xgpi4
title: Cai2023
desc: ''
updated: 1791494477199
created: 1791494477199
---

## 2026.10.08 - MCF2Chem 2023 settled: an aggregation NOT admitted (row 46)

Source: `torchcell/datasets/ecoli/cai2023.py` (no dataset class; a provenance record).
Tests: `tests/torchcell/datasets/ecoli/test_cai2023.py`.
Measurement: `experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory.py`,
results beside it as `mcf2chem2023_release_inventory.json`,
`mcf2chem2023_accession_probe.json`, `mcf2chem2023_table_s1_reviews.csv` and
`mcf2chem2023_titer_surface.csv`.
Schedule row: `experiments/database/scripts/build_bacteria_candidate_datasets_table.py`
under `name="MCF2Chem 2023"`, E. coli, tier 3, `status="aggregation"`.

Cai et al. 2023, "MCF2Chem: A manually curated knowledge base of biosynthetic compound
production", Biotechnol. Biofuels Bioprod. 16:170, doi:10.1186/s13068-023-02419-8,
citation key `caiMCF2ChemManuallyCurated2023`, `paper.md` sha256 `3c3557e6...`.

**DECISION: NOT LOADED.** No dataset class, no adapter, no conf, no graph class, no
schema change, no raw mirror. The deliverable is the measurement plus this record.

### What the release actually contains

| artifact | sha256 | bytes | tables | holds |
|---|---|---|---|---|
| `paper.md` | `3c3557e6...` | 46,476 | Tables 1-3 inline | the statistics, Table 1's per-category record counts |
| `si/si1.docx` | `88cb9cc3...` | 40,017 | 1, 269 x 2 | Table S1: 268 (Review_title, Review_doi) rows |
| `si/si2.docx` | `e6985953...` | 20,723,784 | 1, 5 x 4 | Figs. S1-S9 captions + Table S2 (coverage of Metabolic Engineering) |
| `si/si3.docx` | `2c790469...` | 20,343 | 1, 7 x 2 | the two recommendation scoring equations + six (Route, Score) rows |

**No per-record artifact is released.** Measured by `release_inventory()`: three tables
across the three supplementary files, and not one header cell naming a titer, yield,
productivity, content, strain, compound or species. The rule is deliberately generous
and still returns zero; the hermetic test
`test_a_file_that_did_carry_records_would_be_found` writes a counterfactual file that
flips it, so the zero is a measurement rather than a rule that never fires. PMC OA
serves exactly these three objects: `MOESM1`-`MOESM3` answer 200 at 40,017 / 20,723,784
/ 20,343 bytes, `MOESM4` and `MOESM5` answer 404, and no `.xlsx`, `.zip`, `.csv` or
`.pdf` variant of any `MOESM` exists.

The paper's complete data-availability statement:

```
All data are available at https://mcf.lifesynther.com.
```

No repository, no Zenodo or figshare deposit, no DOI'd data file. A web search for a
bulk export or a deposit under either name found none.

**The one named channel served no data on 2026-10-08** (`mcf2chem2023_accession_probe.json`,
probed 21:12 UTC; reproduce with the script's `probe`):

| path | status | bytes | |
|---|---|---|---|
| `/` | 200 | 3,821 | a static page titled "Service Relocation in Progress": "The website will be temporarily unavailable during the move." |
| `/openapi.json` | 502 | 166 | the FastAPI backend the paper describes ("FastAPI 0.73.0 ... MongoDB 5.0.4") is not answering |
| `/download`, `/api/download`, `/data`, `/api/data`, `/browse`, `/api/browse`, `/docs`, `/redoc` | 404 | 162 | no export, no data route, no API documentation |

Eight further paths probed by hand the same day (`/downloads`, `/statistics`,
`/api/statistics`, `/help`, `/about`, `/api/v1/openapi.json`, `/api/v1/docs`,
`/api/record`, `/api/production`) also answered 404. The DNS name resolves
(39.98.45.1), so this is a live host serving a holding page, not a dead domain.
The Internet Archive could not be queried on 2026-10-08: its availability API answered
429 and its CDX API served an "Internet Archive services are temporarily offline" page.
That check is therefore NOT MEASURED, and it would not change the decision: a Wayback
capture is not a released artifact, and the record table is rendered client-side.

This is exactly the state the schedule's own status vocabulary names, verbatim from the
`Status` comment in that same file:

```
a corpus that re-serves other papers is an
"aggregation" and its records are not net new until it is split by source, and a
row whose per-record values are not released is "blocked".
```

The row is both. There is no artifact to hash, so no
retrieval command could ever be recorded, re-run and verified against a sha256, which is
the minimum this project requires of any value.

### Does it carry per-row provenance to the originating paper?

It carries a per-record CITATION. It does not carry a per-record MEASUREMENT. Both halves
are the paper's own words, verbatim from the pinned `paper.md`.

Methods, Data collection and processing, two consecutive sentences:

```
Te raw data of MCF2Chem were extracted from reviews of microbial biosynthesis over the last 5 years (from August 1, 2017, to July 31, 2022).
Based on the reference columns in review tables, direct references to each record were obtained and supplemented programmatically or manually.
```

Results, Database overview, three spans of one paragraph (the first two are one
sentence, split where the OCR carries a non-breaking space into "Table 1"):

```
In total, 8888 items of production records were extracted from 268 review articles, involving information from 4765 original microbial metabolic engineering articles
(92 records were those of patents
Te 4765 articles concerned spanned the period from 1946 to 2022
```

So the chain is three hops: MCF2Chem row -> a review's table cell -> the primary article
the review's reference column names, over 4,765 articles spanning 1946 to 2022, of which
92 records have a patent rather than an article behind them. The value stored in
MCF2Chem is a review author's transcription of a primary paper's number, re-transcribed
by the curators; the citation sits beside that number without being evidence for it.

The Discussion states the substitution plainly, so it is not an inference:

```
manually extracting information directly from original literature is both time-consuming and labor-intensive
```

and names its cost two paragraphs later:

```
owing to their lagging nature, omission of the latest data is inevitable
```

The release's one measured statement about its own completeness is Table S2, the coverage
of a single primary journal, *Metabolic Engineering*: 66% (2016), 65% (2017), 60% (2018),
63% average, over 40/61, 42/65 and 36/60 articles. `table_s2_coverage()` asserts those
rates when it reads them, because a docx cannot be quote-audited against its bytes after
the fact (the lim2025 precedent).

### The convention the two landed aggregation loaders set, and whether it was followed

Six rows of the fifty carry `status="aggregation"`; two are loaded.

| row | what it aggregates | what the stored value IS | per-record attribution |
|---|---|---|---|
| Lim 2022 `PutidaPrecise321Lim2022Dataset` | 321 RNA-seq samples from 118 conditions over 21 projects; 225 of the 321 reprocessed from 21 prior projects, 96 new | log2(TPM + 1) that **Lim 2022 recomputed** from read counts, unit back-solved (`2**X - 1` sums to 1e6 in all 321 columns) | the sheet's own `DOI`, else `PMID`, else Lim 2022; `SourceStudy` per sample in the ledger |
| Borchert 2024 `RbTnseqBorchert2024Dataset` | 332 RB-TnSeq samples into one 4,732-gene matrix, the superset of three other rows | gene fitness **Borchert 2024 computed** from barcode counts, with the t statistic | a sample is attributed to a prior study only when that study's MIRRORED Methods or Results name its condition |
| MCF2Chem 2023 | 8,888 production records from 268 reviews over 4,765 articles | a review's table cell, re-typed | a citation from the review's reference column; no source paper mirrored |

The convention those two set has two clauses, and MCF2Chem fails both:

1. **An admissible aggregation RE-MEASURES.** Both landed rows aggregate RAW primary data
   that the aggregating paper reprocessed itself, so the stored number is the
   aggregator's own measurement and the aggregator is the right citation for it.
   MCF2Chem re-types a number someone else already summarized.
2. **Attribution requires the source paper to be MIRRORED.** Borchert 2024's rule is
   explicit, and it is enforced by omission: "none of these papers is mirrored, so none
   becomes a record's publication". Honoring MCF2Chem the same way would mean mirroring
   268 reviews plus up to 4,765 primary articles to put one verbatim quote behind one
   record. That is not a scale objection, it is the reason the chain cannot be closed:
   the quote that would justify a titer lives in a paper we would have to read anyway,
   at which point the primary paper is the loadable row and MCF2Chem is redundant.

Followed, by refusing the row. The third clause both landed rows also satisfy, a measured
superset or net-new statement against what we serve, is below.

### Measured overlap against the datasets we already serve

The duplication surface is the phenotype MCF2Chem carries, a product titer. Of the **51
served bacterial dataset classes**, exactly **4** have `experiment_class ==
ProductTiterExperiment` (`mcf2chem2023_titer_surface.csv`); the other 47 carry proteome,
RNA-seq, metabolome, fitness, environment-response, essentiality, interaction or
protein-turnover phenotypes, none of which MCF2Chem has a column for.

| served titer class | citation key | DOI | year | inside MCF2Chem's window? |
|---|---|---|---|---|
| `IsopentenolTiterFoo2014Dataset` | `fooImprovingMicrobialBiogasoline2014` | 10.1128/mbio.01932-14 | 2014 | **yes** |
| `IsoprenolTiterCarruthers2025Dataset` | `carruthersAutomationMachineLearning2025` | 10.1038/s41467-025-66304-8 | 2025 | no |
| `IsoprenolTiterDeSiqueira2025Dataset` | `desiqueiraAlternateRoutesAcetate2025` | 10.1128/aem.02123-24 | 2025 | no |
| `IsoprenylAcetateTiterKang2026Dataset` | `kangMultilayeredMetabolicRemodeling2026` | 10.1016/j.mec.2026.e00274 | 2026 | no |

MCF2Chem read reviews published 2017-08-01 to 2022-07-31, so three of the four name
papers no review in that corpus could have summarized. **One served dataset, Foo 2014,
is a genuine duplication candidate**, and at Table S2's own 63% single-journal coverage
rate it is a candidate rather than a hit.

**Source-corpus overlap: 0 of 268.** None of Table S1's 268 review DOIs is the DOI of any
of the 51 served bacterial datasets (34 distinct mirrored DOIs, since several classes
share a paper), and none is any of the 95 DOIs the bacteria candidate table names across
all fifty rows. MCF2Chem's corpus is reviews; ours is primary papers. The two sets do not touch at the paper level, which is
also why a DOI join could not have resolved MCF2Chem's records to our rows.

**The per-record overlap is NOT MEASURED and cannot be.** The 8,888 records are not
released, so there is no table to intersect with the four titer stores. That is a
"could not measure", not a "measured and null", and it is stated that way deliberately.
What the record counts do bound: bacteria are 5,276 of the 8,888 records over 356 species
(Table 1), and no per-species row is published, so the E. coli subset size remains
undetermined, exactly as the schedule row says.

### Four required schema fields the release has no value for

Even granted the records, the titer family could not be populated. Each blocker is read
off the live schema by `test_the_blocked_titer_fields_are_required_on_the_live_schema`
and paired with the release's own statement of the limit.

| required field | what the schema needs | the release |
|---|---|---|
| `ProductTiterExperimentReference.phenotype_reference` | a required `ProductTiterPhenotype`, so every record needs a released parent-strain titer read in the same `CultureEnvironment` | "Production data of compounds were divided into four columns: titer, yield, productivity, and content." Four product columns, no control column. Three sibling loaders already refuse the titer family for want of a released reference titer. |
| `ProductTiterPhenotype.product` | a typed `Compound`, the join key onto the shared compound layer | "approximately $3 2 \%$ of the compounds in MCF2Chem cannot be retrieved from PubChem" |
| `ProductTiterPhenotype.titer` | one required float, validated finite and non-negative | "titer range data were divided into maximum and minimum titers" |
| `ProductTiterPhenotype.titer_unit` | a typed UO-aligned `ConcentrationUnit` | "original units were retained for those that could not be converted"; the Discussion gives the cause, "the production units used were diverse, and some units were difcult to unify" |
| `ProductTiterExperiment.genotype` | a `Genotype` of typed bacterial perturbations, each an edit to the genomic content of the cell | "All other parts included possible strain modifcation methods, strain genotypes, and other information." The hedge "possible" is the release's own. |

Quotes are verbatim against the OCR, which drops the ft/fi ligatures ("Te", "fle",
"difcult", "modifcation", "classifed"). All 15 `SourcedValue` quotes audit PASS against
`paper.md` sha256 `3c3557e6...` (`test_every_quote_audits_against_the_pinned_paper`).

### The argument, in one paragraph

An aggregation belongs in a provenance-first experimental-data ontology when it
re-measures primary data and releases the result as an artifact we can hash, which is
what Lim 2022 and Borchert 2024 do and why they are loaded. MCF2Chem does neither: its
values are transcriptions of review tables rather than measurements, and they are
published only through a web view that served a relocation notice and a 502 the day the
row was settled, so there is nothing to pin and no retrieval command that could be
re-run. Either failure alone is disqualifying; together they make the row a record rather
than a loader. The honest route to these numbers is the primary papers themselves, where
a titer comes with its control, its units, its replicate design and a quote we can bind
to a sha256.

### Follow-ups

- **A curation decision for the owner, not taken here.** If the Foo 2014 overlap or any
  E. coli titer in MCF2Chem is wanted, the route is the primary paper, which means adding
  that paper to the library. Naming it and stopping: no paper was added, nothing was
  written to Zotero, and `zotero_add_ref.py` was not run.
- **Oyetunde 2019, another of the six aggregation rows, is the closer analog and is
  still open.** Its `accession_confirmed=True` and its S1 File is an Excel of the paper
  list plus the extracted per-factory data (about 1,200 E. coli cell factories), so
  unlike MCF2Chem it HAS a hashable per-record artifact. It would be decided on clause 1 of the convention alone (does it re-measure,
  or re-type ~100 papers?), which is a different argument from this one. MCF2Chem cites
  it as prior art, verbatim: "manually extracted data from $\sim 1 0 0$ articles and
  curated a dataset comprising $\sim 1 2 0 0$ experimentally implemented cell
  factories".
- **The row's `accession_confirmed=False` is correct and should stay.** The accession
  resolves and answers 200; it serves no data. The flag separates the six aggregation
  rows cleanly: the four with `accession_confirmed=True` (D2Cell 2026, Oyetunde 2019,
  Borchert 2024, Lim 2022) name a file, and the two with `False` (MCF2Chem and the
  CeCaFDB flux compendium at `cecafdb.org`) name a web service. Both loaded rows are in
  the first group, which is the pattern this measurement put a reason behind. The schedule table is the triage snapshot
  and is not edited when a row is settled (Wetmore 2015, subsumed, is still
  `status="candidate"` there), so no row was changed.
- **`docx_text` / `docx_tables` now exist in three modules** (`lim2025`, `yunus2026` and
  here). A shared home in `bacteria_common` is the obvious cleanup; it was not made here
  because that module is shared with the sibling branches in flight.

### The sweeper removed this worktree mid-task

`.wt-in-progress`, the marker the fan-out convention recommends, is listed in
`.git/info/exclude`, so `git status --porcelain` never shows it and a worktree holding
only that marker reads as clean. `/wt-cleanup` classifies a clean worktree whose commits
are all on `origin/main` as `landed` and removes it, which it did. A marker has to be
untracked AND not ignored to protect a tree; this branch used
`WT_IN_PROGRESS_mcf2chem.txt` instead, which `git check-ignore` rejects.
