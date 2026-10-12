---
id: 6lifu8pzgnewh587p8bvscq
title: Teteneva2024
desc: ''
updated: 1791541893938
created: 1791541893938
---

## 2026.10.09 - Row 41 scheduled: the W3110 host gate closed, the loader refused

Teteneva, Sanches-Medeiros and Sourjik 2024, *Genome-wide screen of genetic determinants
that govern Escherichia coli growth and persistence in lake water*, The ISME Journal
18(1):wrae096, `doi:10.1093/ismejo/wrae096`, `PMC11188689`. Row 41 of the fifty
(`notes-tex/database/database-expansion-bacteria/tables/final.tex`), tracked under
umbrella issue #826. Module: `torchcell/datasets/ecoli/teteneva2024.py`.

The row has two gates. The host gate is now closed. The loader gate is refused, with the
measurement that refuses it.

### Gate 1, the host: the W3110 assembly set is deposited

W3110 is a K-12 derivative distinct from the BW25113 and MG1655 sets the genomes tier
carried, so it needs its own set, which is what issue #826 names as the closer for this
row. Deposited on 2026.10.09 by

```bash
PYTHONPATH=$WT python scripts/provision_bacterial_genomes.py \
  --refetch-dir $R --set ecoli_K12_W3110_ASM1024v1
```

| field | value |
|---|---|
| assembly set | `ecoli_K12_W3110_ASM1024v1` (`ECOLI_K12_W3110`) |
| GenBank / RefSeq | GCA_000010245.1 / GCF_000010245.2, ASM1024v1 |
| replicon | AP009048.1 / NC_007779.1, 4,646,332 bp |
| organism | Escherichia coli str. K-12 substr. W3110 (E. coli), taxid 316407 |
| submitter, date | Nara Institute of Science and Technology, 2006-01-23 |
| members | 10, every md5 matched against its NCBI directory's own `md5checksums.txt`, every sha256 re-hashed by `deposit_assembly_set` and again by `verify_assembly_set` |

Two deviations from the other four sets, both measured, neither worked around.

**The GCF `_assembly_report.txt` is a member, which no other set needs.** The GCA report
still names the 2006 RefSeq release, `GCF_000010245.1` with RefSeq-Accn `AC_000091.1`,
while the current RefSeq release is `GCF_000010245.2` with RefSeq-Accn `NC_007779.1`. The
accession pair that `ASSEMBLY_SET_ACCESSIONS` reads off the deposited report is therefore
only readable off the GCF one, and reading the GCA one would pin a release whose files
are not in the set.

**The GenBank member carries no locus tag at all**, so the tier's GenBank-first ingest
does not reach this strain. Measured off the deposited bytes by
`teteneva2024.annotation_routes()`:

| member | gene features | with `locus_tag` | `read_genbank` |
|---|---|---|---|
| `GCA_000010245.1_..._genomic.gbff.gz` | 4,444 | 0 | refuses: `a gene feature at [189:255](+) has no locus_tag` |
| `GCF_000010245.2_..._genomic.gbff.gz` | 4,531 | 4,531 | parses, 4,531 loci on NC_007779.1 |

The 2006 DDBJ/NIG annotation keys genes by symbol and carries the crosswalk as a `/note`
on the CDS features: 3,730 `ECK:JW:b` triples, 3,730 distinct JW numbers, 3,726 distinct
b-numbers (`teteneva2024.eck_jw_b_notes()`). So the GenBank deposit crosswalks W3110 to
MG1655 without ever naming a W3110 locus tag. The RefSeq route exists instead:
`Y75_RS\d{5}` locus tags, `Y75_p\d{4}` `old_locus_tag` values on 4,254 of the 4,531
gene and pseudogene GFF rows, and 2,298 `Ontology_term` rows for the GO route BW25113 and
REL606 already use.

Which route this set is read through is an **open decision and is not taken here**,
because it decides the `BacterialGeneNamespace` member and that belongs with the
identifiers Supplementary Table S4 actually reports, which is gate 2. Consequence: no
genome class, no `BacterialReferenceStrain` member, and no schema vocabulary names this
set. `scripts/schema_impact_check.py --base origin/main` reports **no schema contract
changes**, so nothing served is staled by this branch.

### Gate 2, the loader: refused, the paper is in neither mirror

Checked on 2026.10.09: `$DATA_ROOT/torchcell-library/` holds 674 citation keys and
`$DATA_ROOT/torchcell-raw/` holds 62, and not one of either matches `teteneva`. The
library mirror is populated from Zotero by `scripts/lit_sync.py`, and filing a paper in
Zotero is a curation decision that needs an explicit instruction each time. The
2026.10.08 entry of [[experiments.database.expansion-bacteria]] records that decision as
still open for this row: Teteneva 2024 entered the fifty and nothing was added to Zotero.

Nothing was fetched. No paper, no PDF, no workbook, and no mirror entry was created.

So every schema value a loader needs is a typed absence
(`teteneva2024.LOADER_GAPS`, all six `deferred_pending_source_review`, the one
recoverable reason) rather than a guess:

| field | why it cannot be sourced today |
|---|---|
| `n_samples` | the replicate count behind one fitness value |
| `phenotype_statistic` | which statistic Table S4 holds and on what scale, which decides `EnvironmentResponsePhenotype` against `FitnessPhenotype` (the latter clamps non-positive values, so a signed log2 ratio cannot be stored as fitness) |
| `time_zero` | what the ratio is taken against |
| `media` | an oligotrophic natural lake water, filtered and non-filtered, is a NEW `Media` entry that must be defined from the paper's own description of the water |
| `gene_namespace` | which identifiers Table S4 names its genes by, which also decides gate 1a |
| `genotype` | the library construction and background, and whether a stored genotype is the insertion mutant or the gene |

`SourcedValue` cannot even be constructed for this row: it requires a `citation_key` and
the sha256 of a mirrored artifact, and there is neither. That is the cleanest statement of
the block.

The counts in the schedule row, 66,162 non-empty fitness values over 11,027 gene by
sample rows for 3,691 genes, are the measurement of
`experiments/database/scripts/build_bacteria_candidate_datasets_table.py`, read with
`openpyxl` on an earlier pass. They are recorded in the module as that script's numbers
and are **not** restated as this branch's measurement, because this branch never opened
the workbook.

### L0 to L4: not run, and cannot be

There is no dataset class, so there is no build and no store. L0 (store opens), L1
(record schema), L2 (counts), L3 (provenance audit) and L4 (cross-source) all have no
subject here. Recording a table of levels would be recording a build that did not happen.
What this branch verified instead is the tier deposit: `verify_assembly_set` re-hashed all
ten members, and the sixteen tests of
`tests/torchcell/datasets/ecoli/test_teteneva2024.py` pass (11 hermetic, 5 data-gated with
`DATA_ROOT` exported). The three measurement functions are covered hermetically as well,
on synthetic one-locus flat files written with Biopython and served through a stubbed
`resolve`, so `annotation_routes` is exercised in CI where the tier is absent: the module
reports 100% statement coverage without the data-gated tests.

### What unblocks it, in order

`teteneva2024.EDITS_NEEDED` holds this list in code. Everything after the first item waits
on the first.

1. The owner files the paper in Zotero (group library `database/Escherichia-coli`, matched
   by DOI), then `scripts/lit_sync.py` and `scripts/lit_capture_si.py` mirror it. Cheap:
   every object is already in the PMC open-access bucket under `PMC11188689.1`, listed on
   2026.10.09 and recorded in `teteneva2024.PMC_OA_DEPOSIT`: the paper as PDF, XML and
   text, four supplementary figures, Tables S1 and S2 as PDF, and **Tables S3 and S4 as
   xlsx**, the second of which is the workbook the row was counted off. The existing
   `pmc_cloud` retriever reaches all of it with no by-hand step.
2. Deposit Table S4 in the raw mirror with its retrieval record and sha256, and re-count
   its rows and non-empty values off the pinned bytes.
3. Decide the annotation route for the set against the identifiers Table S4 reports.
4. A W3110 genome class, then the schema vocabularies, then `bacteria_common`, then the
   verification gene universe, then the loader with its adapter, conf, map entry and the
   four adapter pin places.

Note that step 5 onward is where the `BacterialReferenceStrain` Literal grows, and that
is the schema-impact event for this row: adding a member to that Literal changes the
schema closure of every model that references it, so it stales the served bacterial
datasets and lands in a full rebuild. This branch deliberately does not do it, so its
verdict is clean.

### Files

- `torchcell/datasets/ecoli/teteneva2024.py`
- `tests/torchcell/datasets/ecoli/test_teteneva2024.py`
- `torchcell/sequence/genome/registry.py` (`ECOLI_K12_W3110`)
- `scripts/provision_bacterial_genomes.py` (the set and its ten digests)

## 2026.10.12 - The SI is deposited and Table S4 measured: the genes are named by symbol only (#910)

The owner filed the paper in Zotero (group item `PUKZFNFZ`, #910) and downloaded five
supplementary files from the ISME Journal page in a browser; they are deposited under
`$DATA_ROOT/torchcell-raw/tetenevaGenomewideScreenGenetic2024/si/` (`SI_DEPOSIT`,
`manual_browser`, arrived 2026-10-12 03:40 CDT, `DEPOSIT.md` and `SHA256SUMS.txt`
beside them) and the key's `manifest.json` is CREATED. All five are byte-identical to
the PMC Article Datasets objects of the same name under `PMC11188689.1/`, fetched and
compared on 2026-10-12, which each record names as its scripted equivalent:

| file | bytes | sha256 | holds |
|---|---|---|---|
| `supplementary_figure_s1_wrae096.pdf` | 354,566 | `65bcac7a...` | hierarchical clustering of gene fitness |
| `supplementary_figure_s2_wrae096.pdf` | 251,076 | `d88f962b...` | PCA of gene fitness per water sample |
| `supplementary_table_s2_wrae096.pdf` | 36,151 | `c41e8b17...` | date, pH, air and water temperature of five lake water samples |
| `supplementary_table_s3_wrae096.xlsx` | 14,056 | `5cec5dd2...` | sequencing QC of 63 samples |
| `supplementary_table_s4_wrae096.xlsx` | 1,809,613 | `678bbe6c...` | gene fitness, principal components, Euclidean distance |

Not downloaded: Supplementary Table S1 and Figures S3 and S4 (in the bucket; nothing
reads them). The paper text is in neither mirror yet; `lit_sync` mirrors `PUKZFNFZ` on
its next run, and the tripwire test now waits for exactly that.

**Table S4, measured** (`table_s4_inventory`, re-derived by the `--data` test): 11,027
gene-by-water-sample rows (water sample 1: 3,675, 2: 3,679, 3: 3,673), six fitness
columns (filtered and non-filtered water at days 2, 4 and 8), **66,162 non-empty
fitness values** (the schedule's count, now off the pinned bytes), and **3,691 genes
named only by gene symbol** (column `Gene name`; no b-number, JW number or locus tag).
The legend calls them "Gene fitness values" and names no statistic, so the scale stays
the `phenotype_statistic` gap until the paper text is read.

**The identifier finding for the owner's route decision (gate 1a).** Whichever member is
read, a symbol route is needed. By `/gene` alone (`symbol_routes`):

| annotation member | land on one gene feature | on two or more | on none |
|---|---|---|---|
| GenBank `GCA_000010245.1` (no locus tags) | 3,036 | 0 | 655 |
| RefSeq `GCF_000010245.2` (`Y75_RS` tags) | 3,607 | 44 | 40 |

So a RefSeq-primary route resolves 3,607 of 3,691 symbols uniquely before any synonym
rule, and the GenBank route 3,036. The choice, and the synonym rule that closes the
remainder, is the owner's (#910); no loader is written.

Table S3: 63 samples, per water sample 3 controls (day 0), 9 filtered and 9
non-filtered (days 2, 4, 8 x 3 technical replicates); the QC-failed sample the legend
highlights in red is `sample_10`. Table S2 lists five water samples (#1 to #5); Table S4
carries #1 to #3. Measurements:
[[experiments.036-dataset-fixes-before-kg-build.scripts.manual_deposits_2026_10_12]].

## 2026.10.12 - The loader: all 66,162 Table S4 values, each under its released symbol (#910)

`RbTnseqTeteneva2024Dataset` (`torchcell/datasets/ecoli/teteneva2024.py`, PR #1068)
stores every value Supplementary Table S4 releases: 11,027 rows (water sample 1: 3,675,
2: 3,679, 3: 3,673) x 6 conditions = **66,162 records**, one
`BacterialEnvironmentResponseExperiment` per (row, filtration, sampling day). No row
is refused and no annotation decides which records exist.

### The gene is stored as released

Owner, 2026-10-12, relayed: "the data is the data!" Each record's genotype is one
`BacterialReleasedStrainPerturbation` (the released-identifier leaf of #691, landed with
Nichols 2011): the 'Gene name' verbatim as `systematic_gene_name`, `perturbed_gene_name`
and `strain_label` (a pooled release labels its row by the gene alone),
`identifier_kind` 'Gene name' (the column header), `released_kind`
`transposon_insertion`, `cassette` kanamycin, and the W3110 host namespace
`ecoli_k12_w3110_refseq_locus_tag`. The leaf refuses a locus tag, and none of the 3,691
symbols is one.

`ReleasedStrainKind` had no member for a transposon insertion (deletion, affinity tag,
degron, linked marker, sequence allele, truncation), so it gains `transposon_insertion`
with the SO pair the placed `TransposonInsertionPerturbation` pins (SO:0001218
`transgenic_insertion`); the paper states the class of every row:

> For the sake of simplicity, only insertions in the coding gene regions, which were assumed to inactivate the respective gene function, were considered for the assessment of the gene fitness effects.

How the symbols land on the W3110 annotations is a DOWNSTREAM artifact, measured by
[[experiments.database.scripts.teteneva2024_symbol_annotation]]: 3,607 of 3,691 land on
exactly one RefSeq gene feature by `/gene` and 3,036 on exactly one GenBank one, and a
three-step mapping onto RefSeq `Y75_RS` loci places 3,607 + 65 + 9 = 3,681, leaving 10
(`dcuC`, `dmpG`, `gatA`, `mddA`, `nanS`, `pat`, `rzoD`, `tal`, `tuf`, `wbbL`). None of
that is loader logic or a record field. An earlier draft of this branch resolved the
symbols in the loader and kept the 10 out of the store; the owner's correction removed
that route, and a draft `identifier_kind` field on `TransposonInsertionPerturbation`
(a field-meaning change) was dropped in favour of the #691 leaf.

### Every value sourced, the paper text in the raw mirror

The library mirror held no Teteneva key on 2026-10-12 (Zotero item `PUKZFNFZ` was filed
that day and `lit_sync` had not run), so the paper text is the PMC Article Datasets
object, recorded in the raw mirror by `record_paper_and_texts` with its two derived
texts:

| file (`torchcell-raw/tetenevaGenomewideScreenGenetic2024/`) | sha256 | how |
|---|---|---|
| `paper/PMC11188689.1.txt` | `422bd792...` | `pmc_cloud`, key `PMC11188689.1/PMC11188689.1.txt` |
| `si/supplementary_table_s2_wrae096.layout.txt` | `78f0ba62...` | `pdftotext -layout -enc UTF-8` (21.01.0) of the deposited Table S2 PDF |
| `si/supplementary_tables_s3_s4.rendered.txt` | `b2f8ad50...` | both LEGEND sheets and Table S3's red-filled row, openpyxl |

All 33 module quotes and the five Table S2 rows audit against those bytes
(`test_every_quote_is_in_its_pinned_file_and_the_texts_rerender`, `--data`).

| field | value | source |
|---|---|---|
| value | gene fitness, log2 ratio against the day-0 culture, normalized, chromosome running median (window 251) and mode subtracted, centred on 0 | Materials and methods, 'RB-TnSeq data analysis' |
| `n_samples`, `sample_unit` | technical replicates behind the mean: 3, except 2 for lake water #1 filtered day 4 | "Gene fitness values were then averaged across the technical replicates."; Table S3 legend: "One sample that failed the QC is omitted from the further analysis and is highlighted in red." The red row is `sample_10` (replicate 1, day 4, filtered water, technical replicate 1) |
| uncertainty | typed gap (`not_reported_by_primary`) | Table S4 releases the mean only |
| medium | each water sample #1 to #3, filter-sterilized (0.22 um Steritop) or not, a `Media` deriving from the new library base `LAKE_WATER` (composition `intrinsically_undefined`) with its Table S2 row (date, pH, air and water temperature) | 'Water sampling'; Table S2; Fig. 2 legend for the #1-#3 numbering |
| `duration_hours` | 48, 96, 192 (24 x the stated day) | "Samples for further analysis were taken on Days 2, 4, and 8 of cultivation, ..." |
| temperature | typed gap (`not_reported_by_primary`) | "incubated without antibiotics at room temperature"; Table S2's temperatures are the lake's at sampling |
| aerobicity | field default `aerobic`, not sourced (static 24-well plates; the field cannot be None, as in goodall2018 and price2018) | none stated |
| reference | `W3110 RpoS+` on `ecoli_K12_W3110_ASM1024v1`, RefSeq accession GCF_000010245.2; the typical gene of the same environment, response 0 | "aligned to the E. coli K-12 W3110 reference genome (NCBI reference sequence: NC_007779.1)"; "This normalization removes local biases and ensures that the gene fitness distribution is centred around zero." |
| background | `W3110 RpoS+`, genotype statement 'RpoS+', no typed allele | "a derivative of the laboratory strain with the restored RpoS activity" (the allele is ref. 14's) |

### Schema and genome tier

- W3110 vocabulary: `BacterialReferenceStrain`, `BacterialAssemblySet`,
  `BACTERIAL_ASSEMBLY_SETS`, `ASSEMBLY_SET_ACCESSIONS` (GCA_000010245.1 /
  GCF_000010245.2, from the GCF report), `BacterialGeneNamespace` +
  `BACTERIAL_LOCUS_TAG_PATTERNS` (`^Y75_RS\d{5}$`). Record
  `database/schema_changes/2026-10-12-w3110-reference-strain-vocabulary.json`.
- `ReleasedStrainKind.transposon_insertion` and its SO pair. Record
  `database/schema_changes/2026-10-12-released-strain-kind-transposon-insertion.json`.
- Impact (`scripts/schema_impact_check.py --base origin/main` at 464c5cfc3): 60 loader
  groups stale, 0 breaking; both records' rebuild scope `dev_stores` (no served
  closure: the newest release snapshot serves no bacterial dataset).
- `EcoliK12W3110Genome` reads the set RefSeq first
  ([[torchcell.sequence.genome.ecoli.w3110]]); G2 of the candidate gate now reads
  W3110. The loader itself reads no genome.

### Candidate verdict

`database/candidates/tetenevaGenomewideScreenGenetic2024.json`, written by
[[experiments.database.scripts.teteneva2024_candidate_verdict]]: admissible, G1 to G5
pass (G3: 7 files, 11,100 worksheet rows intact).
