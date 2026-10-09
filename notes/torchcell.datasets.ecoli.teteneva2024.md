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
