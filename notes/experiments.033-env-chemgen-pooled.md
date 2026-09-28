---
id: 6u8milf0p31h54sdff5gc73
title: 033 Env Chemgen Pooled
desc: ''
updated: 1790549452016
created: 1790549452016
---

## 2026.09.27 - The pooled chemogenomic build, and why it is a new experiment

The build the experiment 031 planning document asks for
(`notes-tex/031-unified-representation`, [[experiments.031-env-chemgen-inhibitor-tolerance]]):
one store holding every environment-response record of the four kept chemogenomic datasets,
so that the modeling phase trains one model over them. 031 was the data exploration and the
representation design, read from the dev-tree loaders; 033 is the served-graph build those
scripts never made, in the same three-part shape as 029 and 030: a query, a build script and
a launcher.

**Datasets, by decision.** Vanacloig 2022 (the application target), Hillenmeyer 2008 HET,
Hoepfner 2014 with both ploidy arms, Wildenhain 2015. Hillenmeyer 2008 HOM is left out by the
decision recorded in the 031 document ("drop hillenmeyer hom"). Served counts from
`kg_manifest.json` at release 2026.09.21-ab6d8c5d: 143,218 + 2,698,797 + 3,124,319 + 428,206 =
6,394,540 records.

**Three build decisions, each measured or traced.**

1. The query decides the record population only: a dataset and every perturbed gene in the
   S288C gene set ([[experiments.033-env-chemgen-pooled.queries.001_env_chemgen_pooled]]).
   No medium, dose or compound filter, so the Hillenmeyer records with no dosed compound
   (media swaps, temperature, radiation; 147,953 of HET's records carry a blank dose basis in
   the 031 axis table) stay. Which records an arm reads is a read-time selection, as in 030.
2. Aggregation by the (genotype, environment) cell, with a new aggregator
   ([[torchcell.data.genotype_environment_aggregate]]). `GenotypeAggregator` keys on the gene
   set alone, which would put a gene's 41 to 5,170 conditions into one entry and leave the
   label table with the first of them. The raw and conversion stores hold one record per
   entry, and the processed reader iterates a JSON list, so an aggregator is not optional: a
   build with `aggregator=None` copies single dicts into `processed/` and `get()` fails on
   them.
3. No conversion and no deduplication. Nothing here is essentiality or synthetic lethality,
   so the converter would copy about 90 GB byte for byte; and which measurement of a cell a
   trainer reads is a label policy at read time, not a build decision.

**What the cell key groups, measured on the dev stores.** The environment identity is the
content address the graph adapter uses (`environment_identity`: medium composition,
temperature, added compounds at their doses, aerobicity, duration), so the only records that
share a key are repeated screens of one compound at one concentration, Hoepfner's fourteen
compounds run in more than one study. No environment is shared across datasets, because the
media, durations and dose units differ. Parsing a record's environment from its stored JSON
costs 0.12 to 0.36 ms (500 records per dataset, scratch timing on 2026-09-27), so aggregation
pass 1 over 6.4M records is under an hour.

**Sizes, measured on the dev stores.** One record with its reference serializes to 15 KB
(Vanacloig), 12 KB (HET), 13 KB (Hoepfner) and 38 KB (Wildenhain), so the raw store is about
90 GB and the three stage copies about 280 GB. `/db` had 759 GB free on 2026-09-27.

**Not the 030 raw stage.** The batched, multi-worker `Neo4jQueryRaw` and the raw completion
marker live on the unlanded 030 branch (PR #414). This build runs main's single-threaded raw
stage, about 630 records per second on the 025 measurement, so the query is about three hours
and the whole build one slurm job ([[experiments.033-env-chemgen-pooled.scripts.query]]).

**What the build does not do.** The tensor layer still reads only gene names: no copy number,
no ploidy and no environment field reaches the model (section 11 of the 031 document). The
processed entries carry all of it in their JSON, so the processor changes in the 031
implementation order are read-side work on this store, not another build.

## 2026.09.28 - Build 001 complete

Slurm 2929, COMPLETED in 7 h 48 m, finished 06:00 CDT, 16 CPUs and 64 GB from the detached
worktree `033-build` at commit 1200cedb2. Root
`/db/experiments/033-env-chemgen-pooled-001-pooled-build`. Summary in
`results/dataset_index_summary.json`.

**6,394,540 experiments became 6,042,771 cells, and nothing was lost.** The returned count
equals the served count for all four datasets, so the gene filter dropped no record: every
gene these screens measure is in the S288C gene set.

| dataset | served | returned | cells | measurements folded |
|---|---|---|---|---|
| Vanacloig 2022 | 143,218 | 143,218 | 143,218 | 0 |
| Hillenmeyer HET | 2,698,797 | 2,698,797 | 2,591,199 | 107,598 |
| Hoepfner 2014 | 3,124,319 | 3,124,319 | 2,880,165 | 244,154 |
| Wildenhain 2015 | 428,206 | 428,206 | 428,189 | 17 |

**No cell mixes two datasets.** The per-dataset entry counts sum to 6,042,771, exactly the
store length, so no environment is shared across sources. That was predicted from the media,
duration and dose-unit differences and is now measured.

**Measurements per cell**, the multi-measurement structure a read-time label policy chooses
among: 5,764,190 cells hold one, 237,264 hold two, 30,380 three, 604 four, 36 five, 10,246
six, one holds ten and 50 hold twelve. The folded excess totals 351,769 and reconciles with
the table above. Hoepfner supplies 69 percent of it, which is the expected direction: it is
the dataset whose campaign repeats reference compounds across studies. Wildenhain supplies 17
because the paper already averages its replicate screens into one z score per cell, and
Vanacloig none because it reports one measurement per gene-by-compound cell.

**Cost.** Raw query 2 h 15 m at about 790 records per second, beating the 630 projected from
the 025 build. Stage sizes are raw 108 GB, aggregation 106 GB and processed 106 GB, 320 GB in
total against the 350 GB the preflight demanded. The index and label-table stages took the
balance of the run, in the middle of the 2-to-4-hour projection.

**Disk.** `/db` is at 95 percent with 441 GB free. The raw and aggregation stages hold 214 GB
and are regenerable from the query; deleting them is the user's action, not this note's.
