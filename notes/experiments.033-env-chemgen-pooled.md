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
