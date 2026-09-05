---
id: fkqlxry80cgcypnd26xp5si
title: 025 Solid Growth
desc: ''
updated: 1788495958266
created: 1788495958266
---

Experiment 025 is the direct progression of `010-kuzmin-tmi`: the same 376,732 Kuzmin
triple genotypes with 010's exact train/val/test split pinned, embedded in every
solid-medium growth record the uncapped knowledge graph holds (all Costanzo 2016
singles/doubles, Kuzmin 2018/2020, SGD essentiality, SynthLethalDB). The experiment
number carries the lineage: 010 = triples only, 025 = 010 plus all fitness and all
interactions around it. Hypothesis (untested): tmi inference improves when the predictor
also learns dmi and the fitness values both interaction scores derive from.

## 2026.09.03 - Build Record: Uncapped KG, Storage Tiers, Dataset, Split Transfer

### 1. The uncapped knowledge graph (slurm job 1558)

Rebuilt the database with both Costanzo 2016 double-mutant sets at their full
20,705,612 records each -- previously capped at 100k (job 970) or prefiltered to ~90
records (job 990, still on radiant). Six attempts; the five failures each exposed and
fixed a real defect in the adapter pipeline, the decisive one being fork +
copy-on-write + the cyclic GC (`gc.freeze()` before forking, PR #266). Verified end to
end (LMDB entries = CSV rows = imported nodes = live cypher count):

- 83,048,247 nodes / 310,922,919 relationships -- 11.6x / 14.8x the previous build
- ~20 h wall, bulk import in 19 min
- Config `torchcell/knowledge_graphs/conf/kg_uncapped.yaml`, slurm script
  `database/slurm/scripts/gilahyper_uncapped_build-slurm_docker.slurm`

### 2. Storage tiers (three new drives in service)

- `/scratch` (Micron 7 T, power-loss protection) -- ML-only tier, freed from 96% to 75%
- `/db` (SN850X 7.3 T, no PLP) -- neo4j store + serving container + the 025 build;
  rule: nothing irreplaceable lives here
- `/bulk` (RAID1 over 2x WD Gold, 23.6 T) -- cold archives: every build's CSVs (the
  re-import insurance), dumps, the deprecation graveyard, a copy of the literature
  mirror
- `scripts/setup_storage_tiers.sh` + `scripts/migrate_storage_tiers.sh`; all copies
  rsync-verified before sources were purged. RAID6 across four drives when the next
  two Golds arrive.

### 3. Serving, locally

Radiant's 500 G quota cannot hold the 525 G store, so GilaHyper serves it:
`tc-neo4j-readonly`, db `torchcell`, `bolt://gilahyper.zapto.org:7687` (browser
:7474). Client code no longer hardcodes hosts: `NEO4J_URI`/`NEO4J_USER`/
`NEO4J_PASSWORD` resolve from `.env` (PR #288), so retargeting is one line.

### 4. The 025 dataset

Query `experiments/025-solid-growth/queries/001_all_solid_growth.cql` (15 union
blocks; filters verified against the served build: `m.state='solid'`, no temperature
filter, graph_level per block). Build `experiments/025-solid-growth/scripts/query.py`
(slurm job 1559 + resumes) via `CompositeFitnessConverter` +
`MeanExperimentDeduplicator` + `GenotypeAggregator` + `SubgraphRepresentation`.
Pipeline: 43,819,983 fetched -> 27,040,733 after dedup (16.78M replicates merged) ->
13,525,071 genotype groups.

Four scale fixes were required at 43.8M records, each proven output-identical to the
old code on a smoke build (PRs #291/#292/#294/#298): return properties from cypher and
never nodes (the neo4j driver retains every hydrated Node); stream the experiment
reference index; two-pass streaming dedup/aggregation with `STAGE_COMPLETE` resume
markers; index passes read stored dicts instead of validating pydantic per record.

Final index summary (`experiments/025-solid-growth/results/dataset_index_summary.json`):

- 13,525,071 records; `phenotype_label_index`: fitness 13.53M, gene_interaction 13.52M
- `perturbation_count_index`: {1: 5,694, 2: 13,142,648, 3: 376,732} -- the triple
  count EXACTLY equals 010's dataset length
- Slice smf/dmf/tmf/dmi/tmi via `phenotype_label_index` x `perturbation_count_index`,
  not `dataset_name_index` (804 composite keys; pure-name keys are tiny)

Build root: `$DATA_ROOT/data/torchcell/experiments/025-solid-growth/001-full-build`
-> symlink to `/db/experiments/025-solid-growth-001-full-build` (regenerable, so the
/db rule holds).

### 5. 010 split parity

`experiments/025-solid-growth/scripts/transfer_010_tmi_splits.py` keyed on the sorted
gene-name SET (the `GenotypeAggregator` identity). A first pass keyed on
(gene, perturbation_type) matched only 682/376,732 and hard-failed, because the
perturbation-ontology refactor renamed types between builds (deletion ->
sga_kanmx_deletion/mean_deletion); the gene-set key matched 376,732/376,732 with zero
unmatched (fix PR #302). Result
`experiments/025-solid-growth/results/pinned_splits_from_010_seed_42.json`: train
301,386 / val 37,673 / test 37,673, identical to 010's `index_seed_42.json`.
`CellDataModule(pinned_split_indices=...)` (landed with 5 tests) reproduces 010's tmi
train/val/test exactly while all new data is seed-assigned around it.

### Next experiment - train over all fitness and interactions

Train on the full 025 dataset (both labels: fitness + gene_interaction, all
perturbation orders) with 010's tmi splits pinned, and compare tmi val/test metrics
against 010's triples-only runs. Open before training: inspect the smoke-flagged
genotype that aggregated 23 `SmfKuzmin2020` experiments (dedup identity may be finer
than intended); optionally reclaim 1.87 T of pipeline intermediates on /db
(conversion + dedup + aggregation stage copies; raw 839 G is refetch insurance, only
`processed/` 517 G is required at train time).

## 2026.09.04 - Training Campaign Plan

The revised subset-ladder, split, and model plan lives in
[[experiments.025-solid-growth.training-plan]]; the recapitulation gate (can 025's own
fitness reproduce its tmi labels) in
[[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]].
