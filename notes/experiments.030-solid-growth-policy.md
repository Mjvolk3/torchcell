---
id: f91m1ih5ek9k40dfulg82tt
title: 030 Solid Growth Policy
desc: ''
updated: 1789932265479
created: 1789932265479
---

## 2026.09.20 - Design of the 030 query and build

**Principle.** The query decides what the dataset is: the gene universe, the phenotypes, the
medium, the record shape. It does not decide which measurement of a genotype counts, because that
choice is a modeling assumption and belongs with the run, next to the split and the seed. A build
that has already chosen has thrown the alternatives away, and the store is the one thing that is
expensive to make. So 030 = the 025 query (all alleles) + the 029 build (no merge) + the one record
both lacked, the Kuzmin double-mutant query-strain fitness.

**Why a third build.** The within-screen control ([[experiments.025-solid-growth.s3-closure]],
2026.09.19) reproduces the released trigenic score at r 0.985 (Kuzmin 2018) and 0.976 (2020) from
the triple row plus its two control-query rows; replacing only the double-mutant query fitness by
the product of singles drops it to 0.538 / 0.419. That measurement is released on 86,111 of 91,111
2018 rows and 256,852 of 301,798 2020 rows and is ingested by no loader (both filter to digenic rows
before the branch that reads it). The 029 Kuzmin-first closure stops at 0.517 for exactly this
reason. No label policy over the current records can pass about 0.54 trigenic.

### What the query decides (dataset)

| decision | 030 | inherited from |
|---|---|---|
| gene universe | `$gene_set`, the 6,607 S288C genes | 025, 029 |
| phenotypes | fitness (smf/dmf/tmf), gene interaction (dmi/tmi), essentiality, synthetic lethality; SynthRescue excluded (no conversion) | 025 |
| medium | `m.state = 'solid'`; no temperature filter | 025 |
| alleles | ALL perturbation types (deletion, TS, DAmP, suppressor, allele) | 025; reverses 029 |
| SynthLethDB shape | `SIZE(pert nodes) = 2`, drops the self-pairs that put two fitness values in one record | 029 |
| NEW blocks | the Kuzmin double-mutant query-strain fitness, 2018 and 2020, as their own datasets | new |

Restoring all alleles restores the 376,732 triple identities of 010 and 025, so the 010 pinned
random split and the query-pair-disjoint split transfer exactly, and deletions-only becomes a
SUBSET INDEX over 030 (like S0/S2/S3), not a separate build. The 029 build is subsumed.

### What the query does not decide (policy, at read time)

- which entry of a record supplies each label: precedence Kuzmin same year > other year > Costanzo
  30 C > 26 C > converted 0 (0 only without a measurement);
- for a triple's three doubles: the entry whose strain id equals the triple's query strain id
  (this is the new record), else the pair's own digenic screen under the precedence;
- for a triple's array single: the Costanzo smf entry matched by array strain id where present,
  else by gene (Kuzmin's array fitness IS Costanzo's, si1.md line 135);
- same-source replicates: mean, sd propagated, Stouffer over source p;
- the Kuzmin 2020 single-control convention: the source scored with 1.0 for every NaN query fitness
  (only 2 of 480 control queries carry a value); a policy switch `source_convention` reproduces the
  released score, `measured` substitutes Costanzo. A policy option, not a query decision.

The policy is a pydantic object, hashed; it emits one label table, record index to one value per
label with the supplying entry, cached beside the split caches. Two arms on one build under two
policies are two run configs.

### The new record

`QueryDmfKuzmin2018Dataset` / `QueryDmfKuzmin2020Dataset` (names provisional): one fitness record
per double-mutant query strain, from the trigenic rows' "Query single/double mutant fitness"
column, both perturbations carrying `strain_id` = the query strain id (`GENE1+GENE2_tmNNNN`), the
form the trigenic records' query perturbations already carry. Counts: 2018, 172 of 182 strains
with a value; 2020, 201 of 240. n_samples and uncertainty type from the SI (query fitness is the
12 to 24 colony bootstrap standard, to be re-sourced with quote + sha256 per the loader rules).
Additive classes, so the kg_manifest admission gate should admit them for incremental import
(`gilahyper_increment_kg-slurm_docker.slurm`, manifest at
`/scratch/projects/torchcell/database/kg_manifest.json`, bootstrapped 2026-09-18, 51 datasets).

### Build

`converter=CompositeFitnessConverter, deduplicator=None, aggregator=GenotypeAggregator`, root on
/db, launched from a detached worktree as 029 was. Projection from the measured 029 rates scaled
to 43.8M + about 0.4k records: raw about 10 h, conversion about 1.2 h, aggregation about 6.5 h,
indices 15 to 20 h (029's index stage was the one that grew without the merge), about 35 h total.
Memory: 029 peaked at 155 GB for 26.8M records; linear scaling says about 255 GB, and the DB build
reserves 256 GB of the 513 GB node, so aggregation pass 1 (one key per record) should be measured or
made leaner (hashed keys) before launch. Disk: 029's four stages took 2.3 TB, so 030 needs about
3.7 TB; /db has 523 G free now, plus 1.8 TB once the 029 intermediates are deleted and 2.7 TB once
the 025 intermediates are, both archived on /bulk. Both deletions must precede the build.

### Order of work

1. Delete 025 and 029 intermediates from /db (archives verified on /bulk).
2. Rebase the branch onto main (120 commits behind; kg_manifest lives there).
3. Write the two query-strain loaders + tests + SI-sourced provenance; dev build; L0-L4.
4. `kg_manifest admit`; incremental import into the served store.
5. 030 query = 025's 15 blocks + the 029 SynthLethDB node rule + 2 new blocks = 17; validate on a
   20-gene set; launch the build.
6. Label policy written and tested against 029 while 030 builds (only the strain-matched double
   rule waits for 030).
7. On 030: closure recompute (hypothesis, untested: Kuzmin-first with the strain-matched double
   reaches about 0.98 on Kuzmin trigenic entries), then the S3 / deletions-only / essentiality
   subsets and the training cells.

### Not a dataset id in the model

Two measurements of the same genotype are two samples of one quantity; a squared-error loss over
both is minimized at their mean, so the model learns the mean with no dataset token. Where two
values are genuinely different quantities the record already carries the distinguishing field
(temperature; for interaction, the screen whose controls defined it). Fitness as f(genotype,
environment) needs no dataset id at inference. A dataset token would absorb every between-screen
offset and demand a screen name at inference, which is the limitation to avoid.
