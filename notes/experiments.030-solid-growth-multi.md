---
id: f91m1ih5ek9k40dfulg82tt
title: 030 Solid Growth Multi
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

## 2026.09.20 - Which 028 split the essentiality overlap uses, and what "held out" means

Corrected by the 028 session. The held-out set is the 195 released test genes, not 353. The
released Merzbacher file marks 223 genes as test; 195 of them are in the build and carry 31
essential. The 158 validation genes pick the epoch in 028 and are not scored, so they need not
leave a fitness training pool. Two genome splits exist and differ: `ess_genome_val0.2_s0` changes
only the training labels and keeps the same 195 test genes, while `ess_genome_heldout_val0.2_s0`
trains on the released genes and tests on 4,674 genome-wide genes whose label origin coincides with
class and none of which carries a metabolic-model reaction.

Measured on the 025 S3 pool (1,121,645 records), holding out every record that contains a held-out
gene:

| held out | genes (essential) | records lost | triples lost |
|---|---|---|---|
| released test only | 195 (31) | 68,918 (6.1%) | 27,661 (7.3%) |
| released test + validation | 353 (56) | 124,521 (11.1%) | 50,855 (13.5%) |
| released test + the 10 resplit test sets | 896 (143) | 316,398 (28.2%) | 116,788 (31.0%) |
| every labeled gene present in the pool | 991 (156) | 344,391 (30.7%) | 122,611 (32.5%) |

The 028 session's caveat decides the design: the released 195 rank more easily than a random fifth
of the labeled genes (layer-free transformer 0.908 on the released set against a mean 0.843 over
ten re-splits), so any difference this arm shows must be confirmed on `ess_fcl_resplit{0..9}`. Each
resplit has its own 198 test genes overlapping the released set by 26 to 42, so confirming by
retraining is eleven cells.

Design, two cells rather than eleven:

- **Cell A, comparable**: hold out the 195 only, 6.1 percent of the pool. This is the number that
  sits beside 028's 0.893 and Flux Cone Learning's 0.742 on the same genes.
- **Cell B, confirmable**: hold out all 991 labeled genes present in the pool, 30.7 percent. One
  trained model then scores the released 195, all ten re-splits and the five folds with no leakage
  and no retraining, for 2.5 points more removal than the resplit union alone. Its caveat: the
  labeled genes are the metabolic-model genes, so the removed block is functionally coherent rather
  than a random sample, and it trains on a third less data than Cell A.

Pilot both on S2 (about 12 min an epoch) before promoting either to S3 (about 36 min an epoch, 130
epochs). The 028 split files are gene lists, so on 030 the same file defines the subset by gene
membership over the full-allele records.

## 2026.09.20 - The query-strain fitness is already on disk: a released standard, not a reference

Checked whether the double-mutant query fitness the trigenic identity needs is a reference to a
record we already hold. It is not a reference, and we do have the file.

Both Kuzmin papers released a fitness standard for their query strains, and both are already
mirrored:

| year | file | rows | rows with fitness (all of which carry a sd) |
|---|---|---|---|
| 2018 | `torchcell-library/kuzminSystematicAnalysisComplex2018/si/si_data/Data File S4_Fitness standard for single and double mutant query strains.xlsx` | 546 (364 double, 182 gene+HO single) | 507 |
| 2020 | `data/torchcell/dmf_kuzmin2020/raw/aaz5667-Table-S5.xlsx` | 720 (240 double, 480 single) | 673 |

Every value carries a standard deviation: 507 of 507 in 2018 and 673 of 673 in 2020, median 0.013
and 0.005. The trigenic table's own query-fitness column has none, which is why the 2018 loader
records `fitness_std = np.nan` for it. Both files also list the single-mutant control query
strains, the other term of the identity. The 2020 file is already in the raw mirror the digenic
loader reads, so no retrieval is needed for either.

Coverage of the double-mutant query strains the trigenic rows use, and agreement where both exist:

| year | strains | in both sources | standard only | trigenic column only | neither | union of rows covered | rows with a sd |
|---|---|---|---|---|---|---|---|
| 2018 | 182 | 165 (r 1.000000, max diff 5e-5) | 0 | 7 | 10 | 86,111 of 91,111 (94.5%) | 82,460 (90.5%) |
| 2020 | 240 | 201 (r 0.993, max diff 0.168) | 0 | 0 | 39 | 256,852 of 301,798 (85.1%) | 256,852 (85.1%) |

So the standard adds the uncertainty rather than the coverage: it never has a value the trigenic
column lacks, and in 2018 the column has 7 strains it does not. Ingesting both and preferring the
standard for its sd, falling back to the column, covers 94.5 percent of the 2018 trigenic rows and
85.1 percent of the 2020 rows.

It is a distinct measurement, not a duplicate of a record we hold. Of the 172 Kuzmin 2018 query
pairs with a released value, 129 also exist as a double in the 029 build from an array screen, and
those two measurements of the same genotype agree at r 0.777 with a median absolute difference of
0.045. Substituting the array-screen value is what costs the reconstruction: within a screen the
identity reproduces the published score at 0.985, and replacing this one term drops it to 0.538.

So the work is ingestion only: two loader classes reading local files, provenance sourced from the
SI, a dev build, the admission check and an incremental import. No data chase.

## 2026.09.20 - Exactly how the published trigenic score is computed, and what that means to ingest

Verified against the raw tables for both years
(`experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen.py`,
results/s3_closure_trigenic_within_screen.json). The released score is

    tau_ijk = f_ijk - f_ij * f_k - eps_ik - eps_jk

that is, the tau-SGA model with the single-mutant query fitness set to 1.

| variant | Kuzmin 2018 r | median residual | Kuzmin 2020 r | median residual |
|---|---|---|---|---|
| as published, f_i = f_j = 1 | 0.990 | 2.9e-05 | 0.976 | 2.9e-05 |
| f_i, f_j read from the control rows' own released fitness | 0.985 | 1.6e-03 | 0.976 | 2.9e-05 |
| f_ij dropped | 0.495 | 7.4e-02 | 0.420 | 4.1e-02 |

Exact to within 1e-4, the rounding of the published five-decimal value, on 95.0 percent of
the 91,111 2018 rows and 92.0 percent of the 301,798 2020 rows.

**The single-mutant control query strains' FITNESS is not used.** 2020 releases it on 0.4 percent
of control rows and the SI says every missing fitness was set to 1.0 during scoring. 2018 releases
it on 99.2 percent and still did not use it: substituting those measured values makes the
reproduction fifty times worse. What a control screen contributes to a trigenic score is its
INTERACTION, eps_ik and eps_jk, and those are already in the graph as the Kuzmin digenic records.

**So the ingestion list is one item, not two.** The double-mutant query strain fitness f_ij is the
only term missing. The single-mutant query fitness standard is not needed to reproduce published
scores; it is optional and would only support a recomputation that departs from the source, which
is a policy question and not a prerequisite. Recommend ingesting the double-mutant query strains
only: 364 rows in the 2018 standard and 240 in the 2020 one, each with a standard deviation.

**What the label policy must do for the trigenic target**, now fully determined: f_ijk and f_k from
the triple's own record; f_ij from the query-strain record matched by strain id; eps_ik and eps_jk
from the Kuzmin digenic records of the same study; and the query singles entering as 1, which
becomes the `source_convention` switch. Departing from the last of these is what a `measured`
convention would mean, and it is a different number from the published one by construction.
