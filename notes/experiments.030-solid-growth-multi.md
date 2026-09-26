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

## 2026.09.20 - What already existed on deriving the interaction from fitness

Four prior passes, none of which tested the identity against the source tables.

| when | where | what it did | result |
|---|---|---|---|
| 2025.01 to 2025.04 | `notes/phenotype.gene_interaction.md` | derives the symmetric form tau = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k and generalizes it to n-way by inclusion-exclusion | the formula the S3 closure recompute used |
| 2025.04.18 | 003-fit-int, `hetero_cell_bipartite_bad_gi_analytic_v_direct.py` (commit 8aafe6e3c) | predicted digenic epsilon directly versus derived it from predicted fitness, n = 22,303 | direct Pearson 0.0714, derived 0.0171; read at the time as the derived route being worse |
| 2025.10.21 | 007-kuzmin-tm FBA | the same formula on Yeast9 flux-balance fitness, after first implementing it without the digenic terms | r 0.0019 against experimental tau, plus a row-alignment bug found later |
| 2026.09.01 | 010 additive baseline | writes the asymmetric form to argue triples of one query pair are not independent draws | structural, never measured |

The 2025.04 comparison now has a data explanation it could not have had then. In the 025 build the
stored fitness and the stored interaction are only consistent at r 0.445 digenic and 0.230
trigenic, so a perfectly accurate fitness predictor could not have derived the stored score better
than that. The derived route was not losing to the direct route because deriving is worse; it was
losing because the two labels in the build disagree. Within one source screen the identity is
exact, so the ceiling is a property of the join, not of the arithmetic.

## 2026.09.20 - It is dmf, and both loaders were written to include it

Not a new dataset. The double-mutant query strain's fitness is double-mutant fitness, and both
`DmfKuzmin` loaders already contain the code to ingest it. Neither runs it.

**`DmfKuzmin2018Dataset`.** `preprocess_raw` (kuzmin2018.py line 442) filters the frame to
`Combined mutant type == "digenic"`. `create_experiment` (lines 539 to 542) then branches on
`elif row["Combined mutant type"] == "trigenic": dmf_key = "Query single/double mutant fitness"`,
which the filter has made unreachable. The comment beside it, "std of these fitnesses not
reported", is also wrong now: Data File S4 reports it for 339 of the 364 double-mutant query
strains, and the loader never opens that file.

**`DmfKuzmin2020Dataset`.** `preprocess_raw` (kuzmin2020.py lines 306 to 324) does open Table S5,
filters it to `Mutant type == "Double mutant"`, and merges it onto the digenic frame with
`left_on="Query strain ID"` against `right_on="Query Strain ID"`. The left key is the full strain
string (`YAL015C+YDL227C_tm461`) and the right key is the bare tm number (`tm461`), so the merge
matches **0 of 537,911 digenic rows** and the `fillna` fallback silently uses the S1/S3 columns
every time. The mismatch warning below it can never fire. Two defects in one block: the join key,
and the row set, since an S5 double-mutant entry describes a TRIGENIC row's query strain, not a
digenic one. Joined correctly, trigenic rows on the tm number, it matches 215,617 of 256,861 rows.

So the intent was right in both places and the execution is dead code in both. Nothing belongs in
`dmi`: a trigenic row carries no interaction score for its query pair.

**Consequence for the graph.** The manifest fingerprints each served dataset's loader closure
(`loader_relpath`, `loader_closure_from_source` in `kg_manifest.py`), so editing `preprocess_raw`
of a served dataset is served schema drift and the admission gate blocks: FULL REBUILD, not an
incremental import. That is the honest cost of modeling it correctly, and it is the right moment to
land issue #410 as well, the Costanzo single-mutant fitness stamped at two temperatures, which is
also rebuild-only. One rebuild, two verified defects.

**A consequence for the build, not the loader.** After the fix a gene pair can carry two
double-mutant fitness records: its array-screen measurement and its query-strain measurement, which
agree at r 0.777 with a median absolute difference of 0.045. They are different strains and
different experiments, distinguished by `strain_id`. A no-merge build keeps both, which is what the
identity needs. The 025 merge would average them into one value that is neither, which is a further
reason the deduplication stage does not return.

## 2026.09.20 - Three independent audits, and a correction upward

Three agents re-derived the claims from the raw tables and the supplementary text without using
our scripts. All three CONFIRMED. Two numbers of mine were wrong and both move in the right
direction.

**The reproduction is 99.98 and 99.56 percent, not 95 and 92.** My matching paired a trigenic row
to its control screens by gene name and averaged where a gene had more than one. Five 2018 genes
carry two control strains, and 99 percent of my failures sat on those five. Pairing through the
released query-strain list (Data File S3) instead gives 99.98 percent on 91,111 Kuzmin 2018 rows
and 99.56 percent on 301,798 Kuzmin 2020 rows, median absolute residual 2.7e-05, which is the
rounding floor of the four-decimal fitness columns rather than model error.

**Written on the released columns it is an exact identity.** The trigenic row's own RAW score is
eps_ij,k = f_ijk - f_ij f_k, so tau = raw_eps - eps_ik - eps_jk, the row's raw score minus the two
control queries' adjusted scores at the same array. Median residual 6.5e-19 on 2018 and exactly 0
on 2020. Two routes therefore reconstruct the published score: through the query strain's fitness,
or through the row's raw score. The build stores neither column; the raw interaction score is
ingested by no loader (every Dmi/Tmi class takes only "Adjusted genetic interaction score").
The fitness route is the one worth having, because f_ij is a genotype's measured fitness and
enriches the label space, while the raw score is a derived quantity.

**The SI does not say the query singles are 1, and for 2018 the released values disagree with the
method as described.** Both papers write the model with the control scores weighted by f_i and f_j,
and the 2018 text says these "are single mutant fitness estimates available from a previous study"
(si1.md line 135). For 2020 this reconciles: 0.4 percent of control rows carry a query fitness and
the methods say every missing fitness was set to 1.0 during scoring. For 2018 it does not: 99.2
percent of control rows carry the measurement and the released standard carries it for 165 of 182
query strains, yet using either reproduces 4.3 and 5.5 percent of rows against 99.98 percent for
unit weights. The released 2018 scores were computed with weights the methods do not describe.
State it that way in anything outward-facing: it is a reproducibility observation about the source,
evidenced, not an accusation.

**Loader defects: both confirmed exactly.** 2018 built 410,399 records, the digenic row count, so
no trigenic row reached it; Data File S4 holds 182 double-mutant query strains, 165 with a fitness
and a standard deviation, agreeing with the raw column at r 1.000000. 2020 built 632,797 records,
its digenic row count, and the Table S5 merge matched 0 rows. A third suspected defect was flagged
and left alone: `SmfKuzmin2020Dataset` labels S5's bootstrap standard deviation as a sample sd over
four colonies, where the SI describes bootstrapped means over 12 to 24 colony measurements.

## 2026.09.26 - Per-entry training with a source-dataset token: built, smoke queued

The training path for this build (plan [[plan.030-per-entry-dataset-token.2026.09.25]], branch `feat/030-per-entry-dataset-token`, PR #444). Every stored entry is a training row; the row's source dataset enters the readout of both heads as a one-hot over the build's 15 source datasets projected to 8 learnable dimensions (`CellGraphTransformer.dataset_token`, readout mode; the input-side placement is the deferred ablation). Validation and test on the pinned triples report one value per genotype under its own screen's token, the label policy's precedence (Kuzmin 2018, 2020, Costanzo, converted 0), so `val/gene_interaction/Pearson` reads against the 025 S3 arms; beside it the per-entry, per-token and cross-token (2018 rows under the 2020 token and back) Pearson. The single-value path is untouched when the token is off (tested bit-for-bit).

The arm ([[experiments.030-solid-growth-multi.scripts.arm_030]]): S3 (1,121,662) minus the essentiality holdout (698 singles: the 198 released Merzbacher test genes that resolve here plus 250 + 250 coverage-matched singles, [[experiments.030-solid-growth-multi.scripts.build_essentiality_holdout_030]]) = 1,120,964 records; 010's random split pinned over the triples (301,386 / 37,673 / 37,673), everything else in train, 1,045,618 training records. The holdout is served as a second validation loader and scored as the AUROC of predicted single-deletion fitness (negated) against essentiality under the Costanzo single token and the SGD token (`val_ess/*`); PPI degree alone gives 0.591 released and 0.706 matched, the confound baseline. Normalization constants are fitted once on the training ENTRY rows and committed with the training set's fingerprint ([[experiments.030-solid-growth-multi.scripts.make_normalization_stats_030]]: gene_interaction sd 0.0506 over 2,339,105 rows, fitness sd 0.1579 over 2,360,393 rows).

Smoke ([[experiments.030-solid-growth-multi.scripts.gh_smoke_dataset_token]], GilaHyper job 2861, one GPU): the split cache for seeds 0 to 2 is warmed under the final pool, then a token run and a control run in which every entry is cloned 0.3 normalized units higher under a synthetic token (or, in the control, under its original token), and [[experiments.030-solid-growth-multi.scripts.smoke_report_030]] applies the three criteria of the plan. On PASS: the build and the cache to IGB ([[experiments.030-solid-growth-multi.scripts.gh_sync_igb_030]]), then seed 0 of the first arm, closure composite + fitness (`cgt_030_s3_r_tok_embfit_001`), on mmli ([[experiments.030-solid-growth-multi.scripts.igb_mmli_cgt_030]]); seeds 1 and 2 after the first epoch's wall time and MaxRSS are read.
