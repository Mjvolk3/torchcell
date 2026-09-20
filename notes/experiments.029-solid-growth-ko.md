---
id: vhrmvscpbx9umjmbw4o8wef
title: 029 Solid Growth Ko
desc: ''
updated: 1789739318148
created: 1789739318148
---

## 2026.09.18 - Why a new build, and what it changes

Successor of [[experiments.025-solid-growth]] decided from the S3 closure recompute
([[experiments.025-solid-growth.s3-closure]], `notes-tex/025-s3-closure/`). Three changes:

1. Deletions only. The query ([[experiments.029-solid-growth-ko.queries.001_ko_solid_growth]])
   keeps a record only when every perturbation is a deletion (`sga_kanmx_deletion`,
   `sga_natmx_deletion`). TS, DAmP, suppressor and generic alleles leave, so no measured allele
   shares a gene name with a deletion or with the SGD essentiality 0. About a fifth of the
   Kuzmin trigenic rows carry a TS array allele or a non-deletion query allele and leave with
   them; the retained triples keep their 010 split by gene-set transfer.
2. No mean-merge. The deduplication stage is skipped (`deduplicator=None`); the aggregator
   groups every source entry per genotype and each keeps its fitness, SD, score, p-value,
   temperature and screen. Which entry becomes a training label is a label policy applied at
   read time (to be written; the trainer masks a record with two values until then).
3. Both temperatures kept. Costanzo 26 C and 30 C and Kuzmin's 26 C screens are separate
   entries; the policy prefers 30 C for Costanzo and Kuzmin as screened (all Kuzmin selection
   steps ran at 26 C because the diagnostic array carries TS alleles).

Goal stated by the user: include as much deletion data as possible and test whether the
trigenic interaction can be reconstructed from the query's own singles and doubles better
than 025's r 0.230 (digenic 0.445). If it cannot, the join is the problem; if it can, the
approach gains confidence.

Build: `experiments/029-solid-growth-ko/scripts/query.py` under
`scripts/gh_query_build_001.slurm`; root on `/db/experiments/029-solid-growth-ko-001-ko-build`
symlinked from `$DATA_ROOT/data/torchcell/experiments/029-solid-growth-ko/001-ko-build`.
Stage optimizations landed for it in commit 2824b6335 (conversion byte passthrough, batched
raw writes, byte-keyed aggregation).

## 2026.09.18 - Plan: essentiality overlap, the linear model, and what the S3 cell embeds

### What the current S3 cell embeds

`cgt_s3_r_kl_fit_031` (and every R-split arm: fit_014, ctrl_013, the lambda ladder) trains
with `cell_dataset.node_embeddings: []`, that is the LEARNABLE TABLE only: `learnable_embedding`
size 180, a 2-layer preprocessor, 4,775,042 parameters, 8 layers, hidden 180. No sequence
embedding of any kind. The nine graphs (physical, regulatory, tflink, six STRING channels)
enter through the layer-1 attention penalty. The composite embedding
(`[fudt_upstream, calm, prot_T5_all, fudt_downstream]`, 3,328 dims) exists only on the Q-split
arms (`emb_017`, `embfit_027`); there is no R-split composite arm.

### Essentiality overlap with the 028 arm

028 (branch `feat/kinetics-equilibrator-datasets`, `experiments/028-gene-essentiality/`)
predicts essentiality from gene features and reports AUROC 0.893 (chrom_pathways + ProtT5 +
FBA affine, fold ensemble; MLP on the inputs 0.886) on the FCL split:
`results/splits/ess_fcl_val0.2_s0.json.gz` = 1,121 labeled genes, 157 essential, train 629 /
val 158 / test 195 (the 195 is the published held-out set), plus five folds
`ess_fcl_fold{0-4}of5_s0` and a genome-wide split `ess_genome_val0.2_s0` (5,795 genes, same
195 test).

In the 029 build an essential gene has exactly one record, the SGD entry converted to
fitness 0, because deletions only removes every measured TS or DAmP allele; a non-essential
gene has its Costanzo single plus every double it appears in. So "has any measured double"
already separates the classes, and a held-out gene must have EVERY record that contains it
removed from training, singles, doubles and triples alike, or the test leaks. The gene then
enters the model only through its position in the nine graphs and the attention of the other
genes: that is the hypothesis in its cleanest form, essentiality learned from the perturbation
space of the OTHER genes and the graph, with no per-gene feature.

Arm design, to be built once the label policy exists:

- Pool: S3 (or S2 first, singles + triples, 12 min/epoch, as the cheap pilot).
- Train: the pool minus every record containing a gene in the 028 val or test set of
  `ess_fcl_val0.2_s0` (353 genes). The trigenic val and test stay EXACTLY the pinned 010
  records (a val/test triple that contains a held-out gene is not trained on anyway), so the
  trigenic readout is comparable to the S3 arm, with train smaller by the removed records
  (count to be measured by the subset script).
- Readouts: (a) trigenic val/test as every arm; (b) essentiality: the predicted single-deletion
  fitness of each held-out gene, ranked, AUROC against the 028 labels on the same 158 val and
  195 test genes; report beside 028's 0.893 and its MLP 0.886.
- Cells: 3 seeds x {learnable table, composite embedding}. The table arm is the "no embedding
  tricks" reading; the composite arm says how much a sequence feature adds. Same three seeds
  and the same pinned trigenic splits (random, and the disjoint Q split as a second pool
  variant) as the rest of 025/029.
- Same design transfers to the five folds for a fold ensemble if the single split is
  promising; 5 folds x 3 seeds x 2 embeddings = 30 cells is too many at S3 scale, so folds
  run on S2.

### The linear model on 029

The additive null (B1 of the additive-baselines report, `additive_baselines_025.py`) ports
to 029 with three changes: it reads the policy's label table instead of `label_df` (one value
per record per label, the policy hash in the artifact name); the pinned splits transfer by
gene set as now; and fitness gets its own additive model, log f(S) = sum over deleted genes
of w_g (the multiplicative null, exact for non-interacting genes), fit on all orders with the
interaction ridge unchanged on the trigenic labels. The per-gene weight w_g of the fitness
model is then the linear essentiality score for a gene WITH records; a held-out gene has no
weight, so the linear essentiality baseline is a logistic model on gene features, which is
what 028 already reports. The physical null (tau recomputed from the policy's fitness values,
the S3 closure recompute rerun on 029) sits beside both, and beating 025's r 0.230 there is
the first thing 029 has to show.

## 2026.09.19 - Build 001 complete (slurm 2400)

Job 2400 finished at 05:10 CDT after 20 h 20 min, state COMPLETED, peak RSS 155 GB against
the 160 GB request (`sacct -j 2400`), so the next build of this size asks for 200 GB. Stage
wall times from the log and the stage directory mtimes, 025 beside them:

| stage | 029 (deletions only) | 025 (all alleles) |
|---|---|---|
| raw fetch | 5 h 58 min, 26,796,499 records | 17 h 55 min, 43.8M |
| conversion | 44 min, 15,137 converted, rest passthrough | 13 h 36 min |
| deduplication | none | 10 h 52 min |
| aggregation | 3 h 47 min, 9,297,912 genotypes | 3 h 43 min |
| processed copy | 18 min | 16 min |
| label table and indices | 9 h 33 min | 2 h 50 min |

The index stage is the one that grew: with no merge a genotype carries every source entry, so
the label table reads more entries per record. Counts written by the build:

| index | value |
|---|---|
| records | 9,297,912 |
| with a fitness label | 9,297,912 |
| with an interaction label | 9,284,062 |
| perturbation count 1 / 2 / 3 | 5,665 / 8,993,101 / 299,146 |
| Costanzo 2016 smf / dmf | 4,528 / 8,946,423 |
| Kuzmin 2018 smf / dmf / tmf | 1,170 / 280,827 / 57,883 |
| Kuzmin 2020 smf / dmf / tmf | 235 / 501,066 / 241,671 |
| SynthLethDB pairs | 13,990 |
| SGD essentiality | 1,140 |

Triples: 299,146 gene sets from 299,554 Kuzmin trigenic rows, against 376,732 in 025; the
difference is the 21 percent of trigenic rows with a temperature-sensitive array allele or a
non-deletion query, dropped by the deletion predicate. The 010 pinned val and test triples
therefore need a survival count before any training cell: a pinned triple whose only
measurement used a TS allele is absent from 029 and the split transfer by gene-set identity
must report it rather than silently shrink. Singles: 5,665 against 5,694.

Stage directories on /db: raw 615 G, conversion 614 G, aggregation 547 G, processed 547 G;
/db at 93 percent with 523 G free. The three intermediates are regenerable and go to /bulk
as tar.zst like the 025 ones, then off /db.

Next, in order: the read-time label policy, the pinned-split survival count, the S3 closure
recompute rerun on this build (target r above 0.230 trigenic and 0.445 digenic), then the
training cells.

## 2026.09.19 - Closure recompute on build 001: the policy decides whose identity holds

[[experiments.029-solid-growth-ko.scripts.closure_recompute]] reran the S3 closure recompute on
the deletion-only build with every source entry kept, under four label policies. Replaying the
025 join (mean of every entry) gives trigenic r 0.193 and digenic 0.435, within 0.04 of 025, so
the allele filter alone was not the loss. Kuzmin first gives the best trigenic reproduction
(0.245 overall, 0.517 with slope 0.93 on the Kuzmin 2018 screen) and the worst digenic (0.227);
Costanzo first the best digenic (0.508; 0.90 and 0.75 on the Costanzo 26 C and 30 C screens)
and the worst trigenic (0.137). Every screen reproduces itself and no other, so no single policy
is consistent for both orders; a model trained on both under one policy holds two standards for
the same gene. Kuzmin 2020 is capped at 0.41 / 0.29 under every policy by its unit query fitness.
The remaining gap to 1 on Kuzmin 2018 (0.52) is the released tables' row structure: the digenic
terms of a trigenic row come from the double-mutant query and its control queries, not from the
pair's own digenic screen, and the array single fitness is Costanzo's.

Consequences for the label policy: precedence Kuzmin (same year) > Kuzmin (other year) >
Costanzo 30 C > 26 C > converted 0 is the consistent choice for the trigenic target and is what
the source did; 1,116 of the 1,140 essential genes have no deletion measurement at all in this
build, so the converted 0 is a label for a gene with no record rather than a taint on a mean;
every interaction entry carries its source p (1,848,623 of 1,848,623). Pinned 010 splits transfer
at 79 percent (val 29,842, test 29,641 of 37,673); a 029 arm is compared with 025 on those
59,483 triples. Section 7 of `notes-tex/025-s3-closure` carries the figure and tables (12 pages,
`make check` clean). Copies: intermediates archiving to /bulk (slurm 2480), processed LMDB
mirroring to IGB scratch (slurm 2481, about 76 MB/s, 2 h).

## 2026.09.20 - Same genotype, several fitness values: noise or a hidden screen variable?

`scripts/same_genotype_spread.py` on the 029 closure entries (results/same_genotype_spread.json).
For every genotype with more than one fitness entry, the spread is split into the part within one
screen and the part between screens, and same-genotype pairs across screens are correlated.

| comparison, same genotype | n | r | mean diff | median abs diff |
|---|---|---|---|---|
| singles: Costanzo 26 C vs 30 C | 4,515 | 1.000 | 0.0000 | 0.0000 |
| singles: Costanzo 30 C vs Kuzmin 2018 query single | 1,158 | 0.984 | +0.012 | 0.011 |
| singles: Costanzo 30 C vs Kuzmin 2020 query single | 226 | 0.650 | +0.036 | 0.040 |
| doubles: Costanzo 30 C, the two query/array orientations | 353,218 | 0.891 | | 0.031 |
| doubles: Costanzo 30 C vs Kuzmin 2018 | 118,280 | 0.857 | +0.013 | 0.042 |
| doubles: Costanzo 30 C vs Kuzmin 2020 | 441,724 | 0.741 | -0.009 | 0.047 |
| doubles: Kuzmin 2018 vs 2020 | 13,162 | 0.603 | -0.036 | 0.048 |
| digenic interaction: Costanzo 30 C, two orientations | 353,218 | 0.142 | | |
| digenic interaction: Costanzo 30 C vs Kuzmin 2018 | 118,280 | 0.318 | -0.015 | 0.029 |

Variance split: singles, total sd 0.015, within-screen sd 0.013, between-screen share 0.25 (the
Costanzo 26 C and 30 C singles are one genome-wide standard stored twice, r 1.000). Doubles, total
sd 0.048, within-screen sd 0.021, between-screen share 0.81; but the two ORIENTATIONS of one pair
inside Costanzo, which are two independent strains, agree at r 0.891, the same level as Costanzo
against Kuzmin 2018 (0.857), and the mean offsets between screens are 0.01. So for fitness the
screen of origin is not a hidden variable: two screens disagree about a double as much as two
strains of the same double disagree inside one screen, with no systematic shift. For the digenic
interaction the same-pair agreement is near noise even within one lab (0.142 across orientations),
because most scores are near zero and the score is a difference of noisy terms; a screen token
would memorize that noise, not explain it.

Consequence for the model: no dataset token. Several entries of one genotype are replicates of one
quantity, so a loss over all of them, or a read-time mean with inverse-variance weight, learns the
conditional mean; temperature, which the record carries, goes onto the input. The interaction is
better derived from predicted fitness under the identity with a declared reference than learned as
a per-screen quantity; the 030 recapitulation is the test of that. Revisit only if a target on 030
shows a large between-screen share together with a large mean offset, which would be a
calibration term marginalized to a reference at inference, not an input.

## 2026.09.20 - The identical Costanzo 26 C and 30 C singles: a source-level duplication the loader turned into two assertions

Checked by an agent against the loader, the raw spreadsheet and the SI (issue
<https://github.com/Mjvolk3/torchcell/issues/410>). Costanzo 2016 released ONE temperature-combined
single-mutant fitness for deletion and DAmP strains, repeated in both the 26 C and 30 C columns:
SI line 96, "Because we observed a close correlation between fitness measured at 26°C and 30°C for
deletion mutants, we combined measurements from different temperatures in the average for each
deletion mutant. Fitness associated with TS mutants was computed separately at either 26° or 30°."
Raw columns: KanMX n 3,876 and NatMX n 3,845 and DAmP n 733 are 100 percent identical across the
two columns; TS alleles n 1,753 are 0 percent identical, r 0.810. `SmfCostanzo2016Dataset.preprocess_raw`
splits every strain class into a 26 C and a 30 C record, so 8,454 of the 20,484 built records
(41 percent) are the twin of a deletion or DAmP record. The values are faithful; the temperature
assertion is not. The double-mutant loader is clean (temperature from `Arraytype/Temp`).

Consequence: for the 030 build the reader drops the twin (a deletion or DAmP SMF is one value with
no temperature); the loader fix (one record, temperature unspecified or a range) is a served-dataset
change and waits for the next full knowledge-graph rebuild. TS-allele singles stay per temperature.
