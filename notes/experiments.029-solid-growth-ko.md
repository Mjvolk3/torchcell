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
