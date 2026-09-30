---
id: bhptaavkrs4pbjfivjsbd4g
title: 035 Env Chemgen Vanacloig Cgt
desc: ''
updated: 1790715991314
created: 1790715991314
---

## 2026.09.29 - Vanacloig alone, compound-cold: design and the first finished arm

The first modeling experiment of the inhibitor-tolerance line. The question is whether the
cell graph transformer predicts Vanacloig 2022 for a compound it has not seen once the
compound reaches the readout. Plan: `notes-tex/031-unified-representation` (PR #436). Store
check: `notes-tex/033-post-query-analysis` (PR #455). This branch is PR #473.

**Inputs, by path.** The cell table of experiment 033
(`$DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table/cell_table.parquet`, slurm 2988)
and the FCFP4 table of experiment 031 (`results/embeddings/fcfp4_count.npz`). 143,218 cells,
3,598 queried genes, 41 compounds.

**Strain.** Every strain is the queried deletion plus the three host deletions (PDR1, PDR3,
SNQ2), so the perturbation set has four genes and three are constant. The library is not
BY4741: Piotrowski 2017 builds the host deletions in the SGA query strain Y7092 (Y13206)
and crosses it to the MATa deletion array, so each strain is a haploid segregant of two
S288C-derived parents. The selection Piotrowski describes (canavanine and thialysine)
implies the query strain's `can1` and `lyp1` deletions are present as well, which the
loader does not record, and no SGA-derived loader does. The background is constant across
the library, so none of this changes a Vanacloig-only result.

**Split and score.** Five grouped folds over compounds
([[experiments.035-env-chemgen-vanacloig-cgt.scripts.vanacloig_data]]); every compound is
tested once; four validation compounds per fold pick the epoch. Per held-out compound,
Spearman over its genes, raw and centered, beside the compound's ceiling. Each side of the
centered score is centered by its OWN mean over the training compounds. Subtracting the
measured gene mean from the prediction rewarded an untrained model: 0.356 centered and
-0.001 raw at 60 steps (slurm 3009).

**Same-fold references**
([[experiments.035-env-chemgen-vanacloig-cgt.scripts.baselines_same_folds]]), median
Spearman over the 41 test compounds, 28 or 29 training compounds per fold:

| model | raw | centered |
|---|---|---|
| gene mean | 0.209 | undefined |
| ridge on FCFP4 | 0.286 | 0.316 |

These agree with the 031 leave-one-compound-out values (0.212, 0.311).

**Arms** ([[experiments.035-env-chemgen-vanacloig-cgt.scripts.train_vanacloig_cgt]]): the
encoder is used unchanged and the readout is built in the experiment, `[h_CLS ; z_S ; z_E]`
with `z_S` the sum of the perturbed embeddings over the four genes and `z_E` a projection of
`log1p` of the fingerprint. Model half from the 025 replication config; batch 256, 12 epochs,
AdamW with a one-cycle schedule to 5e-4. Batch 512 with the graph prior ran out of memory at
44 GB (slurm 3010).

| arm | launch | slurm | state on 2026.09.29 16:40 |
|---|---|---|---|
| `embedding_env`, no encoder | CPU | 3020 | finished, 5 folds |
| `cgt_env`, graph prior 1 | GPU | 3016 | partial, fold 0 at epoch 2 of 12 |
| `cgt_env`, graph prior 0 | GPU | 3017 | partial, fold 0 at epoch 3 of 12 |
| `cgt_genes`, floor, folds 0 and 1 | GPU | 3019 | queued |

**Finished: the embedding table is below ridge.** `embedding_env` over all 41 test compounds
at the validation-selected epoch
([[experiments.035-env-chemgen-vanacloig-cgt.scripts.summarize_round]],
`results/round_summary.csv`): median Spearman 0.134 centered against ridge 0.316 on the same
compounds, above ridge on 14 of 41; 0.190 raw against ridge 0.286 and gene mean 0.209. Per
fold the centered median runs from 0.005 to 0.318. Validation loss is unstable: the last
epoch's validation MSE is 9.12 on fold 3 and 1.12 on fold 1 against 0.30 to 0.54 on the
others.

**Observed, not explained.** Sodium glyoxylate has a reliability index at or below zero, so
its ceiling is recorded as zero, and both the embedding arm (0.402) and ridge (0.316) score
it well above zero on the centered target.

## 2026.09.29 - Without the graph prior the encoder collapses every gene token

Measured by [[experiments.035-env-chemgen-vanacloig-cgt.scripts.diagnose_token_collapse]]
(slurm 3024, `results/token_collapse.csv`) on 512 sampled genes and 512 sampled strains per
checkpoint. "Across share" is the fraction of the tokens' total squared norm that varies
across rows: 1 is every row distinct, 0 is every row identical.

| model | gene tokens, across share | strain summary, across share | prediction sd over strains | over compounds |
|---|---|---|---|---|
| transformer at initialization | 0.073 | 0.005 | 0.011 | 0.001 |
| transformer, prior 0, fold 0 | 0.0000 | 0.0000 | 0.0000 | 0.539 |
| transformer, prior 0, fold 1 | 0.0002 | 0.0000 | 0.0000 | 0.119 |
| transformer, prior 1, fold 0 (epoch 10 of 12) | 0.399 | 0.562 | 0.031 | 0.418 |
| embedding table, 5 folds | none | 0.28 | 0.07 to 0.57 | 0.19 to 0.71 |

Without the prior, training drives all 6,607 gene tokens onto one vector: the strain
summary is identical for every strain and the prediction is one value per compound. That is
the round-1 symptom: train MSE flat at 0.60 (fold 0) and 0.975 (fold 1), within-compound raw
Spearman near zero. Fold 0 of that arm was selected at 0.063 centered. The arm was cancelled
after fold 1 (slurm 3017), and the genes-only floor with it (slurm 3019), since neither could
be read while the encoder collapses.

With the prior at 1 the tokens stay distinct, and fold 0 was still improving at epoch 10 of
12 (test centered median 0.129, raw 0.024). Its prediction still varies 14 times more over
compounds than over strains.

Hypothesis (untested): the encoder starts near collapse, with gene tokens at a mean cosine of
0.93 at initialization, and nothing but the graph prior pushes them apart, because a readout
that can fit the compound effect alone gets most of its loss reduction without them.

**Next arm, launched.** The prior at 1 plus an identity skip: the strain's genes summed from
the encoder's input embedding table, passed to the readout beside the transformer summary
(`model.identity_skip`, slurm 3025). It asks whether the transformer adds anything once strain
identity cannot be lost.

## 2026.09.29 - Model search: best compound-inductive model on Vanacloig alone

**Question.** Which model predicts the Vanacloig gene profile of a compound it has never seen,
from the compound's embedding alone, and does the cell graph transformer beat the simpler
ones? Budget: 48 hours, two GPUs, beside the two round-1 GPU jobs (slurm 3016, 3030) that
finish tonight.

**Data.** 143,218 cells: 3,598 queried genes by 41 compounds, 97% filled, one measurement per
cell (mean of three biological replicates). Median served SE 0.138 against a response sd of
0.699. Median per-compound ceiling 0.834 (square root of the reliability index).

**Protocol, fixed before any result.**

- Compound-cold folds of `vanacloig_data.make_folds`, 5 folds, fold seeds 0, 1 and 2.
- Every choice is nested. The ladder picks options by leave-one-compound-out over the fold's
  non-test compounds. The neural models pick their step on the fold's 4 validation compounds.
  The test compounds never influence a choice.
- The primary target is centered Spearman, per compound, and the headline statistic is its
  median over the 41 compounds.
- Every model is compared PAIRED against two references on the same compounds, with a
  bootstrap interval over compounds (`compare_models.py`). The references are nested ridge on
  FCFP4 (`ladder:krr|linear:fcfp4_count`) and the ladder's whole-pipeline pick
  (`ladder:selected`).
- Test centering for the ladder and the factorized models uses the mean over the fold's
  non-test compounds. Round 1 (`train_vanacloig_cgt.py`, `baselines_same_folds.py`) centered on
  the 28 training compounds, so those numbers are not paired with round 2.

**Rounds.**

1. `baseline_ladder.py`: kernel ridge and nearest-neighbor maps over 12 compound embeddings
   and their kernels (slurm 3034).
2. `train_factorized.py`, `conf/factorized/r2_table.yaml`: a learned gene table times a
   compound encoder, bilinear or MLP head, 5 folds by 3 seeds plus the seed ensemble
   (slurm 3035).
3. The same trainer with the cell graph transformer as the gene encoder,
   `conf/factorized/r2_cgt.yaml`. One encoder pass per strain per step, not per cell.
4. Hypothesis to test if Vanacloig alone plateaus: pretrain the compound encoder on the other
   chemogenomic stores, excluding every Vanacloig compound. Wildenhain has 5,170 compounds on
   242 genes, Hillenmeyer heterozygous has 303 compounds and Hoepfner 148 with InChIKeys.
   Wildenhain shares 10 of the 41 Vanacloig compounds, and Hoepfner and Hillenmeyer 2 each.

## 2026.09.29 - Audit of the ingestion, and what it changes for the scoring

A full reread of the paper, Piotrowski 2017 and the GEO raw counts against the loader and the
cell table (issue #501 holds the report; issue #500 the strain background). The parquet
reproduces the loader's formula exactly, the sign is right (benomyl TUB3 is the most depleted
strain), `response_ses` is the replicate SD over sqrt(3), and every InChIKey matches its
name. Three findings bear on modeling:

- Nine served compounds were never reported by the paper, and their replicate reliability is
  near or below zero (sodium glyoxylate -0.91, sodium butyrate -0.43). They stay in the folds;
  `compare_models.py` reports every model over all 41 and over the 32 published compounds
  (`vanacloig_data.UNREPORTED_COMPOUNDS`).
- The loader normalizes by library size (CPM) where the paper used TMM. That leaves a
  compound-wide offset (crystal violet -2.8 log2 units) that a within-compound Spearman does
  not see; per-compound rank agreement with an edgeR reconstruction is 0.975 median.
- DMSO-delivered compounds carry the vehicle's own profile (r 0.58 to 0.67 with the DMSO
  column) and no DMSO in their environment. Which compounds those are is in the unmirrored
  Table S1.

## 2026.09.29 - Round 2 results: nested ridge on FCFP4 is the bar, and nothing simple beats it

`compare_models.py` over `results/ladder/ladder_r2_scores.csv` (slurm 3034) and
`results/factorized/r2_table` (slurm 3035). Centered Spearman, median over 123
compound-evaluations (41 compounds by fold seeds 0, 1, 2) for the ladder and 41 (fold seed 0)
for the table models. "vs ridge" is the paired mean difference on the same compounds with a
bootstrap 95% interval.

| model | median, all 41 | median, 32 published | vs ridge (95% CI) |
|---|---|---|---|
| ridge on FCFP4 counts, nested penalty | 0.303 | 0.343 | reference |
| kernel ridge, RBF on FCFP4 | 0.297 | 0.327 | +0.006 (-0.014, +0.027) |
| kNN over the mean of all 12 RBF kernels | 0.298 | 0.326 | +0.002 (-0.024, +0.031) |
| ladder's own nested pick, per fold | 0.269 | | -0.018 (-0.044, +0.009) |
| best pretrained embedding (kNN, ChemBERTa-2 MTR) | 0.291 | 0.312 | -0.022 (-0.059, +0.015) |
| gene table x compound MLP, raw FCFP4, seed ensemble | 0.287 | 0.301 | -0.003 (-0.052, +0.046) |
| the same on a 64-dim PCA of FCFP4 | 0.233 | | |

Every other embedding, kernel and table variant is below ridge, most with intervals clear of
zero (`results/compare_models.csv`). The ladder's per-fold pick loses to fixed ridge because
the inner leave-one-compound-out score over 32 compounds is itself noisy enough to pick a worse
model about as often as a better one.

What this says: with 28 to 32 training compounds the compound side is the limit, and a
2,048-dimensional count fingerprint under ridge shrinkage is as good a compound representation
as any pretrained embedding here. The gene side is not the limit: every gene is seen in
training. So the cell graph transformer, which acts on the gene side, is not expected to move
this number (hypothesis, being measured in round 2c, slurm 3038), and the lever is more
compounds (round 3, slurm 3040).

## 2026.09.29 - Rounds 3 to 5, the transformer verdict, and the input audits of the other stores

**Transformer, five folds (`r2_cgt`, slurm 3038).** Cell graph transformer as the gene
encoder of the factorized model, graph prior 1: centered median 0.061, paired -0.229 vs
ridge (CI -0.29 to -0.17, 5 wins of 41). With the identity skip 0.030. The round-1 arm with
the transformer unchanged (slurm 3016) scored 0.134, 0.002 and 0.043 on folds 0 to 2. The
identity-skip round-1 job (3030) was cancelled after fold 0 (0.040, partial).

**Other stores as auxiliary training (`r3`, `r3b`, slurm 3040, 3042).** Shared compound
encoder and gene table, per-source bias and adapter, every Vanacloig compound removed from
the other stores. Over 123 compound-evaluations Wildenhain (5,160 compounds) is -0.008 vs
ridge (CI -0.042, +0.026); its plain mean with ridge +0.013 (CI -0.013, +0.038). Hillenmeyer
and Hoepfner as sources are below that. Graph smoothness on the shared gene table (`r4`,
slurm 3046) is -0.02 to -0.05. A predicted profile in another store as a compound embedding
(`phenotypic_embeddings.py`) is +0.003 at best; inside Wildenhain, structure predicts the
profile's first principal direction at r 0.39 out of fold, the median direction at 0.11.

**Coverage, not model (`similarity_vs_score.py`, slurm 3043).** The ridge score of a held-out
compound tracks its nearest training compound's FCFP4 Tanimoto (Spearman +0.29, n 123,
p 0.001). Gamma-valerolactone (-0.20), methylglyoxal (-0.07) and MMS (-0.03) have ceilings
above 0.89 and no analog.

![](./assets/images/035-similarity_vs_score.svg)

**Input audits of the other stores** (report-only agents, issues carry the `dataset` label):
Wildenhain #504 (33 of 242 strains are essential genes served as haploid deletions; the
served release is the uncited 2016 Scientific Data extension; z is relative to the strain's
own median well), Hillenmeyer #505 (no strain background; HIS3, LYS2, MET15 dosage wrong on the
BY marker loci; 33 arrays where the matrix header and the key file disagree; minimal medium
served as plain SD; merged strains), Hoepfner pending. Vanacloig: #500 and #501.

**Round 6, launched (`train_hit.py`).** Where the molecule enters: readout only, the compound
attending over all gene nodes (a hit distribution), the hit distribution propagated over the
nine networks (the ego net around the molecule's targets), and the deleted gene attending
over the compound as one more perturbation token; each with 0 or 2 layers of message passing
(`r6_mix`, slurm 3053), and a sizing sweep over width, depth, heads, hops and temperature
(`r6_size`, slurm 3054).
