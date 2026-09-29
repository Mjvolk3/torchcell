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
