---
id: ydevzzcgpchzh27w051z0g0
title: Dango_full_dataset_inference
desc: ''
updated: 1792992828122
created: 1792992828122
---

## 2026.10.26 - DANGO checkpoints scored on the experiment-006 splits

Script: `experiments/010-kuzmin-tmi/scripts/dango_full_dataset_inference.py`, run once on 2026-10-26 on GilaHyper CPU (`--device cpu`, `OMP_NUM_THREADS=32`; all four GPUs were in use by other jobs, 9.3 to 10.6 GiB and 67 to 84% utilization each). It fills panels e, f, g of `FigS-dango-full-dataset` (SI note `note:dango-full`, `paper/nature-biotech/sections/si-note-dango-full.tex`). The 006 LMDB is not read; records, gene membership, and the seed-42 split come from the build's side files (332,313 records; train 265,851, val 33,231, test 33,231), and the gene index space is `sorted(genome.gene_set)` (6,607 genes), the recipe of [[experiments.010-kuzmin-tmi.scripts.score_010_checkpoints_directly]].

### Verification against the logged validation metrics

The recorded values are the DDP-synced torchmetrics values the training run logged at the checkpoint's epoch (`results/dango_full_dataset_history.csv`). The script asserts |delta Pearson| < 0.003 and |relative delta MSE| < 1%; all three pass with the genome index space, first try, so no alternative gene set was needed.

| run | ckpt | epoch | val Pearson (ours) | val Pearson (logged) | val MSE (ours) | val MSE (logged) | val Spearman |
|---|---|---|---|---|---|---|---|
| 014mprap (v9.1) | minmse | 81 | 0.367222 | 0.367252 | 0.00323466 | 0.00323457 | 0.324510 |
| 014mprap (v9.1) | last | 999 | 0.297349 | 0.297377 | 0.00376077 | 0.00376078 | 0.340700 |
| 9jpfy547 (v12.0) | minmse | 53 | 0.354321 | 0.354353 | 0.00327015 | 0.00327005 | 0.324033 |

Test split (n = 33,231; no logged test value exists, so these are not checked against a recording; same code path as val): 014mprap minmse Pearson 0.374661, Spearman 0.333692, MSE 0.003224; 014mprap last 0.309994, 0.343597, 0.003633; 9jpfy547 minmse 0.358491, 0.326347, 0.003269.

Rebuilt STRING edge counts (with the self loops `to_cell_data` adds): v9.1 neighborhood 52,208, fusion 7,965, cooccurence 9,266, coexpression 320,295, experimental 226,057, database 40,093; v12.0 153,320, 18,394, 17,692, 1,002,806, 828,701, 79,224.

### ReZero ablation at inference (val, n = 33,231)

With `beta1 = beta2 = 0` both HyperSAGNN attention layers return their input, so the dynamic embedding equals the integrated (meta) embedding `e_g` and each gene scores `s(g) = w . (e_g - ReLU(W_s e_g + b_s))^2 + b`; the prediction is the mean of the three per-gene scores, with no term that depends on two genes jointly. The script asserts this: scoring each gene as a singleton set and averaging over a record's three genes reproduces the beta = 0 prediction to a max absolute gap of 8.7e-9 (minmse checkpoints) and 8.8e-8 (014mprap last). Residual ratio = `||beta_l O_l(attn)||_F / ||x_l||_F` over each record's 3 x 64 block, `x_l` the layer input, read by forward hooks.

| run | ckpt | beta1 / beta2 | Pearson trained | Pearson beta = 0 | MSE trained | MSE beta = 0 | Pearson(pred trained, pred beta = 0) | residual ratio L1 mean (median) | residual ratio L2 mean (median) |
|---|---|---|---|---|---|---|---|---|---|
| 014mprap | minmse | 0.0732 / 0.0649 | 0.3672 | 0.0791 | 0.003235 | 0.003920 | 0.2597 | 0.415 (0.383) | 0.334 (0.321) |
| 014mprap | last | 0.1710 / 0.2014 | 0.2973 | -0.0283 | 0.003761 | 0.011305 | -0.0708 | 0.672 (0.677) | 0.618 (0.599) |
| 9jpfy547 | minmse | 0.0921 / 0.0745 | 0.3543 | 0.3370 | 0.003270 | 0.003410 | 0.9629 | 0.393 (0.405) | 0.324 (0.345) |

Spearman, beta = 0: 0.1457 (014mprap minmse), 0.1119 (014mprap last), 0.3141 (9jpfy547 minmse). Arithmetic on the measured numbers: a residual ratio of 0.415 at beta1 = 0.0732 means the unscaled attention output `O_1(attn)` is about 5.7 times the norm of its input, so a small beta does not mean a small residual branch.

Hypothesis (untested): the v12.0 checkpoint's predictions are mostly carried by the additive per-gene term (its beta = 0 predictions correlate 0.963 with the full ones), while the v9.1 checkpoint depends on the attention branch for nearly all of its validation correlation. That reading is not proven by this ablation, because zeroing beta on a model trained with beta > 0 removes a branch the downstream layers were fit with; it does not measure what a model trained without that branch would reach.

### Meta-embedding channel weights

Per-gene softmax weights over the six channels, recomputed from `meta_embedding.attention_mlp` and asserted to reproduce `meta_embedding`'s output (atol 1e-5). They depend on the gene only. "val gene slots" weights each gene by the number of validation records that contain it (99,693 slots).

| run | genes | neighborhood | fusion | cooccurence | coexpression | experimental | database |
|---|---|---|---|---|---|---|---|
| 014mprap minmse (v9.1) | all 6,607 | 0.1673 | 0.1247 | 0.1660 | 0.1350 | 0.2770 | 0.1300 |
| 014mprap minmse (v9.1) | 4,036 perturbed | 0.1649 | 0.1232 | 0.1639 | 0.1353 | 0.2843 | 0.1285 |
| 014mprap minmse (v9.1) | val gene slots | 0.1661 | 0.1234 | 0.1642 | 0.1336 | 0.2842 | 0.1286 |
| 9jpfy547 minmse (v12.0) | all 6,607 | 0.1336 | 0.2607 | 0.1355 | 0.1868 | 0.1607 | 0.1227 |
| 9jpfy547 minmse (v12.0) | 4,036 perturbed | 0.1343 | 0.2677 | 0.1371 | 0.1762 | 0.1608 | 0.1239 |
| 9jpfy547 minmse (v12.0) | val gene slots | 0.1355 | 0.2559 | 0.1402 | 0.1766 | 0.1654 | 0.1263 |

Uniform would be 0.1667. The largest channel is experimental for v9.1 and fusion for v12.0.

### Not measured

- No model trained with beta fixed at 0 (or without HyperSAGNN attention); the ablation is inference-time only.
- Only three checkpoints were scored: 014mprap (minmse, last) and 9jpfy547 (minmse). The 9jpfy547 last checkpoint and the other DANGO runs of `results/dango_full_dataset_runs.csv` were not scored. Channel weights were extracted for the two minmse checkpoints only.
- Test-split metrics have no logged counterpart to check against.
- The DANGO val split here is the experiment-006 build; the CGT predictions (`cgt_predictions_*`) are on the experiment-010 build. The two `*_record_rows_*.npy` files index different builds' sorted record ids and cannot be joined record by record.

### Outputs (`experiments/010-kuzmin-tmi/results/`)

- `dango_full_dataset_inference_check.csv`, `dango_rezero_ablation.csv`, `dango_full_dataset_inference_summary.json`
- `dango_predictions_{014mprap_minmse,014mprap_last,9jpfy547_minmse}_{val,test}.npy`, plus `_val_beta0.npy` for each
- `dango_record_rows_{val,test}.npy` (rows into the sorted 006 record ids), `dango_labels_{val,test}.npy` (raw gene_interaction)
- `dango_meta_embedding_weights_{014mprap,9jpfy547}_minmse.csv` (gene, is_perturbed_in_006, n_val_records, six channel weights)
