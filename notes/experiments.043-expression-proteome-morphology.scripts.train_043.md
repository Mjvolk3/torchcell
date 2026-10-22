---
id: jq5rv3z3ysy36b7pqdry8xl
title: Train_043
desc: ''
updated: 1792641704089
created: 1792641704090
---

## 2026.10.21 - The 043 trainer

Source: `experiments/043-expression-proteome-morphology/scripts/train_043.py`; conf `experiments/043-expression-proteome-morphology/conf/train_043_default.yaml` with groups `arm/`, `loss/`, `weighting/`; test `tests/torchcell/trainers/test_043_train_synthetic.py` (10, CPU, no store). Model [[torchcell.models.cell_model_043]]; loss and standardization [[torchcell.losses.hl_gauss]]; split [[experiments.043-expression-proteome-morphology.scripts.global_split_043]].

**Stage `prepare`** (CPU, [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_prepare]]). Reads every record of the ten trainer datasets through `Raw043Dataset` with the route-A representation, assigns the role from `conf/splits/global_split_v1.json`, drops fold-0 test records before their labels are loaded (Sameith Dm is kept as split B), applies the target transform of the primer's `tab:output-scales` and writes `PreparedStore043`: record arrays, marks in CSR form (operator genes, then off-graph tokens), the distinct reference contexts (deduplicated by content hash), targets, and the nine gene-gene graphs as edge lists. Job 6123 (commit `402ab04c4`), 3 min 8 s, wrote `$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/train_043_prep/prep_v1_routeA/` (797 MB):

| dataset | train records | val records | split B records | median marks per record (max) | distinct contexts |
|---|---:|---:|---:|---|---:|
| Kemmeren 2014 | 1,202 | 140 | 0 | 1 (1) | 1 |
| Messner 2023 | 3,712 | 455 | 0 | 1 (1) | 1 |
| Zelezniak 2018 | 79 | 9 | 0 | 1 (1) | 1 |
| Caudal 2024 | 566 | 189 | 0 | 78 (352) | 755 |
| Muenzner 2024 protein | 476 | 159 | 0 | 79 (352) | 635 |
| Muenzner 2024 turnover | 33 | 11 | 0 | 79.5 (334) | 44 |
| Boocock 2025 pooled | 235 | 79 | 0 | 3,112 (3,844) | 1 |
| N'Guessan 2025 | 1,770 | 591 | 0 | 2,909 (4,416) | 1 |
| Vaishnav 2022 | 42,694 | 14,232 | 0 | 1 (1) | 1 |
| Sameith 2015 Dm | 0 | 0 | 72 | 2 (2) | 1 |

784 distinct contexts in all (S288C plus 783 isolate contexts shared between Caudal and Muenzner records of one isolate). Graph edges inside the 6,607-gene index: physical 139,463; regulatory 37,767 (1,869 outside the index); TFLink 200,801; STRING 12.0 neighborhood 146,713, fusion 11,787, co-occurrence 11,085, co-expression 996,199, experimental 822,094, database 72,617.

**Targets.** Kemmeren and Sameith as released; Messner $\log_2$(strain / record reference); Zelezniak strain minus record reference; Caudal $\log_2(\mathrm{TPM}+1)$; Muenzner protein and turnover $\log_2$ of positive values; Boocock and N'Guessan $\log_2(\mathrm{CPM}+1)$ from the record's `total_umis` with omitted keys as zero counts, and a gene never detected in the training rows is not a target; Vaishnav the released level. Then the per-dataset standardization on training rows (per-gene centering for the panel datasets). Under `joint_9` the fitted pooled spreads (transformed units) are Kemmeren 0.198, Messner 0.277, Zelezniak 0.454, Caudal 0.641, Muenzner 0.428, turnover 0.388, Boocock 2.596, N'Guessan 2.398, Vaishnav 3.736; 0.42% of the finite training z values fall outside $[-4, 4]$ (into the open edge bins); statistics sha256 `0d5df0f0...`.

**Stage `train`** (Hydra). Selects the arm, fits and pins the statistics, reads the four `alleles/<encoder>.h5` tables for exactly the keys the arm uses (`load_vector_table`; a missing key raises), builds the model, and trains with AdamW (lr $3 \times 10^{-4}$, 500 warmup steps, then constant), bf16 autocast on the card, gradient clipping at norm 10 (pre-clip norm logged as `train/grad_norm` and `train/grad_norm_clip_frac`). Batches are uniform over the training records; the loss is the equal (or record-count) weighted mean of the per-dataset masked means in the batch. Sameith Dm is read through Kemmeren's dataset token, bias and statistics (the same arrays).

**Logged keys** (W&B project `torchcell_043-expression-proteome-morphology`, group `<arm>_<loss>_<weighting>`, name `<group>_s<seed>`): `train/loss`, `train/<dataset>/loss`, `train/<type>/loss` (type = expression, protein, turnover, promoter), `val/<dataset>/{pearson_per_gene_median, spearman_median, pearson_per_strain_centered_median}`, `val/<type>/...` (mean over the type's datasets), `val/vaishnav/{pearson, spearman}`, `val/acc_within_one_bin` and `val/entropy_mean` under HL-Gauss, the same under `traineval/` on a fixed training subset (256 records per dataset), `val/doubles/{pearson_per_strain_centered_median, additive_model_..., additive_measured_...}` in released units centered on Kemmeren's training-mean profile, `setup/*` (projection, epochs, window), `time/epoch_seconds`, `time/epochs_per_hour`. The W&B config carries `split.version`, `split.store_sha256` and the registered window; the statistics JSON is written to the checkpoint directory and its sha256 into every checkpoint.

**Statistics.** Per-gene Pearson and Spearman over the strains of a dataset's evaluation set (genes with at least 3 measured strains), median over genes; centered per-strain Pearson over genes after subtracting the training-mean profile in z, median over strains. Validation is capped at 2,000 records per dataset (a fixed seeded subset; binds on Vaishnav only). Validation never selects an epoch; the doubles are split B, a test set, logged because the brief asked for them, and must not steer a choice.

**Checkpoints.** `$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints/<run>/last.pt` every epoch and `epoch_<n>.pt` for every epoch inside the registered window, plus `standardization.json`.

Commands:

```bash
python experiments/043-expression-proteome-morphology/scripts/train_043.py --stage prepare --route A --workers 30
python experiments/043-expression-proteome-morphology/scripts/train_043.py arm=joint_9 loss=hlgauss weighting=equal seed=0
python experiments/043-expression-proteome-morphology/scripts/train_043.py arm=alone_kemmeren loss=hlgauss seed=0
```
