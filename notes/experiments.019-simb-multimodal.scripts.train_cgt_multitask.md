---
id: pw5vim74rhfhtemexxgtrrk
title: Train_cgt_multitask
desc: ''
updated: 1784699671781
created: 1784699671781
---

## 2026.07.22 - WS-RUN: GilaHyper training setup (Part A norm, Part B metric, Part C configs + sweep)

Script: `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`

### Part A - Normalization decision (supersedes WS10b z-score-only)

- **Drop 3 degenerate CalMorph features** from the `global` (morphology) target AND the
  head `output_dim`: **281 -> 278**. Dropped list (config `multitask.drop_features.global`):
  `A113_A1B`, `A113_C`, `C123_C`. These are a subset of the 6 robust-CV-flagged
  near-constant features (`A113_A, A113_A1B, A113_C, C123_C, D203, D205`); the author chose
  to hard-drop these 3 (the remaining flagged features stay, floored by the standardizer).
  Implemented via a `keep_mask` over the key-sorted CalMorph vector in
  `build_head_alignments` (analogue of the `per_gene` measured-gene mask), so the decoded
  target is restricted to the 278 kept features and a runtime check asserts
  `heads.global.output_dim == 278`.
- **Per-feature Yeo-Johnson power transform + z-score** replaces plain z-score. Fit with
  `sklearn.preprocessing.PowerTransformer(method="yeo-johnson", standardize=True)` on the
  **TRAIN split only** (in `compute_per_feature_target_stats`). This **realizes Ohya 2005
  SI's published "Box-Cox then standardize"**; we use **Yeo-Johnson** (the zero/negative-safe
  generalization of Box-Cox) because CalMorph features contain zeros and negatives, so strict
  Box-Cox is undefined. The fitted params are stored as checkpointed buffers -- per-feature
  `lambda` + the transformed-space `mean`/`std` -- and re-implemented in torch
  (`_yeo_johnson_forward` / `_yeo_johnson_inverse`) so normalization runs on-device and is
  **invertible** for raw-unit reporting. Verified the torch transform matches sklearn to
  ~3e-4 (float32) and inverse round-trips to ~5e-5. Selectable via
  `multitask.vector_norm_method: {yeo_johnson (default) | zscore}` (`zscore` kept for
  ablation, e.g. sweep 006). Stats + lambdas + dropped list are written to
  `results/calmorph_train_target_norm_global.json`.

### Part B - Honest metric (per-feature-averaged Pearson)

Replaced the per-batch **flatten-Pearson** (a feature-scale artifact -- flattening a
multi-scale vector correlates across features of different magnitudes) with
**per-feature-averaged Pearson** (`per_feature_pearson`), computed at **EPOCH level** in
**ORIGINAL (inverse-transformed) units**:

- morphology (`global`): mean Pearson over the **278** CalMorph features;
- expression (`per_gene`): mean Pearson over the **6127** measured genes;
- fitness (`gene_interaction`): the single-feature reduction (== ordinary Pearson).

Supervised `(pred, target)` rows are cached per step in raw units (normalized heads are
inverted via `denormalize`; expression log2-ratios / fitness are already raw), concatenated
at epoch end, and reduced. Features with a (near-)constant prediction/target column over the
epoch are dropped from the average (undefined correlation), not counted as zero. Logged as
`{stage}/{head}/pearson` for train/val/test; under DDP the per-rank per-feature correlation
is `sync_dist`-averaged. This makes numbers comparable to the abstract's r values.

### Part C - GilaHyper full-Hydra configs + SLURM

W&B project (every config): **`torchcell_019-simb-multimodal_cgt_multitask`**.
Real model size for 4x GPU DDP: `hidden_channels=180, num_transformer_layers=8,
num_attention_heads=9, perturbation_head.num_heads=9`, `precision: bf16-mixed`, 600 epochs,
CosineAnnealingWarmupRestarts, batch_size 32, physical+regulatory graphs with graph-reg on
heads 0/1.

Main configs (`experiments/019-simb-multimodal/conf/`):

| Config | active_heads | Notes |
| --- | --- | --- |
| `gh_cgt_multitask_expr_000` | `[per_gene]` | expression-only baseline |
| `gh_cgt_multitask_morph_000` | `[global]` | morphology-only; Part A (278 + Yeo-Johnson) |
| `gh_cgt_multitask_joint_exprfit_000` | `[gene_interaction, per_gene]` | WS11a joint expr+fitness; ~1416 co-located genotypes (masked loss restricts each head; the intersection is where both are supervised) |
| `gh_cgt_multitask_joint_000` | `[gene_interaction, per_gene, global]` | full triple-head joint; **sweep base** |

Edge-of-config-space sweep (each inherits `gh_cgt_multitask_joint_000`, varies ONE knob):

| Config | Knob | Value (vs base) |
| --- | --- | --- |
| `gh_cgt_multitask_sweep_dmodel_small_001` | `d_model` | hidden_channels 90 (vs 180) |
| `gh_cgt_multitask_sweep_dmodel_large_002` | `d_model` | hidden_channels 360 (vs 180) |
| `gh_cgt_multitask_sweep_layers_deep_003` | `num_transformer_layers` | 12 (vs 8) |
| `gh_cgt_multitask_sweep_lr_high_004` | `learning_rate` | lr 3e-4 / max_lr 1e-3 (vs 1e-4 / 5e-4) |
| `gh_cgt_multitask_sweep_graphreg_off_005` | `graph_reg_lambda` | 0.0 (vs 0.001) |
| `gh_cgt_multitask_sweep_zscore_only_006` | normalization | zscore (Yeo-Johnson OFF) |

The **heads on/off** axis is spanned by the four main configs (1 head -> expr/morph, 2 ->
exprfit, 3 -> joint), so it is not duplicated as a sweep entry.

SLURM: `experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm` -- ONE parameterized
GilaHyper launcher (`#SBATCH -p main`, `--gres=gpu:4`, `torchrun --standalone --nproc_per_node=4`,
conda `torchcell` env, output `experiments/019-simb-multimodal/slurm/output/%x_%j.out`). Pass
the Hydra config name as `$1`. Launch (orchestrator runs these; do NOT sbatch from here):

```bash
sbatch -J gh_cgt_multitask_expr_000            experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_expr_000
sbatch -J gh_cgt_multitask_morph_000           experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_morph_000
sbatch -J gh_cgt_multitask_joint_exprfit_000   experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_joint_exprfit_000
sbatch -J gh_cgt_multitask_joint_000           experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_joint_000
sbatch -J gh_cgt_multitask_sweep_dmodel_small_001  experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_dmodel_small_001
sbatch -J gh_cgt_multitask_sweep_dmodel_large_002  experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_dmodel_large_002
sbatch -J gh_cgt_multitask_sweep_layers_deep_003   experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_layers_deep_003
sbatch -J gh_cgt_multitask_sweep_lr_high_004       experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_lr_high_004
sbatch -J gh_cgt_multitask_sweep_graphreg_off_005  experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_graphreg_off_005
sbatch -J gh_cgt_multitask_sweep_zscore_only_006   experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm gh_cgt_multitask_sweep_zscore_only_006
```

### Verification (no cluster run, no sbatch)

- `dry_run=true` constructs model + heads + masked loss for every config (expr/morph/
  exprfit/joint + all 6 sweeps + updated igb/delta); `global` head is 278-D.
- Yeo-Johnson torch transform matches sklearn (~3e-4) and inverse round-trips (~5e-5);
  `per_feature_pearson` returns ~1 on perfectly correlated synthetic features and drops
  constant columns.
- `bash -n` clean on all slurm; `mypy` + `ruff` clean on `train_cgt_multitask.py`.
- Stale sibling configs (`igb_cabbi/igb_mmli/delta`) updated off the broken `output_dim: 501`
  - `microarray_/rnaseq_expression` to the corrected 278 + `expression_log2_ratio` + Part A knobs.

## 2026.07.31 - Make an arm comparison readable before spending GPU on it: train-side saturation, metric-vs-loss selection, and a masked objective that cannot score its own input

The decoder-ladder round could not be *read* off the harness it started with: the only
train-side number was biased low, the only checkpoint was selected by a loss that moves
OPPOSITE the metric we report, throughput was inferred from job elapsed time, and the
teacher-forced masked-label objective did not exist. This section covers the measurement
apparatus (`traineval/`, `mse`/`nmse`, `perf/epoch_seconds`, best-by-metric checkpointing)
and `_masked_step`, the v9 objective -- both in
`experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`.

### The four measurement gaps, and what each one now answers

| Key | Question it makes answerable | Why the old surface could not |
| --- | --- | --- |
| `traineval/<pheno>/*` | Is the model saturating on its OWN train set (capacity limit) or only on val (generalization limit)? | `train/...` is accumulated over training batches -- dropout ACTIVE, weights still moving within the epoch -- so it is biased LOW. |
| `<stage>/<pheno>/nmse` | Is the magnitude right, not just the ordering? `1.0` is exactly "predict each gene's mean". | Pearson is scale-invariant; the quantile loss is dominated by predictive spread. Neither alone separates the two. |
| `perf/epoch_seconds` | What does one epoch actually cost, per arm? | Throughput was inferred as (job elapsed / epochs), which folds in dataset load + process startup; the bias is not a fixed offset, since setup is constant while epoch cost scales with batch size and GPU co-residency. |
| `{job_id}-best-metric-{epoch}.ckpt` | Which weights actually rank strains best? | 019 kept only best-by-loss, and on this task that is the early-dip model. |

- **`_train_eval_pass`** runs the train split in EVAL mode -- `self.eval()`, `@torch.no_grad`,
  fresh `_metric_cache["traineval"]`, restore `self.train()` if it was training. It runs
  inside `on_validation_epoch_end` (the module is already eval + no-grad there), costs one
  extra forward pass over train (~4 batches at the 019 expression size), and is gated by
  `trainer.train_eval_every` with default `0 = off`, so no pre-existing arm changes cost or
  output. Consumed downstream by `score_decoder_arms.py`.
- **`mse` / `nmse`** exist because loss and metric diverge on this task. As recorded in the
  code from the H1 long runs: `val/expression/loss` bottoms at epoch ~103-136 then rises to
  0.270-0.293 by epoch 1500, while val Pearson climbs the whole way (~0.13 -> 0.198) and
  `pred_sd_ratio` goes 0.001 -> 0.52. The quantile loss is a proper score penalising a model
  for committing to bolder predictions even as the ORDERING improves; a squared error on the
  point prediction has no such term.
- **Best-by-metric checkpointing** is the direct consequence: a second `ModelCheckpoint`
  monitoring `val/mean/pearson_per_feature` with `mode="max"` (overridable via
  `checkpoint.metric_monitor` / `metric_save_top_k`), kept ALONGSIDE best-by-loss because
  they answer different questions -- the loss checkpoint is the calibrated predictor, the
  metric checkpoint is the one that ranks strains correctly. The gap is not marginal: on the
  H1 runs val loss bottoms at ~103-136 while val Pearson peaks at ~1367-1508, and the round's
  wave-6 checkpoint audit put best-by-metric at epochs 933-1906 against best-by-loss at
  147-484 (~1400-epoch median gap). Every expression "best" checkpoint 019 had ever saved was
  the early-dip model.

### `_masked_step` -- the teacher-forced unmasking objective (v9)

Reveal `m` true gene values, predict the rest. The one requirement that makes the number mean
anything is that **the loss is restricted to still-hidden genes**: a revealed gene is model
INPUT at that step, so scoring it rewards copying input to output -- train loss collapses and
nothing transfers to validation, where nothing is revealed. `verify_masked_objective.py` ->
`results/verify_masked_objective.json` pins this as contract C3: `|grad| hidden=0.4941`,
`revealed=0.000e+00`.

- **One random `k` per batch in training, the full sweep at validation.** Running every step
  each optimizer step costs `K+1` forwards (~5x); sampling `k` uniformly is an unbiased
  estimator of the same `k`-averaged objective at 1x. Validation sweeps all `k` because that
  is where `pearson@k` is reported and it is paid once per epoch.
- **Nested observed sets.** One random key `scores` per row, reused across steps; taking the
  `n_reveal` smallest gives `M_1 subset M_2 subset ...`, so the validation sweep is an
  unmasking TRAJECTORY rather than K unrelated draws (contract C4). Only supervised rows may
  reveal (C7: revealed entries on unsupervised rows = 0) -- handing zeros to an unmeasured row
  would teach "unmeasured == 0". `_to_token_space` is the exact inverse of the `col_idx`
  gather (C6 round-trip exact), and the values teacher-forced in are in the SAME normalized
  space the head predicts in.
- **`val/mean/pearson_per_feature` is still published at `k=0`.** At `k=0` the observed set is
  empty, every encoded feature is zero, and the forward pass IS the unconditioned model, so
  `_cache_epoch_metric` is called there under the standard namespace. That is what the scorer
  reads and what the best-metric checkpoint monitors -- without it a v9 run cannot be compared
  to any v8 arm (and its absence is what crashed job 1439; `fast_dev_run` disables
  checkpointing, so the smoke test never hit it).
- **`train/` metrics are deliberately NOT cached on the masked path** (`_cache_masked_metric`
  is guarded by `stage != "train"`). A during-training `pearson@k` is computed over rows that
  were handed part of their own answer, i.e. exactly the quantity this objective can inflate
  for free; the honest train-side reading is the `traineval` pass, which runs the same full
  `k` sweep under its own stage-namespaced cache (`{stage}_mask_k{k}`) so `traineval` rows can
  never be reduced and logged as `val`.
- **Reducer is NaN-aware by necessity.** Which genes are hidden differs per STRAIN, so each
  feature's correlation runs over its own finite pairs; revealed entries are cached as NaN
  (not as their true value, which would inflate `r` with values the model was handed) and
  features with fewer than 3 pairs are dropped.
- **Schedule sizes are log-spaced against a measured rank, and `-1` is rejected.**
  `residual_covariance_diagnostic.json` puts the effective rank of the reproducible residual
  gene-gene structure at 32.78, so identifying it needs `|M| >~ 33`: 10 under-determines, 100
  over-determines ~3x, 1000 saturates. A step revealing ALL genes scores an empty set,
  contributes no gradient, and in training wastes the batch outright, so `mask_schedule`
  containing `-1` raises rather than silently averaging in a zero.

What `k>0` buys is an imputation capability, not a better genotype -> expression score. The
linear oracle in `results/masked_conditioning_oracle.json` reaches val mean per-gene
`r = 0.4084 (m=10)`, `0.6756 (m=100)`, `0.7932 (m=1000)`, and
`results/conditioning_gain_after_genotype.json` shows that gain is essentially ORTHOGONAL to
genotype -- retained fraction `0.975 / 0.992 / 1.006` after removing a genotype predictor.
Which is precisely why `k=0` has to stay the comparable number.

### Metabolism inherits the whole surface

`model.model_class` dispatches `CellGraphTransformerMetabolism` vs `CellGraphTransformer`
(anything else raises). Because that fork reuses the encoder and PERT operator unchanged, the
metabolism arms pick up masking, `traineval/`, `nmse`, `perf/epoch_seconds` and best-by-metric
checkpointing without a second harness -- `forward` simply threads `observed_values` /
`observed_mask` through to whichever model class was built.
