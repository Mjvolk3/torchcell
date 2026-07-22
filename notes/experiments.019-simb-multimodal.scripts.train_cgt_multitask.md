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
