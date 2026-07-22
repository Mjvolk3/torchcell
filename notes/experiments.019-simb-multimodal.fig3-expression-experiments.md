---
id: 1s5o1u9a6fr5mcvmmtlxgo9
title: Fig3 Expression Experiments
desc: ''
updated: 1784755626513
created: 1784755626513
---

## W&B Links

- Project: <https://wandb.ai/zhao-group/torchcell_019-simb-multimodal_cgt_multitask>
- Prior overfit run (373 epochs / 5.4h, the run this sweep responds to):
  <https://wandb.ai/zhao-group/torchcell_019-simb-multimodal_cgt_multitask/runs/3rllp896>

Related:

- Harness: [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]]
- Grid generator: [[experiments.019-simb-multimodal.scripts.generate_expr_grid]]
- Fig-3 query/build: [[experiments.019-simb-multimodal.scripts.query_fig3]]
- Roadmap: `plan.simb-2026-multimodal-cgt.2026.07.21` (memory `simb2026-multimodal-cgt-plan`)

## 2026.07.22 - Fig-3 expression single-GPU sniff sweep

### Motivation - the overfit run + mean-collapse

The prior 4-GPU run (`runs/3rllp896`, ~373 epochs / 5.4h) OVERFIT the Fig-3 expression
head. The per-gene expression Pearson (across strains, per gene) peaked around **0.044**
early and then **collapsed to ~0** as training continued: the model regressed to the
**per-gene mean**. With only ~1.1k-1.5k train expression genotypes and a 180-d / 8-layer /
9-head (~5M param) transformer, the model is grossly **over-parameterized** for the signal,
so it minimizes MSE by predicting each gene's mean and abandons cross-strain structure.

This sweep is a **single-GPU, small-model, no-DDP** grid to sniff out whether ANY real
expression signal survives at small capacity, and which levers (capacity, target
standardization, dataset, regularization) protect it. It is a **~16 h / ~64 GPU-hour**
screen, not a final model.

### Two Pearson metrics - per-gene vs per-strain (and why they differ)

Vector heads now log **both** (single-process, so no DDP-sync subtlety), computed at epoch
end in **raw log2-ratio units** (normalized heads are inverted before the metric):

- **`{stage}/per_gene/pearson_per_gene`** - for EACH gene, correlate its predicted value
  vs actual **across strains**, then average over genes. This is the abstract's honest
  metric and asks "is each gene's up/down-regulation predicted across genotypes?" It is
  **destroyed by mean-collapse**: a model that always emits a gene's train mean has zero
  cross-strain variance -> r = 0.
- **`{stage}/per_gene/pearson_per_strain`** - for EACH strain (row), correlate its
  predicted vector vs actual **across the ~6127 measured genes**, then average over
  strains. It asks "is the shape of this strain's expression profile predicted?" It stays
  **high even under mean-collapse**, because a single strain's profile is dominated by the
  shared per-gene mean structure that every strain roughly follows.

**The gap is the diagnostic.** In a smoke run the z-scored-target config showed
`pearson_per_strain ~= 0.41` while `pearson_per_gene ~= 0.03` - the classic mean-collapse
signature (high per-strain, ~0 per-gene). We want configs that lift **per-gene** without
merely inflating per-strain.

### Anti-mean-collapse lever - per-gene target standardization

New config knob `multitask.standardize_per_feature_target: [per_gene]`. It **z-scores each
gene** across the TRAIN-split strains (mean/std fit on train ONLY, stored as buffers,
inverted for raw-unit metric reporting). The model then predicts **deviations from the
per-gene mean** rather than absolute log2-ratios, so the "predict the mean" solution scores
~0 on the standardized MSE instead of winning it by default. Reuses the existing
train-only per-feature normalization machinery (WS10b/Part A), with method forced to plain
z-score per head. Selectable per head; the grid crosses raw vs standardized.

### Training-path changes (this session)

In `train_cgt_multitask.py`:

1. **EarlyStopping** callback (config `trainer.early_stopping`, default monitor `val/loss`,
   mode min, patience 20) alongside the checkpoints - cuts the marathon that overfit.
2. **`pearson_per_strain`** metric added; the prior `pearson` renamed to `pearson_per_gene`.
3. **`standardize_per_feature_target`** per-gene z-score lever (above).
4. **Single-GPU clean path**: when `trainer.devices == 1` the strategy is forced to `auto`
   (no 1-rank DDP), plain `python` launch.
5. **Config-driven `seed`** + `seed_everything(seed, workers=True)` so seed replicates are
   reproducible (model init + the CellDataModule split).
6. **Row-level dataset restriction** `cell_dataset.restrict_dataset_names` (see below).

### Kemmeren <-> Sameith cross-platform mean-merge confound

The `fig3_core` build fuses Kemmeren + Sameith(Sm/Dm) expression, all under the SAME
phenotype type `expression_log2_ratio`, so there is no per-head phenotype switch to separate
them. Restriction is done at the **row level** via `cell_dataset.restrict_dataset_names`,
which intersects each split with `dataset.dataset_name_index` (exact `dataset_name` keys).

- **Kemmeren-only** = the exact key `MicroarrayKemmeren2014Dataset` (1400 rows;
  ~1126 train / 132 val / 142 test after the fig3_core split).
- **Kemmeren+Sameith** = Kemmeren + `DmMicroarraySameith2015Dataset` (72) +
  `MicroarrayKemmeren2014Dataset+SmMicroarraySameith2015Dataset` (82) = 1554 rows.

The `...Kemmeren...+SmMicroarraySameith...` key is the **confound**: those 82 genotypes were
measured on BOTH platforms and **mean-merged** by `MeanExperimentDeduplicator`, averaging two
different microarray platforms into one target. It is a **separate** key, so Kemmeren-only
**excludes** it (a clean single-platform target) and +Sameith **includes** it. Comparing the
two conditions isolates whether adding Sameith (and the cross-platform average) helps or
hurts the honest per-gene metric.

### Grid design + manifest

Generator: `experiments/019-simb-multimodal/scripts/generate_expr_grid.py` ->
`gh_expr_grid_<NNN>.yaml` (000-287) + `experiments/019-simb-multimodal/results/grid_manifest.json`.

PRIMARY levers, FULLY crossed (5 factors, 48 combos):

- `hidden_channels` {16, 32, 64}
- `num_transformer_layers` {2, 3}
- target: raw vs per-gene-standardized
- dataset: Kemmeren-only vs Kemmeren+Sameith
- `graph_reg_lambda` {0.0, 0.001}

SECONDARY hyperparameter profiles (2, bundled - a screen, not a clean factorial):

- **baseline**: lr 3e-4, dropout 0.1, weight_decay 1e-8
- **aggressive**: lr 1e-3, dropout 0.0, weight_decay 1e-4

SEEDS: 3 per (primary x secondary) combo.

**Total = 48 x 2 x 3 = 288 runs** (array `0-287%4`). All runs: `devices=1`, `strategy=auto`,
`precision=bf16-mixed`, `max_epochs=100`, EarlyStopping patience 20, `batch_size=32`,
`active_heads=[per_gene]`, learnable_embedding on, both Pearson metrics. Attention heads
fixed at 4 (divides 16/32/64); the warmup-restarts LR scheduler is disabled
(`lr_scheduler=null`) so `optimizer.lr` is the honest swept LR (the scheduler otherwise
drives the effective LR from `max_lr`). Every knob + seed is recorded per array index in
`grid_manifest.json`.

This is deliberately **over-provisioned** to fill the ~16 h overnight window on 4 GPUs
(~64 GPU-hours). Each small-model run is expected to finish in minutes-to-~1 h (early stop

- tiny models), and the `%4` throttle keeps exactly 4 GPUs busy. If the array drains early,
that is simply more replicate data.

### Launch

```bash
# regenerate configs + manifest (from repo root):
python experiments/019-simb-multimodal/scripts/generate_expr_grid.py
# launch the single-GPU job array (4 concurrent):
sbatch --array=0-287%4 experiments/019-simb-multimodal/scripts/gh_cgt_multitask_array.slurm
```

### Verification (this session, no cluster run)

- Dry-run builds (`gh_expr_grid_000 dry_run=true`): OK, 10,946-param model.
- `fast_dev_run` on the smallest config (hidden 16, Kemmeren-only): restriction
  4074->1126 train; logs both `val/per_gene/pearson_per_gene` and
  `.../pearson_per_strain`.
- 2-epoch limited-batch run: `[early-stopping] monitor=val/loss patience=20` printed and
  `Metric val/loss improved` fired -> EarlyStopping wired.
- Std config (`gh_expr_grid_024`): `[Part A] 'per_gene' norm=zscore: 6127 features,
  n_train=1126` -> per-gene z-score path works; smoke metrics showed the mean-collapse
  gap (`pearson_per_strain ~= 0.41` vs `pearson_per_gene ~= 0.03`).
- `bash -n` on the array slurm: OK. mypy + ruff clean on both changed `.py` files.

### Results

_To fill as runs land._ Read `grid_manifest.json` to join each array index -> knobs, then
pull `val/per_gene/pearson_per_gene` (honest metric) and `pearson_per_strain` from W&B.
Questions to answer:

- Does ANY config lift **per-gene** val Pearson meaningfully above the ~0.044 prior peak
  (and hold it, rather than collapsing)?
- Does per-gene **target standardization** protect per-gene Pearson vs raw targets?
- Smallest (hidden 16, 2-layer) vs larger (64, 3-layer): where does over-fitting start?
- Kemmeren-only vs +Sameith: does the cross-platform mean-merge help or hurt per-gene r?
- `graph_reg_lambda` 0 vs 0.001, and baseline vs aggressive hyperparameters.
- Seed spread: is any apparent lift within seed noise?
