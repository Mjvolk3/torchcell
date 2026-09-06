---
id: g09wlo2p9ai9lgxhhee1c7s
title: Dango_string_version_sweep
desc: ''
updated: 1788466936842
created: 1788466936842
---

## 2026.09.03 - DANGO replication by STRING release, pulled from wandb

Script: `experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py`. The results note
[[experiments.005-kuzmin2018-tmi.results]] recorded the STRING 9.1 / 11.0 / 12.0 sweep by reading
values off wandb charts. This script pulls every run of `zhao-group/torchcell_005-kuzmin2018-tmi_dango`
through the API, keeps runs with at least 100 logged epochs (drops the one 9-epoch smoke test), and
records per run the maximum over epochs of `val/gene_interaction/Pearson` (the checkpoint-selection
rule; an upward-biased order statistic, so epochs logged sit next to it). Outputs:

- `experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv` (19 runs) and
  `dango_string_version_summary.csv` (mean, SD, SEM per release x schedule)
- `paper/nature-biotech/sections/tab-dango-string-versions.tex`
- the panel below; `--from-csv` re-renders offline from the frozen run table.

Measured: best validation Pearson across the 19 runs spans 0.415 to 0.427; per release x schedule
means are 0.419 to 0.424 with SEM at most 0.003 where n > 1. No release or schedule separates from
the others beyond the run-to-run spread. These are validation maxima, not the test-split baseline of
the main text.

![](./assets/images/005-kuzmin2018-tmi/dango_string_version_sweep.svg)

## 2026.09.03 - Training curves, train Pearson at the selected epoch, and a fresh pull

Re-pulled for the Supplementary Note `note:dango-repro` (figure composed by
[[experiments.005-kuzmin2018-tmi.scripts.compose_dango_si_figures]]). The project holds 21 runs; the
two dropped are smoke tests of 9 and 6 epochs (`2yq5dedk`, `ikh3sj5f`), so the 19 kept runs and their
best-validation values are unchanged from the first pull. Additions:

- `results/dango_string_version_curves.csv` freezes the per-epoch history of every kept run (train and
  validation Pearson and MSE, validation reconstruction and interaction loss, `alpha`, learning rate;
  10,514 run-epochs). The trainer never called `trainer.test`, so no test-split metric exists for
  these runs.
- The run table gains `train_pearson_at_best` (training Pearson at the epoch of the validation
  maximum), `final_train_pearson`, `params_total` (3,138,270 in every run, consistent with the
  6,607-gene vocabulary: `454 N + 138,692`), batch size and learning rate; the LaTeX table gains the
  train-at-best column.
- The curves panel below (full width): train (left) and validation (right) Pearson per epoch, color
  by release, line style by schedule.

Measured: training Pearson at the selected epoch spans 0.488 to 0.553; validation Pearson reaches 0.4
within the first 50 epochs, peaks between epochs 106 and 439, and then declines, most steeply for
pretrain-then-main, to 0.32 at epoch 1,000 in the two runs that went that far (`ytkjmgvs`, `g34rn9ti`),
while training Pearson keeps rising to 0.55 to 0.74 at the last logged epoch. The plateau is therefore a
generalization limit rather than a failure to optimize.

Found while writing the note: `dango.py` takes `lambda_values` from `determine_lambda_values()`, whose
keys are `string9_1_<channel>`, and `DangoLoss.compute_reconstruction_loss` looks each run's edge type
up with `.get(edge_type, 1.0)`. The v11.0 and v12.0 runs therefore trained with `lambda_k = 1.0` for
every channel, not with the 0.1/1.0 assignment; only the v9.1 runs used it (code as of commit
`af2406523`, the version the runs were launched from). Whether this matters is not measured.

![](./assets/images/005-kuzmin2018-tmi/dango_string_version_curves.svg)

## 2026.09.03 - Table caption carries the run hyperparameters

The `tab-dango-string-versions` caption now states the optimizer (AdamW, learning rate 1e-5, weight decay
1e-6, batch 32), hidden width 64, four attention heads, and the 72,841 / 9,105 / 9,104 split, which left
the Note prose during the SI reconciliation. Regenerated with `--from-csv`; every value in the table is
unchanged.

## 2026.09.04 - Panels re-lettered; the frozen run table feeds the full-dataset data-effect panel

No change to the script or its outputs. In `FigS-dango-reproduction` the sweep panel is now (d)
and the curves panel (e), after the STRING-release panel (a) and the schematic (b).
`experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si.py` reads
`results/dango_string_version_sweep.csv` (19 runs) for its data-effect panel, pooling the three
schedules per release: v9.1 n = 4, mean 0.4216 +/- 0.0009 (SEM); v11.0 n = 5, 0.4225 +/- 0.0013;
v12.0 n = 10, 0.4213 +/- 0.0011 ([[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_si]]).

## 2026.09.05 - What wandb holds beyond the Pearson curves; panel e as 2 x 3 small multiples

Author review asked for more of the logged record in the figure. Listed with `run.scan_history()` on
`ytkjmgvs`, `q67k56m4`, `jpckzn4x` (every kept run was launched from the same code):

- per epoch, both splits: `gene_interaction/{Pearson,MSE,RMSE}` and `transformed/` duplicates,
  `loss`, `reconstruction_loss`, `interaction_loss`, `weighted_{reconstruction,interaction}_loss`,
  `integrated_embeddings_norm`, `val/alpha`;
- per step: the same training losses, `train/alpha`, `learning_rate` (constant 1e-5 in every run:
  the ReduceLROnPlateau never fired);
- every other epoch on a subsample: `{train,val}_sample/{MAE,MSE,Pearson,Spearman,JS_div,Wasserstein}_target_0`
  and image panels (`correlations`, `distribution`, `gene_interaction_box_plot`,
  `oversmoothing_integrated_embeddings`), stored as `media/images/...png` files on the run;
- summary only: `model/params_{total,pretrain_model,meta_embedding,hyper_sagnn}`.
- Not logged: gradient norms, per-channel embedding norms, any test-split metric.

The frozen `dango_string_version_curves.csv` now carries, per run x epoch, the eight validation
keys (Pearson, MSE, loss, reconstruction, interaction, both weighted losses, embedding norm), the two
epoch-end training metrics (Pearson, MSE), and the per-step training keys averaged over the epoch
(loss, reconstruction, interaction, both weighted losses, embedding norm) with `alpha` and the
learning rate taken as the last value in the epoch. The pull scans each key group once per run
(`epoch_frame`); `scan_history(keys=...)` does not drop rows for a key the run never logged, so a
group missing from a run comes back empty (the two smoke tests lack the weighted keys and are
skipped as before; a kept run missing a group raises).

Panel e is now a full-width 2 x 3 (179 x 56 mm). Top row on a log epoch axis (epoch + 1): the
pretraining weight `alpha_e` per schedule (drawn once per schedule after checking every run of a
schedule logged the same values), validation reconstruction loss, validation interaction loss; the
schedule legend sits in the alpha subplot and the release legend in the interaction-loss subplot.
Bottom row, linear epochs: train Pearson, validation Pearson, validation MSE with the label variance
over all 91,050 records (SD 0.054 from `dango_dataset_split.csv`, variance 0.00286) as a dashed
line. Right margin 0.985 so the last `1000` tick label is inside the canvas (it was clipped in the
exported PDF); the sweep panel (d) is 46 mm tall so the figure stays under 170 mm.

Measured from the frozen curves (per run, `results/dango_string_version_curves.csv`):

- Pretrain-then-main holds the validation interaction loss at its epoch-0 value (0.00246 to
  0.00253; the v12.0 runs drift up to 0.0028 to 0.0037) through epoch 9, then it drops to 0.0012 to
  0.0014 by epoch 20 like every other run. After alpha reaches 0, the reconstruction loss rises in
  every pretrain-then-main run (epoch 10 to last: 0.0074 to 0.0124, 0.0073 to 0.0133, 0.0074 to
  0.0127, 0.0068 to 0.0122 for v12.0; 0.0015 to 0.0061 for `ytkjmgvs`), and more slowly in every
  linear-to-flipped run (0.0039 to 0.0057, 0.0039 to 0.0060, 0.0065 to 0.0096, 0.0017 to 0.0026),
  while every linear-to-uniform run keeps it falling (0.0062 to 0.0044 at v12.0).
- Reconstruction loss level by release: v9.1 0.0009 to 0.0061, v11.0 0.0027 to 0.0154, v12.0 0.0043
  to 0.0133; v9.1 is lowest throughout, v11.0 and v12.0 overlap. v9.1 is both the release with the
  fewest edges and the only one trained with lambda = 0.1 on three channels (the fall-through above);
  which of the two sets the level is not measured.
- Validation MSE bottoms at 0.00233 to 0.00240 and crosses the label variance (0.00286) at epoch
  669 (`ytkjmgvs`) and 771 (`g34rn9ti`), the two runs that reach 1,000 epochs; no 500-epoch run
  crosses it.

Re-pulled from wandb (three passes; the project still holds 21 runs, the same two smoke tests
skipped): every value in `dango_string_version_sweep.csv` and every previously frozen curve column
is identical to the committed version; `tab-dango-string-versions.tex` is unchanged. Two bugs of the
first batched pull, fixed before the freeze: a run killed mid-epoch leaves a last epoch carrying
per-step keys but no validation or epoch-end training metric, which inflated `epochs_logged` by one
(442 to 443 and so on for the five such runs) and left `final_train_pearson` empty; both series are
now `dropna()`'d before use, restoring the committed definitions.
