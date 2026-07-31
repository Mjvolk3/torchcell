---
id: 9n2csfwzvias0z1l67tkrqb
title: Wave4b_convergence
desc: ''
updated: 1785531958854
created: 1785531958854
---

## 2026.07.31 - Asking whether a run had STOPPED IMPROVING before believing its score

`score_decoder_arms.py` answers "which arm scored higher" by reducing each run to one
number; it cannot answer "was the run finished when we scored it", and on the expression
task that second question dominates the first. Every architecture delta measured in 019 so
far is <= 0.03 and was taken at an epoch cap, so if the budget truncates arm-dependently the
ranking is partly a ranking of who got cut off later. This script exists to test that
directly, and applied to wave 4b it voided the wave's headline comparison.

```bash
python experiments/019-simb-multimodal/scripts/wave4b_convergence.py \
    --stage-tag stage-wave4b --window 5 --tail 100
```

- **What it emits per run** (`results/wave_convergence_stage-wave4b.csv`, one row per run;
  `results/wave_history_stage-wave4b.csv`, tidy per-epoch): `roll_max` / `roll_argmax` /
  `roll_end` off a centered rolling mean (`--window 5`), plus an OLS slope over the last
  `--tail 100` epochs with its SE and t, reported per 1000 epochs so it is readable next to
  a 300-epoch budget. The read is: `roll_argmax` near the final epoch and
  `roll_end ~= roll_max` means the run was still climbing when it stopped.
- **Wave 4b verdict -- both arms were still rising at the cap.** All 8 runs stopped at
  exactly 300 epochs. Three of four seeds in each arm put `roll_argmax` in the last 20% of
  the run (241-297), and the train side is unambiguous: `train/expression/pearson_per_feature`
  had a positive tail slope in **all 8** runs, +0.505 to +0.675 per 1000 epochs with
  t = 6.70 to 30.86. Terminal `train/grad_norm` was 0.042-0.056, nowhere near the clip. The
  optimizer still had somewhere to go in every single run.

  | arm | seed | roll_max | roll_argmax | roll_end | val_slope_k (t) | train_pf_slope_k (t) |
  |---|---|---|---|---|---|---|
  | N_sink_mm | 0 | 0.1520 | 86 | 0.1283 | +0.331 (16.92) | +0.624 (30.26) |
  | N_sink_mm | 1 | 0.0474 | 297 | 0.0474 | +0.171 (17.12) | +0.675 (16.53) |
  | N_sink_mm | 2 | 0.0863 | 241 | 0.0568 | -0.106 (-4.61) | +0.505 (7.34) |
  | N_sink_mm | 42 | 0.0742 | 296 | 0.0723 | +0.054 (3.00) | +0.654 (25.12) |
  | R_ref | 0 | 0.1531 | 89 | 0.0943 | +0.023 (0.84) | +0.670 (30.86) |
  | R_ref | 1 | 0.0555 | 292 | 0.0513 | +0.233 (15.78) | +0.519 (13.24) |
  | R_ref | 2 | 0.0798 | 252 | 0.0610 | +0.071 (4.02) | +0.651 (21.49) |
  | R_ref | 42 | 0.0617 | 296 | 0.0564 | +0.249 (23.99) | +0.665 (6.70) |

- **Val loss bottoms long before val Pearson peaks**, so any checkpoint selected on loss is
  the wrong model. From `wave_history_stage-wave4b.csv`: R_ref seed 1 has its
  `val/expression/loss` minimum at epoch 42 and its `val/mean/pearson_per_feature` maximum at
  epoch 293 (gap 251); N_sink seed 1 is 53 vs 257 (gap 204); seed 42 is 202 vs 297 and 217 vs
  296. Only the seeds whose Pearson peaked early (0 and 2) agree. This is the wave-4b
  instance of the round's checkpointing finding.
- **The paired null-sink delta is a NON-MEASUREMENT, not a null.** On `roll_max`,
  N_sink_mm - R_ref within seed is `{0: -0.0011, 1: -0.0081, 2: +0.0064, 42: +0.0125}` --
  mean **+0.0024**, sd 0.0090 (ddof=1), n=4, 95% CI +/- 0.0143, so the CI contains 0. On
  `roll_end` it is +0.0105, sd 0.0183, CI +/- 0.0292 -- also containing 0. Both arms were cut
  at the same 300-epoch cap while still improving, so this compares two unfinished runs; it
  does not license "the null sink did not help".
- **Series are aligned on the logged `epoch`, never on row position, and pulled in ONE
  unfiltered `scan_history()`.** Both choices are load-bearing failure fixes.
  `train/expression/loss` logs per optimizer step (1171 rows for 300 epochs, 1200 on seed 1 =
  3.90 rows/epoch) while the val metrics, train Pearson, `grad_norm` and the gate log once per
  epoch. The first version stacked per-key `scan_history(keys=[k])` results by row index, so a
  column labelled `epoch` held a step index for that one series, "the last 100 epochs" was
  really the last ~25.6, and the endpoints were single-batch samples. Separately, a multi-key
  `history(keys=[...])` silently returns ZERO rows when any key is not co-logged in the same
  step -- which is why the train side went unread in this project for several rounds. Fix:
  pull the full history once and `groupby("epoch").mean()`, which is the epoch-mean for a
  step-level series and the identity for a per-epoch one.
- **Train loss is reported as a fitted TREND, not a two-point difference.** The first version
  computed `100*(tl[-tail] - tl[-1])/|tl[-tail]|` and produced +/-15% swings that were pure
  sampling noise -- `train/expression/loss` oscillates roughly 0.22-0.28 epoch to epoch, so
  any endpoint difference is dominated by which two epochs got sampled, and two seeds read as
  "loss increased 15%" from that alone. The slope with its t against residual scatter says
  something real but small: -0.031 to -0.110 per 1000 epochs across the 8 runs, |t| 0.67-3.74.
- **Key names are asserted, not guessed.** Every key in `SERIES_KEYS` was verified present on
  run `047tg3h4`, and a missing key raises rather than yielding an empty column that would
  read as "the mechanism did nothing". The one exception is
  `gate/perturbation_transform.null_bias`, which exists only on the null-sink arm; there it
  moved from 0.003-0.006 at the first epoch to 0.077-0.089 at the last.
