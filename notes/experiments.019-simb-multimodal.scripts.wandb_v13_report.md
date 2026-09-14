---
id: kz69xzizelab8infxs7smq3
title: Wandb_v13_report
desc: ''
updated: 1789361807658
created: 1789361807658
---

## 2026.09.13 - Run names, grouping keys, and the comparison report for the v13 round

Offline runs sync under their directory name (`run_compute-0-1-2397311_<hash>`), so the arm was only in the tags. The script renames every run in `torchcell_019_expr_v13` to `<arm>_seed<k>` (for example `V_concat_s0_90_seed1`), writes four top-level config keys the UI groups and filters on (`arm`, `split`, `readout`, `partition`), and builds the report and a saved workspace view. Idempotent; rerun after every sync.

Report (22 blocks, 9 panel grids, 48 panels): headline validation Pearson by arm, by split, by readout and every run; one grid per split with both readouts (val Pearson, spread ratio, val loss); every validation metric by arm; the train side by arm; split 0 at 90/10 against 80/10/10 by partition. Grouped lines are the mean over seeds with the min-max band, x axis epoch, no smoothing.

<https://wandb.ai/zhao-group/torchcell_019_expr_v13/reports/v13-split-round:-H_ref-and-H_concat-on-four-partitions-and-a-90/10-fold--VmlldzoxNzkyNzE3OQ==>

Saved workspace view "v13 split round by arm" (grouped by `arm`, epoch on x):

<https://wandb.ai/zhao-group/torchcell_019_expr_v13?nw=ywphnc96tfh>

Round design: [[experiments.019-simb-multimodal.conf.cgt_expr_v13_split]].

## 2026.09.14 - The Charts view, populated and ranked

The workspace API refuses the personal default view ("Workspace API does not currently support user views"), so the script owns a saved view instead, `v13 split round by arm` (`nw=ywphnc96tfh`), and overwrites it in place on every run: six sections in rank order of importance, all open, every panel with epoch on the x axis, runs grouped by `arm`.

| rank | section | panels |
|---|---|---|
| 1 | headline: validation | val Pearson per feature, Spearman, per instance, spread ratio, val loss, val mean Pearson |
| 2 | train side, the generalization gap | train-eval Pearson, spread ratio, Spearman, per instance, loss, nmse |
| 3 | masked conditioning, revealed 0 / 10 / 100 / 1000 genes | val Pearson at k0 to k3, mask loss at k0 to k3 |
| 4 | error and calibration | val nmse, mse, coverage 50 and 80, PIT KS, train-eval mse |
| 5 | optimization | train loss, grad norm, clip fraction, mask loss k0 and k3, epoch seconds |
| 6 | bookkeeping | scored genes, revealed counts, global step |

The report is updated in place by title rather than recreated (a rerun once duplicated it; the duplicate was deleted). Pick the view from the dropdown on the project's Charts tab:

<https://wandb.ai/zhao-group/torchcell_019_expr_v13?nw=ywphnc96tfh>
