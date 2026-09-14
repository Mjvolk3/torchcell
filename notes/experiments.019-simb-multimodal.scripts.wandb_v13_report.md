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
