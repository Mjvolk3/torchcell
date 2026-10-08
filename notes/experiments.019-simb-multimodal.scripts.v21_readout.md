---
id: 2epvj4az1ask6fa2s3we0nh
title: V21_readout
desc: ''
updated: 1791417172951
created: 1791417172951
---

## 2026.10.07 - The v21 round read at completion

Reads W&B project `torchcell_019_expr_v21` (the Delta round: nine arms on the ProtT5-only small trunk, twelve split seeds, 1,200 epochs at batch 32 on the both-label store) and writes `results/v21_readout.json` and `notes-tex/figure-3-gate/tables/v21_round.tex`.

- Which run counts: the project holds two attempts of most (arm, split) pairs, the first launch of 2026.10.06 (killed inside its job by the in-place swap at 18:38, or by hand after collapsing) and the relaunch. Per pair the run with the most logged epochs is read; the number of attempts is recorded. The cohort of the chosen run is `original` (created before 23:38 UTC on 2026.10.06; only pack 0's `S_ref` splits 0 to 3) or `relaunched`.
- Score: the registered window, the mean of `val/<head>/pearson_per_feature` over epochs 1,000 to 1,199, defined only for runs that reached 1,199. `S_prot` is read on the proteome head and is not paired with `S_ref`.
- Collapse: the same rule as [[experiments.019-simb-multimodal.scripts.collapse_census]] (launch at spread 0.05, dead below 0.01, collapsed at 50 or more consecutive dead epochs). Per arm the window is summarized over all finished runs and over the runs that did not collapse; the paired contrast against `S_ref` is given over all shared finished splits and over the splits where neither run collapsed.
- The baseline bar in the table caption is the best validation mean per head in `results/baselines_both_label.json` (the neighbor average on ProtT5: 0.110 expression, 0.065 proteome).

Result on 2026.10.07 (all 104 relaunched runs and pack 0's splits 0 and 3 at 1,199; `S_ref` splits 1 and 2 are the pack-0 runs killed after collapsing at 1,102 and 939 epochs and were never relaunched):

| arm | finished | collapsed | window all | window healthy (n) | paired vs ref, healthy |
|---|---|---|---|---|---|
| S_ref | 10/12 | 3 | 0.063 ± 0.037 | 0.070 ± 0.032 (9) | |
| S_mask | 12/12 | 0 | 0.094 ± 0.028 | 0.094 ± 0.028 (12) | +0.027 (7/9) |
| S_sink | 12/12 | 6 | 0.052 ± 0.048 | 0.075 ± 0.030 (6) | +0.017 (5/5) |
| S_prop2 | 12/12 | 0 | 0.071 ± 0.044 | 0.071 ± 0.044 (12) | +0.010 (5/9) |
| S_nodrop | 12/12 | 5 | 0.046 ± 0.047 | 0.079 ± 0.033 (7) | +0.002 (2/5) |
| S_stack | 12/12 | 4 | 0.071 ± 0.046 | 0.096 ± 0.019 (8) | +0.043 (6/6) |
| S_basis64 | 12/12 | 3 | 0.055 ± 0.042 | 0.073 ± 0.032 (9) | +0.008 (5/7) |
| S_hadam | 12/12 | 10 | 0.007 ± 0.014 | 0.018 ± 0.001 (2) | -0.040 (0/2) |
| S_prot | 12/12 | 0 | 0.066 ± 0.021 | 0.066 ± 0.021 (12) | |

What it says: the mask schedule is the only expression arm with no collapse and the best mean over all twelve seeds (+0.033 over the reference on all 10 shared splits, 8 above zero); the composite stack is the best when it survives (+0.043 on 6 of 6 healthy pairs, train Pearson 0.65 against 0.52) but collapses on 4 of 12; the Hadamard operator collapses on 10 of 12 and its two survivors sit at 0.018, which settles the operator against it; the null sink, rank-64 basis, no-dropout and two-hop propagation arms are within the twelve-seed resolution (about 0.02) of the reference, and propagation puts 5 of 12 runs in a second low regime (spread 0.15 to 0.21, train Pearson 0.33 to 0.34, window -0.004 to 0.054) without collapsing. Every arm mean is below the twelve-seed neighbor-average baseline on this store (0.110); 12 of 94 finished expression runs are above it, the best at 0.146. The proteome head (0.066 ± 0.021) sits at its baseline (0.065).

Example runs (one per line):

<https://wandb.ai/zhao-group/torchcell_019_expr_v21>

S_ref split 6, the best reference run (window 0.116):

<https://wandb.ai/zhao-group/torchcell_019_expr_v21/runs/vigjy5gb>

S_mask split 6, the best mask run (window 0.137):

<https://wandb.ai/zhao-group/torchcell_019_expr_v21/runs/uga34iwx>

S_prop2 split 10, the low regime (window 0.001, spread 0.16):

<https://wandb.ai/zhao-group/torchcell_019_expr_v21/runs/lkubcb56>

S_stack split 7, collapsed at epoch 292:

<https://wandb.ai/zhao-group/torchcell_019_expr_v21/runs/x6gs6uyy>

S_hadam split 2, collapsed at epoch 653:

<https://wandb.ai/zhao-group/torchcell_019_expr_v21/runs/pwbk62tm>
