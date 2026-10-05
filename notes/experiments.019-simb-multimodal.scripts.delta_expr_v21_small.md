---
id: vg4nz5wfjaos2635zrfmglk
title: Delta_expr_v21_small
desc: ''
updated: 1791240986946
created: 1791240986946
---

## 2026.10.05 - The small-trunk single-change round on Delta, one A40 per job

Track B wave 1 of the Figure 3 plan (notes-tex/figure-3-gate, section "The plan to the
deadline") on the trunk of `conf/cgt_expr_v21_small.yaml`: seven arms times twelve split
seeds, 84 runs, packed four to a card as 21 independent single-GPU jobs. Independent means
one `sbatch --array=<k>` per pack, its own job id and submit time, no `--dependency`, no
array throttle, so no pack loses age priority to anything queued after it (the cabbi lesson
of 2026-10-01, when a chained array sat behind a later 15-day job).

Packing is four because the GilaHyper benchmark of the same day measured 8 to 10 GB of card
memory per run at every trunk shape ([[experiments.019-simb-multimodal.scripts.gh_small_model_bench]]).
Time per pack is a hypothesis until the canary logs `perf/epoch_seconds`: at the zero-worker
rate measured on both machines (78 to 134 s per epoch per run), 1,200 epochs take 27 to 45 h.

The stage `canary` runs PACK copies of S_ref for three epochs with a test pass; it is what
proves the store, the embedding file, the config and the card memory on Delta before 84
runs are queued. The stage `round` refuses to run while `V21_BENCH_PENDING` is 1, which is
flipped in the commit that fixes the trunk shape. `V21_LIST=1` prints the packs without
submitting. The user submits (Delta is Duo-gated); the account is the user's call.

Prerequisites on Delta, none of which this launcher can create: the job-owned clone pulled
to a commit carrying this file, the fig3_proteome store under `/scratch/bbub/mjvolk3/torchcell`
(`sync_delta_store.sh fig3_proteome` from GilaHyper, 14 GB, one Duo prompt), and the log
directory `/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21`.
