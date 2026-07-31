---
id: duobfkg3b4llsg9i4bkfz8h
title: Gh_expr_wave3
desc: ''
updated: 1785532024076
created: 1785532024076
---

## 2026.07.31 - Re-measuring Every Expression Arm After the Scorer Had Averaged Them Into the Baseline

Wave 3 exists because everything before it was scored with a broken instrument, so no ranking from those rounds could be trusted -- this launcher re-measures from scratch under one frozen source, one budget, one packing, and a within-seed reference co-launched beside the arms. It later carried waves 4a and 4b as extra stages, which is where hard graph masking was frozen and the null sink retested.

- **What was broken.** `score_decoder_arms.py` derived the arm from only `free_gene_dim` + `perturbation_propagation`, so `C0/D1/D2/E0/F1/G1/H0/GEARS/S*` all resolved to the literal string `A0_baseline` and were averaged INTO the baseline. On top of that, a source edit at 2026-07-28 22:26:23 split the runs into two code versions that were then pooled, and the per-arm epoch budget was set by early stopping firing on a metric that swings +/-0.05 epoch to epoch.

- **Stage order is load-bearing: `ladder` runs FIRST and alone.** The model never fits its own TRAINING set -- `train/expression/pearson_per_feature` ends at 0.19-0.246 and is still climbing at the cap, `train/expression/loss` falls 8% over 196 epochs, and `train/grad_norm` sits at 0.02-0.13 against a clip threshold of 10.0, so clipping never engages. Every architecture delta measured so far (<= 0.03) was taken on a model explaining ~5% of train variance, so running the architecture arms before settling the optimizer would measure all of them in that starved regime.

- **Two runs per GPU, uniformly, for every run in a comparison set.** A single run uses 20,026 of 46,068 MiB and does not saturate an Ada 6000 -- the encoder runs at BATCH 1 on the wildtype graph, so much of the step is small kernels. Measured 31 s/epoch alone versus 47 s/epoch two-up, i.e. 23.5 s per run-epoch, ~1.45x throughput, with ~6 GB spare. The paired reference is only a valid denominator if it ran under the same conditions as the arms subtracted from it, so packing must not be mixed within a stage.

- **One budget for every arm: 300 epochs, early stopping off.** 300 rather than 200 because the measured ROLL5-argmax was 171 for one `A0` seed-0 replicate and 232 for `H0_factor`, so a 200 cap truncates arm-dependently -- a budget that differs between arms scores "how long did the plateau look flat" alongside "how good is the mechanism". At the measured 47 s/epoch two-up, 300 epochs is ~3.9 h per run, comfortably inside the `--time=12:00:00` wall for all four stages.

- **Seeds are {0, 1, 2, 42}, not {0,1,2,3}.** `index_seed_{0,1,2,42}.json` already exist under `fig3_core/data_module_cache/`, whereas seed 3 would have concurrent array tasks racing to build and write the same split cache. (Wave 5 later dissolved the constraint entirely by pinning `data_module.split_seed=0`, making seeds vary initialization only.)

| stage | arms x seeds | runs | submit | config |
| --- | --- | --- | --- | --- |
| `ladder` | 4 lr arms x 2 | 8 | `--array=0-3%4` | default |
| `arch` | 5 x 4 | 20 | `--array=0-9%4` | default |
| `wave4a` | `R_ref`, `N_sink0` x 4 | 8 | `--array=0-3%4` | `cgt_expr_010` |
| `wave4b` | `R_ref`, `N_sink_mm` x 4 | 8 | `--array=0-3%4` | `cgt_expr_010` |

- **`wave4b` is a separate stage tag because it is not seed-comparable to what came before.** It relaunches `wave4a` with the sink's magnitude confound removed and three defects fixed in the masking path `wave4a` ran with: the fused SDPA branch was gated on having a mask (so 3 of 4 layers still materialized `[1, 9, 6608, 6608]`), attention dropout silently ran at p=0 on the masked layer versus p=0.1 elsewhere, and `attention_mask.layers` was never validated against encoder depth. Because SDPA consumes RNG differently from the manual softmax path, an identical seed is a different run -- hence the separate stage tag and the scorer wave boundary.

- **What `wave4b` actually returned** (`results/wave_convergence_stage-wave4b.csv`, from `wave4b_convergence.py --stage-tag stage-wave4b --window 5`): 8/8 runs finished at 300 epochs, and the paired within-seed `roll_max` delta `N_sink_mm - R_ref` was -0.0011 (s0), -0.0081 (s1), +0.0064 (s2), +0.0125 (s42), mean +0.0024 with sd 0.0090 -- against a between-seed sd of `R_ref` `roll_max` of 0.0449 (values 0.153, 0.056, 0.080, 0.062). The between-seed spread dwarfs the arm effect at this budget -- the same comparison the wave-5 launcher states as between-seed sd 0.0444 versus across-arm sd 0.0058, and the reason wave 5 pinned the split. `train_pf_end` spans 0.271-0.311 across all 8 runs and `roll_argmax` spans 86-297, so every run was still far from fitting its own train set and several were still improving when the budget cut them off.

- **A task fails if EITHER co-resident run fails**, and each run gets its own log file: interleaving two training streams into the array task's stdout would make both unreadable and break the epoch parsing the scorer relies on.

- Sizing note (node: 128 CPUs, 502 GB): 16 CPUs x 4 tasks = 64 of 128, and 110g x 4 = 440 of 502 GB reserved, against ~114 GB actually resident for four concurrent runs (~25 GB each). SLURM schedules on the reservation, so the over-reservation has to fit even though it is never used.
