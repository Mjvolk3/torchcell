---
id: jlzht634j7zscawmaf03ovb
title: Gh_expr_wave5
desc: ''
updated: 1785532009589
created: 1785532009589
---

## 2026.07.31 - Holding Per-Epoch Conditions Fixed So "Is Training Saturated?" Has an Answer

Every architecture delta measured in the expression round so far is <= 0.03, and all of them were taken on runs whose train curve was still climbing at the epoch cap -- so the first thing worth buying is a run whose per-epoch conditions never change for its whole life. That is what this GilaHyper launcher is: one stage = one packing, one budget, one worker count, one config, start to finish, with the stages submitted as separate jobs so a long run and a short run never share a GPU.

- **The two headline stages get DISJOINT GPUs, and the reason is measured.** Packing changes per-epoch time (31 s/epoch alone versus 47 s/epoch at 2 runs/GPU, pre-SDPA-fix). A 2000-epoch `h1` run co-resident with a rotating series of 400-epoch `maskdepth` runs would see its neighbour change several times, so its trajectory would encode the scheduler rather than the model.

| stage | arms x seeds | runs/GPU | epoch cap | config | submit |
| --- | --- | --- | --- | --- | --- |
| `h1` | 2 x 2 = 4 | 2 | 2000 | `cgt_expr_011` | `--array=0-1%2` |
| `maskdepth` | 3 x 4 = 12 | 2 | 400 | `cgt_expr_011` | `--array=0-5%2` |
| `h2gh` | 3 x 4 = 12 | 2 | 300 | `cgt_expr_011` | 6 tasks |
| `bench_b{8,32}_p{2,3}` | 1 arm, 1 seed | 2 or 3 | 60 | `cgt_expr_011` | one cell per task |
| `v9` | 8 x 1 | 2 | 10000 | `cgt_expr_v9_mask` | 4 GPUs |

- **`train_eval_every=5`, not 1 -- the setting decides whether the stage finishes at all.** The first `h1` submission (job 1417) ran every=1 and clocked 85 s/epoch, reaching only epoch 17 in 24:04, against 36 s/epoch for the `maskdepth` stage in the same window. The eval-mode train pass costs ~65 s on its own because it sweeps the 1244-strain TRAIN split against validation's 155 (~8x the rows). At 85 s/epoch the `--time=23:00:00` wall truncates at ~970 epochs, short of the cap; at ~36 s/epoch the full 2000 epochs is ~20 h and fits with margin. Every 5 epochs still yields 400 curve points, far more than "has it stopped moving" needs.

- **Early stopping is OFF for every wave-5 arm.** It selects a different epoch per arm (wave 4b's val-loss argmins landed 42-272, and consistently later for the sink), so it puts epoch-selection into the arm comparison. Fixed budget plus a pre-registered reducer instead.

- **Seeds now vary initialization only.** `cgt_expr_011` pins `data_module.split_seed=0`, so `seed` no longer redraws the validation split. The old scheme had one knob doing both, and the between-seed sd (0.0444) was 7.7x the across-arm sd (0.0058) -- the axis being measured was drowned by the axis that was not supposed to move.

- **Worker count is split across co-resident runs**: `NUM_WORKERS=(SLURM_CPUS_PER_TASK - 2) / RUNS_PER_GPU`, i.e. 7 at 16 CPUs and 2 runs. Handing each run the full allocation oversubscribes the cores and makes the interference non-uniform across tasks, which would show up as arm-dependent epoch time.

- **`h2gh` is independent replication, not extra n.** It re-runs the mmli H2 arms (`W_ref`, `X_mix`, `X_mix_rezero`) at 4 replicates on RTX 6000 Ada with its own co-launched reference, and is deliberately NOT pooled with the A100 runs -- `_007` pooled A100 with Ada and their means differ (0.0242 vs 0.0182). A mechanism showing the same sign on two hardware pools with separate references is a stronger claim than one pool at n=8.

- **The `bench_*` stages are a launch blocker, not science.** Wave 6 wanted 5,000 epochs in 24 h = 17.3 s/epoch; the best rate measured was 32 s/epoch, and a 3-worker configuration gave 270 s/epoch with the A100 at 0% util (dataloader-starved). The lever tested is batch size, straight from the cost model -- the encoder runs on the WILDTYPE graph at batch 1, so `C_enc` is paid once per STEP and per-epoch cost is `ceil(n/B) * (C_enc + B*C_pert)`; at B=8 that is ~155 steps to cover 1,244 strains. Each task is one (batch, packing) cell for 60 epochs, and the answer is read from `perf/epoch_seconds` (median over the last 40 epochs) rather than job elapsed, so dataset load and startup are excluded. The answer that came back was 23.51 s/epoch at 3 runs/GPU and batch 32 (jobs 1431-1434) -- the packing wave 6 was then submitted at.

- **`export TORCHCELL_CONFIG` sits AFTER the case block.** Exporting it earlier silently pinned every stage to `cgt_expr_011` regardless of the `STAGE_CONFIG` the stage selected, which would have run `v9` on the non-masked config.

- **A task fails if EITHER co-resident run fails.** A half-completed task leaves an arm missing a replicate, which breaks the within-seed pairing for that replicate -- silently, if the task were allowed to exit 0.

- One inconsistency to carry forward: the `v9` stage documents a budget matching wave 6 (10,000-epoch cap, 48 h wall) while the in-file directive is `--time=23:00:00`, so that 48 h has to be supplied as `--time` on the submit line (CLI beats in-file directives).
