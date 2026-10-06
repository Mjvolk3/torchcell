---
id: cf8pzsk3n7dxmx2r0a9cx3o
title: Gh_small_model_bench
desc: ''
updated: 1791240979599
created: 1791240979599
---

## 2026.10.05 - Throughput of scaled-down trunks, for sizing single-GPU Delta jobs

Four RTX 6000 Ada cards on GilaHyper for three hours. The question was how to pack a round
of twelve split seeds onto one Delta A40 and finish inside two days. Convergence cannot be
measured in three hours, so the benchmark measures the two quantities Delta sizing needs,
seconds per epoch and card memory per run, on the v19 store (1,107 both-label training rows,
35 batches of 32) under the v19 arms. Cells 0 to 3 run at zero data-loader workers because
Delta does (W&B project `torchcell_019_small_bench`, slurm 3267 cells 0 to 3, 3271 cells 4 and 5).

| cell | input | width, layers | runs per card | workers | params | epoch 1 | steady s/epoch per run | card memory |
|---|---|---|---|---|---|---|---|---|
| 0 | four-part stack (3,328-d) | 90, 6 | 4 | 0 | 6.68 M | 146 to 148 s | 78 | 9.5 GB per run, 38 GB |
| 1 | ProtT5 alone (1,024-d) | 90, 6 | 4 | 0 | 1.45 M | 156 s | 80 to 81 | 9.8 GB per run, 39 GB |
| 2 | ProtT5 alone | 90, 3 | 6 | 0 | 1.15 M | 146 to 157 s | not reached | 8.2 to 8.5 GB per run, 45 GB; one run out of memory |
| 3 | ProtT5 alone | 54, 2 | 8 | 0 | 0.74 M | 142 to 149 s | not reached | 8 to 9.5 GB per run; three of eight out of memory |

Cells 2 and 3 were canceled at epoch 1 once they had answered. Three findings:

- **Card memory does not follow parameter count.** Every shape held 8 to 10 GB per run, so
  the memory is activations over the 6,600 gene tokens at batch 32, and the trunk's width and
  depth do not change it. Four runs fit a 44.4 GiB card; six sit at the edge; eight fail.
- **Epoch time does not follow shape or packing at zero workers.** The first epoch cost 142 to
  157 s in every cell, with four runs or eight, two layers or six. The GPU is not the
  bottleneck; the main process is (loading and collating the heterogeneous graphs). Delta's
  v11 round logged 81 to 134 s per epoch with ONE run per A40 at zero workers, the same band
  as four runs per card here, which says the same.
- **Width and depth therefore buy nothing**, which matches the 2026-10-04 audit (an epoch costs
  the same at 0.68 and 1.45 million parameters). The scaling-down that is free is the input:
  ProtT5 alone drops 85 percent of the parameters at no measured accuracy cost. The v21 round
  adopts that and keeps width 90 and six layers.

Cells 4 and 5 (second wave, same window) time the lever that is left, persistent data-loader
workers, at four runs with three workers each and five runs with two; numbers in the weekly note.

Cells 4 and 5 (job 3271): four runs with three persistent workers each read 35 to 38 s per
epoch at steady state, five runs with two workers each 38 to 40 s (one of the five ran out of
card memory, confirming four as the ceiling), against 68 to 75 s at zero workers in cells 0
and 1 at the same time. The first epoch fell from 146 to 182 s to 94 to 109 s. So persistent
workers halve the epoch, and the zero-worker Delta rule, measured in July with non-persistent
workers that re-spawn every epoch, is the setting to retest there. Cell 6 (job 3273) is
Delta's shape, four runs with one worker each on eight CPUs.

Cell 6 (job 3273, four runs with ONE persistent worker each on eight CPUs, Delta's share of a
gpuA40x4 node): 58 to 64 s per epoch at steady state, first epoch 122 to 135 s, 38 GB on the
card, no failures. On this card a 1,200-epoch pack of four therefore takes about 20 h; the
A40 figure is a hypothesis until the Delta canary logs it (expected slower, 1.5 to 1.8 times,
still inside two days).

| cell | runs per card | workers per run | CPUs | steady s/epoch per run | 1,200 epochs, pack |
|---|---|---|---|---|---|
| 0, 1 | 4 | 0 | 16 | 68 to 75 | 23 to 25 h |
| 6 | 4 | 1 persistent | 8 | 58 to 64 | about 20 h |
| 5 | 4 alive of 5 | 2 persistent | 16 | 38 to 40 | about 13 h |
| 4 | 4 | 3 persistent | 16 | 35 to 38 | about 12 h |

## 2026.10.05 - Depth by operator grid (job 3274, cells 10 to 13)

One card per depth (2, 3, 4, 6 layers at width 90), the four operators co-resident on each
(softmax reference, null sink with the gate open, Hadamard product, rank-64 response basis),
v21 config on split seed 0, three persistent workers per run, two hours. Steady-state seconds
per epoch per run at epoch 9, cards at 86 to 100 percent utilization:

| layers | S_ref | S_sink | S_hadam | S_basis64 | card memory |
|---|---|---|---|---|---|
| 2 | 37 to 40 | | | | 33 GB |
| 3 | 38 to 41 | | | | 37 GB |
| 4 | 39 to 43 | | | | 36 GB |
| 6 | 39 to 42 | | | | 37 GB |

(Per-operator columns are within 3 s of each other at every depth; the table collapses them.)
So with the card fed, the epoch still does not depend on depth: the encoder's six layers over
6,600 gene tokens are not where the time goes, and neither is the main process any more (that
was the zero-worker regime). What remains is everything that scales with batch size times
genes and not with depth: the per-batch node tensors, the per-gene readout and loss over
6,607 outputs for 32 strains, the metrics, and the eval passes. Hypothesis (untested): the
cost center is the batched node-feature tensor and the readout, which a batch-size sweep
or a profile of one step would show in minutes. The 2026-10-04 audit's cached-encoder
speedup (42 s to 2.6 s per step) was measured on a CPU, where the encoder dominates; on a
GPU this grid says it would not. The operator read at the two-hour mark follows below.

## 2026.10.06 - Depth x operator grid read (job 3274, cells 10 to 13)

Job 3274 was cut from two hours to 96 minutes (`scontrol update TimeLimit=01:36:00`) to fit the user's window and ended by time limit with all sixteen runs alive at epoch 119 (six layers) to 132 (two layers). W&B project `torchcell_019_grid_bench`, read by `scratchpad/grid_bench_read.py` (val last-20 mean, rolling max, train-eval Pearson, pred_sd_ratio at the last epoch, mean seconds per epoch after epoch 5). One split seed, one init, at about a tenth of the budget: NOT a result. The validation Pearson per feature is 0.00 to 0.04 against 0.09 to 0.13 for the same arm at 1,200 epochs in v19; pred_sd_ratio 0.05 to 0.27 against the launch gate 0.05.

| layers | S_ref | S_sink | S_hadam | S_basis64 | s per epoch (mean of four) |
|---|---|---|---|---|---|
| 2 | 0.010 | 0.017 | 0.020 | 0.007 | 41.9 |
| 3 | 0.013 | 0.018 | 0.033 | 0.017 | 43.5 |
| 4 | 0.004 | 0.010 | 0.029 | 0.012 | 44.4 |
| 6 | 0.034 | 0.032 | 0.039 | 0.039 | 45.8 |

Train-eval Pearson at the last eval-mode pass: S_ref 0.166 to 0.246, S_sink 0.152 to 0.235, S_hadam 0.146 to 0.217, S_basis64 0.191 to 0.250. Rolling max 0.046 to 0.063 on every run, separates nothing.

- Throughput: the operators are within 1 s of each other at every depth; depth costs about 1 s per layer per epoch, a tenth of the epoch between two and six layers. That is the whole case for a shallower trunk, and it does not buy packing (memory is 33 to 37 GB per four runs at every depth).
- Hypothesis (one seed, early): the Hadamard operator is above the softmax reference at all four depths on val last-20 (+0.005 to +0.025) with the lowest train-eval fit at every depth. Six layers is above the shallower trunks on every operator despite the fewest epochs, with the lowest train fit. Both are read against v21 (twelve paired seeds at budget), not acted on.
- Example runs: depth 3 Hadamard <https://wandb.ai/zhao-group/torchcell_019_grid_bench/runs/ydyxxub2> and its reference <https://wandb.ai/zhao-group/torchcell_019_grid_bench/runs/gfzwmdq8>
