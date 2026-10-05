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
