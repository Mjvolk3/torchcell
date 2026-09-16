---
id: rwf6z4zn4apfk5qp1rhefpm
title: Delta_cgt
desc: ''
updated: 1788922185845
created: 1788922185845
---

## 2026.09.08 - Delta launcher for the graph-regularization sweep

Four scripts, one arm per job, all on the 010 build:

- `experiments/025-solid-growth/scripts/sync_delta_010_build.sh`: run from GilaHyper, rsyncs
  the 1.5 GB migrated 010 build `001-small-build-schema-v2` to
  `/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2`
  (one Duo prompt). The 025 full build is 3.2 TB and stays here; the 010 build holds the
  same 376,732 records with bit-identical labels (`results/label_parity_010_vs_025.json`).
  The frozen `001-small-build` cannot be used: its records predate the perturbation-ontology
  refactor and fail validation (204 errors per record), and its JSON files are owned by the
  graph-build user so the dataset's file lock cannot even be taken. The schema-v2 copy is
  key-preserving, so 010's split indices apply unchanged (checked in
  `make_010build_index_artifacts.py`).
- `delta_preflight_025.sh`: run on a Delta login node from the repo root; checks the
  interpreter, the build, the graph roots shipped for 019, the index artifacts and configs
  in the checkout, the slurm output dir, and reports the Taiga mount and account balances.
- `delta_cgt.slurm`: one gpuA40x4 node, 4-GPU DDP through torchrun on loopback, zero
  dataloader workers (the Lustre lesson from `019 delta_grid_common.sh`), config name and
  free Hydra overrides as arguments. Account, time and job name are `sbatch` flags.
- `delta_submit_sweep.sh canary|sweep|disjoint`: the canary is one 24 h KL run at
  lambda 1e-3, seed 1, to read minutes per epoch before anything else is submitted.

Configs, all `SAME MODEL AND SCHEDULE` as their S0 parents but pointing at the 010 build via
the new `dataset.root_rel` key and 010-native index artifacts:

| config | arm | split | parent |
|---|---|---|---|
| `cgt_010b_r_kl_005` | soft KL, layer 1; base of the lambda sweep, lambda = 0 is the no-penalty arm | 010 random | `cgt_s0_r_kl_000` |
| `cgt_010b_r_mask_006` | hard mask, layer 1 (the lambda to infinity limit) | 010 random | `cgt_s0_r_mask_003` |
| `cgt_010b_q_kl_007` | soft KL on 010's query-pair-disjoint split (Table 10's partition) | 010 disjoint | `cgt_s0_r_kl_000` |

Script changes in `equivariant_cell_graph_transformer.py`: `dataset.root_rel` (default the 025
full build), `seed` (default 42, `L.seed_everything`), and W&B tags `lambda_<x>`, `seed_<n>`,
`build_<name>` read from the resolved config so command-line overrides are what gets tagged.

Sweep: lambda in {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1} x 3 seeds (1 is included because 010's
coefficient carried a x367 defect, effective weight about 0.37), plus the hard mask x 3 and
the disjoint KL x 3: 27 runs. GPU-hours charge per GPU, 4 per node-hour: 21 KL runs x 96 h +
6 cheap runs x 48 h = 2,304 GPU-h for the sweep, 288 for the disjoint arm. Measured minutes
per epoch, 4 GPUs, batch 256, bf16: GilaHyper RTX 6000 Ada 19.4 (KL) / 14 (mask); IGB 58
(KL) / 18 (lambda = 0). Delta A40 unmeasured; the canary measures it.

Not in this package (phase 2, needs trainer code): the gradient probe at epochs 0, 1, 2, 5,
10, 20 for panel c of the mock-up, and the degree-matched random-graph control for panel f.

## 2026.09.12 - The 025 LMDB is staged onto node-local NVMe; Lustre and Taiga are both too slow

Three measured read paths for the same 554 GB build, same model, 4-GPU DDP, zero
dataloader workers:

| where the LMDB lives | min per epoch | evidence |
|---|---|---|
| GilaHyper local NVMe | 19 | jobs 1598 to 1659 |
| IGB local disk | 17 | jobs 2391132 to 2391134 |
| Delta, Taiga NFS mount | 63 | 21934082 (16 epochs in 16 h 42 m); both jobs later died on a 30-min NCCL watchdog |
| Delta, Lustre `/scratch` (4 stripes of 114) | 200 to 220 | 21947151: epochs at +3.30, +3.28, +3.53, +3.66 h; CPU load 1.15 on 16 cores, node otherwise empty |

With `num_workers=0` every LMDB page read happens in the training process, so read
latency is the epoch time; Lustre's random small reads are the worst of the three. The
launcher now copies `data.mdb` to `/tmp` at job start (1.5 TB NVMe on the A40 nodes,
measured on gpub066 with `srun --overlap`; the copy job did 290 MB/s from Taiga, so about
30 min) and trains from a node-local `DATA_ROOT` mirror: every entry along the build path
is a symlink back to `/scratch` except the LMDB directory, which is a real copy so LMDB
can write its `lock.mdb`. Genome, GO, STRING, TFLink, the index files, the split
artifacts, W&B directories and `data_module_cache` all stay on `/scratch`. The mirror
layout was checked on a fake tree before committing. `STAGE_LMDB=0` reads `/scratch`
directly; the 010-build configs never stage (1.5 GB, page-cache resident). The local
copy under `/tmp/$SLURM_JOB_ID` is removed at job exit.

## 2026.09.14 - Sweep submitter rewritten for the 025 build

`delta_submit_sweep.sh` now composes on `cgt_s0_r_kl_ctrl_013` (the constant-rate
protocol) instead of the 010b configs: `sweep` submits 21 jobs, lambda 0 and the hard
mask `cgt_s0_r_mask_028` first inside each seed, then 1e-2, 1e-1, 1e-4, 1e-5, 1; lambda
1e-3 is the three ctrl_013 seeds already chained (22034665, 22034668, 22034671). One
chain of `--dependency=after:<prev>+30`, 24 h clocks (27 min/epoch measured on the NVMe
path; 30 epochs plus the stage-in took 16 h 01 m in 22030924), independent of the fitness
chain so Delta can run one job from each. `DRY=1` prints; `AFTER=<jobid>` chains the
first job. Submission waits for the 08:00 to 09:00 scheduler maintenance and is set to
fire at 09:01 CDT from this session.

Checked before the plan: the checkpoints of a staged run land on `/scratch` (the mirror
links `models/`), so the per-arm best checkpoints the figure's panels a, d and e read are
kept. Four NCCL watchdog core dumps from the Taiga-era failures sit in the Delta
worktree's `experiments/` at 9.2 GB each (37 GB, untracked, `core.pt_nccl_watchdg.*`);
they can go.

### Submitted 2026-09-14 01:50 CDT

The user chose to submit before the maintenance rather than after it (queue age counts
toward priority; pending jobs persist across a scheduler restart). From 8e9a20fb, one
chain, first job waiting on Priority beside the fitness chain's 22034665:

| seed | lambda 0 | mask | 1e-2 | 1e-1 | 1e-4 | 1e-5 | 1 |
|---|---|---|---|---|---|---|---|
| 1 | 22055147 | 22055149 | 22055151 | 22055152 | 22055153 | 22055154 | 22055155 |
| 2 | 22055156 | 22055157 | 22055158 | 22055159 | 22055160 | 22055161 | 22055162 |
| 3 | 22055163 | 22055164 | 22055165 | 22055166 | 22055167 | 22055168 | 22055169 |

Lambda 1e-3 remains the fitness chain's 22034665 / 22034668 / 22034671. What the sweep
does not cover: the random-graph control (panel f, needs the rewiring option), the
per-term gradient probe (panel c, needs a trainer hook), and the checkpoint readouts
for panels a, d and e.

### Random-graph control submitted 2026-09-14 03:20 CDT

`delta_submit_sweep.sh random` from 5d676892, chained after the sweep's last job
22055169: 22056267 (seed 1), 22056268 (seed 2), 22056269 (seed 3), config
`cgt_s0_r_kl_rand_031`. With these the queue holds everything the figure trains on:
the fitness chain (whose three controls are lambda 1e-3), the 21-job sweep, and the
three random-graph seeds; 32 jobs, all on bfjt-delta-gpu. The gradient probe is in the
worktree they read at start. What remains is readout code over the finished runs and
their checkpoints, not more training.

## 2026.09.14 - The first two heads of both chains OOM'd; the KL arms were at 96 percent of the card

22034665 (ctrl_013 seed 1) and 22055147 (lambda 0 seed 1) each staged in about 830 s and
then FAILED at their first training batch with `torch.OutOfMemoryError` in a
transformer layer's softmax: 43.08 GiB in use on a 44.42 GiB A40, 1.46 GiB requested
(the fp32 [9, 6608, 6608] attention of one layer). W&B system metrics for the run that
succeeded on the same code path, 22030924 (b3n4ax4a), show GPU memory at 94.8 to 96.5
percent for the whole run, so every KL arm has been training within about 0.3 GiB of the
card, and what tips it is not code but the few hundred MB of CUDA context and NCCL
buffers that vary from node to node. The two failures are on gpub nodes other than
22030924's gpub051.

The bulk is a design leftover: `need_attention_for_graph_reg = graph_reg_lambda > 0`
made every KL arm take the manual attention path in ALL eight layers, saving scores,
probabilities and dropout mask (about 5 GB per layer) for backward, while the KL reads
layer 1 only and `compute_graph_regularization_loss` skips the other seven. Fixed in the
model: the manual path is taken only at layers named by a regularized head
(`regularized_layers`, {1} for every 025 KL arm); the other layers use the fused SDPA
kernel, which is what the mask arms already use in their unmasked layers. `graph_reg_loss`
is unchanged in value (it only ever read layer 1); layers 0 and 2 to 7 change kernel,
which moves their arithmetic at bf16 rounding and consumes dropout RNG differently, so a
run at the same seed is not bit-identical to a run under the old path. The trainer now
logs `train/cuda_peak_allocated_gb` and `train/cuda_peak_reserved_gb` at every epoch end
and prints them, so the margin is a number in every log from here on.

Consequence for comparability: 22030924 (fit_014 seed 1) is the only complete run on the
old all-layers path; every other job in the three chains starts under the new path. It is
resubmitted at the end of the fitness chain so the fitness experiment is read from one
code path; the old run stays on W&B as a check that the kernel change does not move the
number.

Resubmitted 21:50 CDT from 14c02243, after the Delta worktree was fast-forwarded so
every pending job starts under the fix: 22080055 ctrl_013 seed 1 (after the fitness
chain's 22034673), 22080056 fit_014 seed 1 rerun (after it), 22080057 lambda 0 seed 1
(after the random-graph 22056269). 33 jobs queued. Measured on one RTX 6000 Ada at 64
records per GPU, the KL control now peaks at 13.5 GiB allocated, 14.8 GiB reserved,
against 41 GiB on the old path; the probe printed point 6.19, dist 0.15, graph penalty
1.01 at epoch 0 there.

### Re-chained into six parallel chains, 22:40 CDT

One chain per seed instead of one chain per experiment, changed in place with
`scontrol update Dependency` so no job lost its queue age. Sequential chains would
have taken the 21-job sweep about two weeks at one node at a time; with a chain per
seed Delta can run up to six of these at once, and the stagger that motivated the single
chain (filelock contention on Taiga) no longer applies because each job stages its own
LMDB and holds its own locks on node-local disk.

| chain | order |
|---|---|
| fitness s1 | 22034666 fit_015 -> 22080055 ctrl -> 22080056 fit_014 rerun |
| fitness s2 | 22034667 fit_014 -> 22034668 ctrl -> 22034669 fit_015 |
| fitness s3 | 22034670 fit_014 -> 22034671 ctrl -> 22034673 fit_015 |
| sweep s1 | 22055149 mask -> 22055151 1e-2 -> 22055152 1e-1 -> 22055153 1e-4 -> 22055154 1e-5 -> 22055155 1 -> 22056267 random -> 22080057 lambda 0 |
| sweep s2 | 22055156 lambda 0 -> 22055157 mask -> 22055158 -> 22055159 -> 22055160 -> 22055161 -> 22055162 -> 22056268 random |
| sweep s3 | 22055163 lambda 0 -> 22055164 mask -> 22055165 -> 22055166 -> 22055167 -> 22055168 -> 22055169 -> 22056269 random |

## 2026.09.15 - Second OOM, at the diagnostic validation; queue ages

22034666 (fit_015 seed 1, started 03:37 after 40 h in the queue) and 22055149 (mask
seed 1, started 04:04 after 26 h) both trained ten epochs at 20 to 25 min per epoch on
the layer-restricted path and then died in `validation_step` at validation batch 2 of
37 of the epoch-10 diagnostic pass: `torch.OutOfMemoryError`, 42.1 GiB allocated,
1.46 GiB requested. On a diagnostic epoch (`plot_transformer_diagnostics_every_n_epochs:
10`, `_is_scheduled` fires when `(epoch + 1) % 10 == 0`) validation asks every layer for
its attention, so all eight layers take the manual path and the model returns eight
[1, 9, 6608, 6608] fp32 matrices (11.7 GiB) on top of the per-record tensors of a
256-record validation batch. That is the true peak of a run; the training-side fix of
2026-09-14 did not touch it, and 22030924's 96.5 percent W&B memory reading was this
pass, not training. Validation batch size is not part of the protocol (torchmetrics
accumulate over records; no training step changes), so `cgt_s0_r_kl_000` now sets
`data_module.val_batch_size: 64`, and the trainer logs `val/cuda_peak_allocated_gb` at
every validation end.

Two changes, both outside the protocol: the trainer returns the diagnostic attention for
the first validation batch only (the attention is computed on the wild-type graph in
eval mode, so every batch of the epoch carried the same eight matrices and the
accumulators averaged 37 identical samples), and `cgt_s0_r_kl_000` validates at 32
records per rank. Measured on one RTX 6000 Ada with diagnostics on every epoch: the
diagnostic validation peaks at 16.4 GiB allocated (14.5 in the sanity check), the
training epoch at 6.8 GiB, against 42 GiB before. `val/cuda_peak_allocated_gb` is logged
at every validation end.
