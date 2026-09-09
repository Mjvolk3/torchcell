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
