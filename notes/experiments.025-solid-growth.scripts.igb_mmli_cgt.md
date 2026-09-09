---
id: f35auvn0quiy1hfpdcet80z
title: Igb_mmli_cgt
desc: ''
updated: 1788924442365
created: 1788924442365
---

## 2026.09.08 - IGB mmli Launcher for the 025 Full Build

4-GPU DDP launcher for one 025 arm on the IGB `mmli` node (compute-5-7, 4x A100 40 GB,
500 GB, no time limit), the IGB counterpart of `gh_cgt.slurm` and `delta_cgt.slurm`.
Takes the Hydra config name plus free overrides. Written for the joint-fitness arms
([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]]) but runs
any 025 config.

### What is on IGB

`sync_igb_025_build.sh` mirrored the build to
`/home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build/`.
Checked 2026.09.08: `processed/lmdb/data.mdb` is 554,075,586,560 bytes on both machines
with the same mtime, `raw/lmdb` is the empty stub PyG's `_download` needs, and
`data_module_cache/` holds the split indices. `data/sgd/genome`, `data/go/go.obo`,
`data/string` and `data/tflink` are present. The 010 build is not on IGB; only the 025
full build is, so every IGB arm reads the default `dataset.root_rel`.

### Why the job checks out its own worktree on the compute node

IGB's main clone (`~/projects/torchcell`) sits on the 019 branch and must not be flipped
under a job. The login node is access and transfer only, and a `git worktree add` there
checks out 12k files, which drew a warning in July 2026. Compute nodes have no network.
So the split is: `git fetch origin <branch>` once on the login node (network, light), and
the job itself runs `git worktree add --detach <path> origin/<branch>` on the compute node
at first use, or `checkout --detach origin/<branch>` after, printing the SHA it runs. A
later push changes what a later job runs, never a running one. The worktree gets a copy of
the main clone's `.env`; the exports in the script win over it since `load_dotenv` never
overrides a set variable, and `EXPERIMENT_ROOT` points at the worktree so the index
artifacts under `results/` are the branch's own.

### Budget

The weekly note measured the KL arm at ~58 min per epoch on this node against 19.4 on
GilaHyper. The replication (job 1598) peaked at epoch 14 and plateaued through epoch 35,
so the 4 day wall clock (~99 epochs) reads the plateau. Three arms need the whole node
each (4 GPUs, 250 GB), so they run one after another: about 12 days for the three.

### Launch, 2026.09.08

Queued three runs at seed 42, differing in the fitness weight only:

```bash
sbatch -J 025-kl-fit1   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_008
sbatch -J 025-kl-ctrl   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_000
sbatch -J 025-kl-fit0.1 experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_009
```

Runs log offline; sync from the login node with
`experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh` and read progress from
W&B on GilaHyper, never from the IGB logs.
