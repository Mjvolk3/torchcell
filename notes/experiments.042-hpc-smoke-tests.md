---
id: j7qkvf66w54b87a0zg1zuo0
title: 042 Hpc Smoke Tests
desc: ''
updated: 1792222990542
created: 1792222990542
---

## 2026.10.17 - torchcell on Polaris, Aurora and Perlmutter, driven from GilaHyper

Goal: the Genesis hackathon allocations (ALCF project `GenesisHackathonOct26`: 3,884 Polaris
node-hours, 18,468 Aurora node-hours; NERSC project `m5429`) become dispatch targets for
Claude Code sessions on GilaHyper, the way IGB cabbi and NCSA Delta already are. First
milestone: an installed environment, a data root and a `.env` on each machine, and a GPU
job on each that trains and logs to W&B.

### Access from GilaHyper

ALCF has no ssh key or certificate option, so each machine is reached through an OpenSSH
ControlMaster socket the owner opens once with a MobilePASS+ passcode (`ssh polaris`,
`ssh aurora`; `ControlPersist 10h`). NERSC is reached with the 24 h sshproxy certificate
(`ssh -i ~/.ssh/nersc mjvolk3@perlmutter.nersc.gov`). A session checks `ssh -O check
<alias>` before dispatching and asks the owner to log in again when the socket is gone.
Login nodes carry scheduler commands and rsync only; every poll runs on GilaHyper.

### Where torchcell lives on each machine (`scripts/machines.sh`)

| machine | prefix (`TC_PREFIX`) | scheduler | GPU | wheel flavor |
|---|---|---|---|---|
| polaris | `/eagle/GenesisHackathonOct26/mjvolk3` | PBS Pro, `-A GenesisHackathonOct26 -l filesystems=home:eagle` | 4x A100 40 GB per node | `cu128` |
| aurora | `/flare/GenesisHackathonOct26/mjvolk3` | PBS Pro, `-l filesystems=home:flare` | 6x Intel Max 1550 (12 tiles) per node | `xpu` |
| perlmutter | `/global/cfs/cdirs/m5429/mjvolk3` | slurm, `-A m5429_g -C gpu` | 4x A100 per node | `cu128` |

Under each prefix: `miniforge3/` (our own, python 3.13), `torchcell/` (the rsynced tree the
jobs run from), `torchcell-data/` (`DATA_ROOT`). The site Python stacks are below torchcell's
floor (`requires-python >= 3.13`, `torch >= 2.11`, `torch_geometric >= 2.8`): Polaris
`conda` base is python 3.12 / torch 2.8 / pyg 2.6.1, Aurora `frameworks` is python 3.12 /
torch 2.13 Intel-patched without pyg or lightning, Perlmutter `pytorch/2.13.0` is python 3.12.
All three run SLES 15 with glibc 2.38, so the Rocky Linux container the 2025 IGB and Delta
setups needed for glibc is not needed here.

Queues for 1 to 16 nodes: `debug` (2 nodes, 1 h) for tests; `capacity` (Polaris 4 nodes,
Aurora 16 nodes, 7 days) for training. The `small` queue starts at 10 nodes on Polaris and
256 on Aurora. No reservation admits the hackathon group (checked `pbs_rstat -f` on
2026-10-17). Compute nodes on both ALCF machines reach W&B through
`proxy.alcf.anl.gov:3128`; Perlmutter compute nodes reach it directly.

### Scripts

- `scripts/machines.sh`: the table above as `TC_*` variables plus a `tc_ssh` wrapper.
- `scripts/sync_code.sh <machine>`: rsync of the current checkout to `TC_TREE` (no `.git`,
  notes, paper, data, results or job output; one ssh session, `--rsync-path` makes the dir).
- `scripts/write_remote_env.sh <machine>`: writes the machine's `.env` (paths only; W&B
  authenticates through the remote `~/.netrc`, whose entry was streamed over ssh, never
  printed).
- `scripts/install_env.sh`: runs on the login node; Miniforge, `python=3.13`,
  `torch==2.11.0` from the `cu128` or `xpu` index, `env/requirements.txt`, `torch-scatter`
  from the PyG wheel index (cu128 only), `pip install -e --no-deps`. Idempotent.
- `scripts/smoke_train.py`: imports the stack, picks CUDA or XPU, trains a scatter-pooled MLP
  on synthetic data for 300 steps logging every step to W&B project `torchcell_042_hpc_smoke`,
  then on CUDA runs a Lightning `Trainer` with `WandbLogger` on the same run. Exits non-zero
  unless the loss fell. Validated on GilaHyper first (RTX 6000 Ada, `torch_scatter` path).
- `scripts/{polaris_smoke.pbs,aurora_smoke.pbs,perlmutter_smoke.slurm}`: the `debug` jobs;
  `scripts/submit_smoke.sh <machine> [submit|status]` submits from GilaHyper.

Install detail that bit all three machines on the first pass: `env/requirements_experiments.txt`
is RAPIDS (`cudf-cu12`, `cuml-cu12`), nothing in torchcell imports it, and its placeholder
wheels run `nvidia-smi` during `pip install`, which fails on a login node. The installer takes
`env/requirements.txt` only.

### Results

| machine | job | node / GPU | steps/s (manual loop) | loss first -> last | scatter | Lightning | W&B |
|---|---|---|---|---|---|---|---|
| polaris | 7757261 (`debug`, 3 min 9 s wall) | x3108c0s13b0n0, A100-SXM4-40GB | 57.9 | 1.209 -> 0.029 | torch_scatter | ran, loss 0.022 | [1ghwjzac](https://wandb.ai/zhao-group/torchcell_042_hpc_smoke/runs/1ghwjzac) |
| perlmutter | 59896274 (`debug`, 1 min 42 s wall) | nid001229, A100-SXM4-40GB | 103.5 | 2.604 -> 0.023 | torch_scatter | ran, loss 0.033 | [ltt5crz1](https://wandb.ai/zhao-group/torchcell_042_hpc_smoke/runs/ltt5crz1) |
| aurora | 8938709 (`debug`, 5 min 37 s wall) | x4310c0s0b0n0, Intel Data Center GPU Max 1550, one tile | 47.1 | 15.95 -> 0.026 | torch.index_add_ (native) | skipped, no XPU accelerator in Lightning 2.6 | [5a4n3mri](https://wandb.ai/zhao-group/torchcell_042_hpc_smoke/runs/5a4n3mri) |

Aurora ran the pip `torch 2.11.0+xpu` wheel with no site module loaded, so the oneAPI runtime
the wheel bundles is enough on a compute node. The native `index_add_` scatter path works on
XPU; the models that import `torch_scatter` directly (nine files, among them
`equivariant_cell_graph_transformer.py` and `deep_set.py`) are not yet runnable there, and
Lightning-based trainers need an XPU accelerator before a real training run goes to Aurora.

Perlmutter job 59896049 (nid001617, A100-SXM4-40GB, 4 min 6 s) failed before training:
`wandb.init` could not take the lock on `run-*.wandb.syncstate` under `/global/cfs` (errno 524;
CFS does not support `flock`). The job script now sets `WANDB_DIR=/pscratch/sd/m/mjvolk3/wandb`;
anything that locks files (W&B, LMDB, SQLite) belongs on `$SCRATCH` there, with results on CFS.

Result JSON: `results/smoke_polaris_7757261.json` (copied back from the machine). PBS with
`#PBS -o <dir>/` writes the job output as `<dir>/<jobid>.OU`, not `<jobname>.o<seq>`, so a
dispatcher reading job output looks for the job id. The GilaHyper validation run (RTX 6000 Ada)
is [2ix39tqe](https://wandb.ai/zhao-group/torchcell_042_hpc_smoke/runs/2ix39tqe).

## 2026.10.17 - PyG and torchcell-model probe: does PyG run on Aurora's XPU?

The smoke test above trained a plain MLP; it never called a PyG layer or a torchcell model, so
whether PyG works on Intel XPU was still open. `scripts/pyg_probe.py`
([[experiments.042-hpc-smoke-tests.scripts.pyg_probe]]) runs 45 named checks and records each
one; `scripts/{polaris,aurora}_pyg_probe.pbs` run it in `debug`. GilaHyper validation: 45/45.

| machine | job | GPU | PyG core (17) | torch_scatter | torchcell imports (27) | verdict |
|---|---|---|---|---|---|---|
| polaris | 7757532 | A100-SXM4-40GB | 17/17 | present | 27/27 | everything runs |
| aurora | 8938797 | Max 1550, one tile | **17/17** | absent | 2/27 (`act`, `dcell`) | PyG runs; torchcell blocked by one dependency |

PyG on XPU, measured: `torch_geometric.utils.scatter` for sum, mean, max, min, mul with
gradients; `global_add/mean/max_pool`, `to_dense_batch`, index `softmax`; GCNConv, SAGEConv,
GATv2Conv, GINConv, TransformerConv forward and backward; a two-type `HeteroConv`; bf16
autocast. All pass on the pip `torch 2.11.0+xpu` wheel with PyG 2.8.1, no site module loaded.

torchcell on XPU, measured: every one of the 25 failing imports ends in
`ModuleNotFoundError: No module named 'torch_scatter'`, including modules that never use it
(`mlp`, `norm`, `protT5`), because `torchcell/models/__init__.py` imports `deep_set`, which does
`from torch_scatter import scatter_add, scatter_mean` at module top. There is no second root
cause in the probe's reach. The nine files that import `torch_scatter` directly: `models/
{deep_set, self_attention_deep_set, linear, dango, cell_graph_transformer,
equivariant_cell_graph_transformer, hetero_cell_bipartite_dango_gi,
hetero_cell_bipartite_dango_gi_lazy}.py` and `data/graph_processor.py`.

Hypothesis (untested): replacing those imports with `torch_geometric.utils.scatter`, which has the
same reduce semantics and already routes to the native ops when `torch_scatter` is absent, makes
every model importable on Aurora. Whether a full training step then runs on XPU, and at what
throughput against the A100, is the next measurement; the Lightning XPU accelerator is the other
missing piece for the Lightning-based trainers. First probe attempts, for the record: Polaris
7757497 and Aurora 8938786 ran on a tree missing `torchcell/data/` (the unanchored `data/` rsync
exclude, fixed in `sync_code.sh`), and 8938786 died at `import torchcell.models` before the
import section was made crash-proof.

Result JSONs: `results/pyg_probe_polaris_7757532.json`, `results/pyg_probe_aurora_8938797.json`.
