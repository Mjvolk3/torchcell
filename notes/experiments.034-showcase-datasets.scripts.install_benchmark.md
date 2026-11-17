---
id: uzux0zcc07q3f0arzlf3v3l
title: Install_benchmark
desc: ''
updated: 1794935002405
created: 1794935002405
---


## 2026.11.17 - pip versus uv for a first install of torchcell

Times `pip install torchcell` against `uv pip install torchcell` in fresh Python 3.13 virtual environments. Standard library only and Python 3.9 compatible, because it starts on a machine with nothing installed: it downloads the uv release binary, lets uv fetch Python 3.13 with `--no-bin`, and keeps the binary, the interpreter, both package caches, the temporary directory and one venv per arm under one work directory. Nothing is written to `~/.cache`, `~/.local` or `/tmp`.

Six arms, run in an order that alternates the tools. An empty cache is the first install on a machine. A filled cache is a later install into another venv.

| arm | command | cache |
|---|---|---|
| `pip_cold` | `pip install <requirement>` | empty |
| `uv_cold` | `uv pip install <requirement>` | empty |
| `pip_warm` | `pip install <requirement>` | filled by `pip_cold` |
| `uv_warm` | `uv pip install <requirement>` | filled by `uv_cold` |
| `uv_cpu_cold` | `uv pip install --torch-backend cpu <requirement>` | empty, its own |
| `pip_cpu_cold` | `pip install --extra-index-url https://download.pytorch.org/whl/cpu <requirement>` | empty, its own |

Each arm records the wall seconds of the install command, the venv's size (`du -sb`), and an import probe (`import torchcell.datasets.ecoli.campos2018`). An arm that fails is recorded and the run continues. A work directory that already holds a venv or a non-empty cache is refused, because a leftover cache would make a cold arm warm.

```bash
python3 experiments/034-showcase-datasets/scripts/install_benchmark.py \
  --work <empty dir> --requirement torchcell==1.8.4
```

### GilaHyper, local disk, torchcell==1.8.4

Source: `experiments/034-showcase-datasets/results/install_benchmark.gilahyper.json` (one run, 2026-11-17, 128 CPUs, uv 0.13.9, pip 26.3). The work directory is ext4 on NVMe by `findmnt`; the result file says `ext2/ext3` because that run asked `stat -f`, which gives every ext filesystem that name. The script now asks `findmnt`.

| arm | install seconds | venv GB |
|---|---:|---:|
| `pip_cold` | 216.4 | 7.56 |
| `uv_cold` | 57.5 | 7.04 |
| `pip_warm` | 104.0 | 7.56 |
| `uv_warm` | 0.6 | 7.04 |
| `uv_cpu_cold` | 33.7 | 2.37 |
| `pip_cpu_cold` | 148.4 | 2.89 |

One run per arm, so these are single measurements with no spread. uv's own log splits `uv_cold` into 48.6 s to download and prepare 256 packages and 0.46 s to install 258 (`work/logs/uv_cold.log`).

### Radiant VM, NFS scratch, torchcell==1.8.4

Source: `experiments/034-showcase-datasets/results/install_benchmark.radiant.json` (one run, 2026-11-17, 16 CPUs, uv 0.13.9, pip 26.3). The work directory is `/mnt/zhao5/mjvolk3/projects/torchcell/tmp/scratch/`, an NFS mount of Taiga (`nfs4` by `findmnt`), because the VM's root disk had 3.2 GB free. The VM ships Python 3.9 and no pip, so uv fetched Python 3.13 first (32 s).

| arm | install seconds | venv GB |
|---|---:|---:|
| `pip_cold` | 1,174.5 | 7.56 |
| `uv_cold` | 136.6 | 7.04 |
| `pip_warm` | 1,036.9 | 7.56 |
| `uv_warm` | 31.0 | 7.04 |
| `uv_cpu_cold` | 193.6 | 2.37 |
| `pip_cpu_cold` | 1,142.2 | 2.89 |

uv's log splits `uv_cold` into 12.9 s to resolve, 104 s to download and prepare 256 packages, and 15.7 s to install 258 (`work/logs/uv_cold.log` on the VM).

### Both hosts side by side

Ratio of pip seconds to uv seconds, from the two result files, one run per arm:

| comparison | GilaHyper, ext4 on NVMe | Radiant, NFS |
|---|---:|---:|
| empty cache | 3.8 | 8.6 |
| filled cache | 173 | 33 |
| CPU build of torch, empty cache | 4.4 | 5.9 |

On NFS a filled cache removed 138 s of pip's 1,175 s, the downloads and the wheel builds. The other 1,037 s remain with every wheel already on disk. On Radiant `uv_cpu_cold` took longer than `uv_cold` (193.6 s against 136.6 s) although it installs 4.7 GB less. That is one run of each, and the cause was not measured.

### The import probe failed in every venv

All six venvs of torchcell 1.8.4 failed the probe with `ModuleNotFoundError: No module named 'torch_scatter'`: `torchcell/data/graph_processor.py` imported `torch_scatter` at 1.8.4, and pip does not install it (`env/requirements_dependent.txt`, PyG wheel index). Commit `46dcb76c5` on main moved that import to `torch_geometric.utils.scatter`. A wheel built from the branch that holds this script, installed into a seventh uv venv with no `torch_scatter`, passed the probe and ran the tutorial to the end ([[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial_page]]).

## 2026.11.17 - Release check: torchcell 1.8.5 from PyPI

torchcell 1.8.5 reached PyPI on 2026-11-17 at 19:16 UTC. The same script, with `--requirement torchcell==1.8.5`, was run again on both hosts. Its result files are named `install_release_check.<host>.json` so the page keeps the six-arm 1.8.4 comparison in one table and lists the release check under it. The script records the filesystem with `findmnt` in these runs, and for each uv arm it records uv's own split of the install (`uv_phases`).

### GilaHyper, local disk, torchcell==1.8.5

Source: `experiments/034-showcase-datasets/results/install_release_check.gilahyper.json` (one run, ext4, uv 0.13.9, pip 26.3).

| arm | install seconds | venv GB | loader import | seconds of the first import |
|---|---:|---:|---|---:|
| `pip_cold` | 201.8 | 7.56 | passes | 6.6 |
| `uv_cold` | 60.5 | 7.10 | passes | 16.5 |
| `pip_warm` | 101.4 | 7.56 | passes | 7.2 |
| `uv_warm` | 0.6 | 7.10 | passes | 17.8 |
| `uv_cpu_cold` | 33.7 | 2.44 | passes | 16.3 |
| `pip_cpu_cold` | 147.8 | 2.89 | passes | 6.4 |

The import probe passes in all six venvs, with torchcell 1.8.5 and torch 2.15.0 (258 distributions with the CUDA build, 239 with the CPU build). The install seconds agree with the 1.8.4 run to within 15 s per arm. uv split `uv_cold` into 8.7 s to resolve, 51.2 s to prepare 256 packages and 0.47 s to install 258.

The first import took 16 to 18 s in the three uv venvs and 6 to 7 s in the three pip venvs. The cause was not measured. Hypothesis (untested): pip compiles bytecode at install time and uv does not unless `--compile-bytecode` is passed, so the uv venvs compile on first import.

### Radiant VM, NFS scratch, torchcell==1.8.5

Source: `experiments/034-showcase-datasets/results/install_release_check.radiant.json` (one run, `nfs4`, uv 0.13.9, pip 26.3). Three arms were run there, since the two pip arms left out take about 19 minutes each on that filesystem and the 1.8.4 run already measured them.

| arm | install seconds | venv GB | loader import | seconds of the first import |
|---|---:|---:|---|---:|
| `pip_cold` | 1,135.6 | 7.56 | passes | 40.2 |
| `uv_cold` | 135.1 | 7.10 | passes | 138.1 |
| `uv_cpu_cold` | 142.8 | 2.44 | passes | 124.3 |

uv split `uv_cold` into 11.6 s to resolve, 1 min 44 s to prepare 256 packages and 15.3 s to install 258. `uv_cpu_cold` took 142.8 s here and 193.6 s in the 1.8.4 run, so a single run on this filesystem can vary by about 50 s.

Install plus first import, from the two columns above: pip 1,176 s, uv 273 s, uv with the CPU build 267 s.
