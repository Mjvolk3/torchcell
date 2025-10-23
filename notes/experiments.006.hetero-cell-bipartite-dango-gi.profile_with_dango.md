---
id: tuiln1fgbwzdsumobnm0fjy
title: profile with dango
desc: ''
updated: 1761084598861
created: 1761083796340
---
## Problem

- **Dango model**: 600 epochs in 2 days
- **HeteroCell model**: 160 epochs in 14 days (6x slower!)
- Need to identify bottlenecks

## Quick Start

From torchcell root directory:

```bash
# Submit profiling job to SLURM
sbatch experiments/006-kuzmin-tmi/scripts/profile_comparison.slurm
```

## What It Does

1. Profiles HeteroCell model (1 epoch, 1000 samples)
2. Profiles Dango model (1 epoch, 1000 samples)
3. Compares results and identifies bottlenecks

## Files

```
torchcell/
└── experiments/006-kuzmin-tmi/
    ├── scripts/
    │   ├── profile_comparison.slurm            # Main profiling SLURM script
    │   ├── dango.py                            # Fast model
    │   ├── hetero_cell_bipartite_dango_gi.py   # Slow model (6x slower)
    │   └── compare_profiler_outputs.py         # Analysis script
    └── slurm/
        └── output/                              # SLURM job outputs
```

## Output Location

Profile outputs saved to:

```
/scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/
```

## View Results

### TensorBoard (if port 6006 forwarded)

```bash
tensorboard --logdir=/scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output
```

### Chrome Tracing

1. Download JSON files from profiler_output directory
2. Open chrome://tracing
3. Load the JSON files

## What to Look For

- Subgraph operations taking excessive time
- GPU utilization (~40% for HeteroCell)
- Memory transfer bottlenecks
- Graph construction overhead
