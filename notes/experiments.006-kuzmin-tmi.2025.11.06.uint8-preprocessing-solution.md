# Final Preprocessing Solution: UINT8 Full Masks

## Executive Summary

After testing and analysis, we've determined that storing full masks as UINT8 provides the optimal balance of storage efficiency and performance.

## Test Results (1000 samples verified)

- **UINT8**: 847.7 GB projected (2.551 MB/sample)
- **BFLOAT16**: 1693.2 GB projected (5.095 MB/sample)
- **Decision**: Use UINT8 - saves 845 GB with negligible conversion overhead

## Performance Analysis

### Why UINT8 is Optimal

1. **Conversion overhead: <0.001%** of batch time
   - bool→bf16 happens on GPU (7-34 microseconds for 68.6M edges)
   - Your batch time: 2630ms (at 0.38 it/s)
   - Conversion is 0.0003% of total time

2. **Real bottlenecks** (from profiling):
   - Framework overhead: 27%
   - Optimizer: 24%
   - DDP communication: 13.9%
   - GPU operations: 4.9% (ALL operations combined)
   - Data loading: 0.0%

3. **Memory bandwidth advantage**:
   - UINT8: 847.7 GB transferred CPU→GPU
   - BFLOAT16: 1693.2 GB transferred (2x more)
   - PCIe bandwidth saved with UINT8

## Implementation

### Files Updated

1. **`preprocess_lazy_dataset_full_masks.py`**
   - Stores masks as torch.uint8
   - LMDB map_size: 1.1TB
   - Expected: ~847.7GB for 332K samples

2. **`neo4j_preprocessed_cell_full_masks.py`**
   - Loads uint8 masks
   - Converts to bool for use
   - Conversion pipeline clearly documented

3. **`gh_hetero_cell_bipartite_dango_gi_lazy-ddp_074_full_masks.slurm`**
   - Updated storage estimates
   - Clear performance expectations

## Usage Instructions

### Step 1: Run Preprocessing

```bash
cd /home/michaelvolk/Documents/projects/torchcell
python experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset_full_masks.py
```

- Duration: ~50 minutes (one-time)
- Output: 847.7GB LMDB at `/scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-full-masks/`

### Step 2: Submit Training Job

```bash
sbatch experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_074_full_masks.slurm
```

## Expected Performance

- **Training speed**: 0.38+ it/s (matching or exceeding on-the-fly)
- **Mask loading**: <0.1ms (vs 15ms reconstruction with compressed indices)
- **Storage**: 847.7GB (vs 3GB compressed, but eliminates reconstruction overhead)

## Conversion Pipeline

1. **Storage**: UINT8 in LMDB (1 byte per boolean)
2. **Loading**: uint8→bool conversion (CPU)
3. **Transfer**: bool tensors to GPU
4. **Training**: bool→bf16 in MaskedGINConv (GPU, <0.001% overhead)

## Why Not BFLOAT16?

- **2x storage** (1693.2 GB) for no measurable speedup
- **Conversion is not the bottleneck** - framework overhead is (27%)
- **GPU already 80% idle** - eliminating conversion won't help
- **Flexibility lost** - UINT8 works with any precision

## Conclusion

UINT8 full masks provide the best solution:

- Eliminates 15ms reconstruction overhead
- Minimal storage (847.7 GB fits easily)
- Negligible conversion overhead (<0.001%)
- Optimal memory bandwidth usage
