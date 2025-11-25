---
id: zar6tfmtp3oxpsu0trva5i9
title: Results
desc: ''
updated: 1762296831171
created: 1762296576708
---
Continuation of this plan [[Optimization Plan|dendron://torchcell/experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy.2025.11.3.optimization-plan]]

## 2025.11.04 - Data Loading Optimization Results

### Executive Summary

**Finding**: Preprocessing graph masks to disk **degrades performance** rather than improving it. On-the-fly mask generation with incidence maps remains the fastest approach.

**Root Cause**: LMDB deserialization overhead (5-10ms) exceeds the original mask generation cost (~3ms). Profile analysis confirmed data loading was only 3.0-3.2% of total training time in the baseline approach.

### Performance Comparison Table

| Job ID | Config | Method | Speed (it/s) | Storage | Batch Size | Workers | Data Dir |
|--------|--------|--------|--------------|---------|------------|---------|----------|
| 392 | 065 | Incidence Map (Baseline) | **0.42** | 3.8 GB | 28 | 4 | 001-small-build |
| 428/430 | 073 | Incidence Map + Profiling | 0.26-0.38 | 3.8 GB | 28 | 4 | 001-small-build |
| 447 | 074 | Compressed Indices (LMDB) | 0.24 | 8.8 GB (2.3x) | 28 | 6 | 001-small-build-preprocessed-lazy |
| 456 | 075 | Full Masks (LMDB uint8) | 0.22 | 855 GB (225x) | 28 | 12 | 001-small-build-preprocessed-full-masks |

### Data Loading Methods Tested

1. **Incidence Map with Worker Mask Generation** (Job 392, 428, 430)
   - Zero-copy incidence map lookups (per edge type)
   - Boolean masks created on-the-fly in workers
   - Mask creation: ~3ms per sample
   - **Result**: Fastest approach (0.42 it/s)

2. **Compressed Indices to Disk** (Job 447/074)
   - Saved `perturbation_indices` (int64 arrays) to LMDB
   - Reconstruct boolean masks on load: `torch.zeros(...).scatter_(0, indices, 1)`
   - Mask reconstruction: ~15ms per sample
   - Storage: 8.8 GB
   - **Result**: 43% slower (0.24 it/s vs 0.42 it/s)

3. **Full Masks to Disk** (Job 456/075)
   - Saved complete boolean masks as uint8 to LMDB
   - Direct mask loading, convert uint8→bool on GPU
   - LMDB deserialization: ~8-10ms per sample
   - Storage: ~855 GB
   - **Result**: 48% slower (0.22 it/s vs 0.42 it/s)

### Profile Analysis Summary (Jobs 428/430)

From `profile_analysis_428_detailed.txt` and `profile_analysis_430_detailed.txt`:

```
Operation Category Breakdown:
- Data Loading:        3.0-3.2%  (34-36 seconds)
- Graph Processing:    3.0-3.2%  (34-36 seconds)
- Model Forward:       14.2%
- DDP Communication:   13.3-13.9%
- Optimizer:           24.0-24.1%
- Other/Overhead:      27.3%

Key Finding: "DATA LOADING OPTIMIZED (3.0%)"
→ Data loading is not a bottleneck. No need to save masks to disk.
```

**CPU vs GPU Utilization**: Profile identified 19.3-19.5% CPU overhead vs 4.9% GPU kernel time, suggesting the real bottleneck is CPU-side preprocessing or DDP communication, not data loading.

### Configuration Details

**Common Settings Across All Jobs**:

- Model: HeteroCellBipartiteDangoGI with lazy subgraph representation
- Precision: bf16-mixed
- Strategy: DDP (4 GPUs)
- Dataset: 332,313 samples (001-small-build query)
- Graph: 6,607 genes, 9 edge types + metabolism bipartite

**Differences**:

```text
Job 392 (Baseline):     batch_size=28, num_workers=4,  pin_memory=True
Job 428/430 (Profile):  batch_size=28, num_workers=4,  pin_memory=True,  profiler=enabled
Job 447/074 (Indices):  batch_size=28, num_workers=6,  pin_memory=False
Job 456/075 (Masks):    batch_size=28, num_workers=12, pin_memory=True
```

### Conclusions

1. **Incidence maps are optimal**: Zero-copy lookups with ~3ms mask generation cannot be beaten by disk I/O
2. **Disk I/O adds latency**: Even with LMDB's memory-mapped design, deserialization adds 5-10ms overhead
3. **Storage-performance trade-offs are terrible**:
   - Compressed indices: 2.3x storage (8.8 GB vs 3.8 GB) for 43% slower training
   - Full masks: 225x storage (855 GB vs 3.8 GB) for 48% slower training
4. **Profile-guided optimization**: The 3% data loading time confirmed this wasn't the bottleneck to optimize

### Next Steps

Since preprocessing isn't the answer, we need to investigate the real bottlenecks:

1. **Profile jobs 074 and 075** with PyTorch profiler to compare detailed traces
2. **Investigate DDP communication** (13-14% of time)
3. **Optimize CPU overhead** (19% CPU vs 5% GPU suggests underutilization)
4. **Consider batch size increases** to improve GPU saturation
