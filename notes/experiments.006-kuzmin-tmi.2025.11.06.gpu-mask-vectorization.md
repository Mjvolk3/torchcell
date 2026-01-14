---
id: rvv2l3y6c7dhq1sneywz2wk
title: Gpu Mask Vectorization
desc: ''
updated: 1767845278067
created: 1767845278067
---

## GPU Mask Generation Vectorization - Implementation Summary

**Date:** 2025-11-06
**Experiment:** 006-082-HCPD-VECTORIZED

Comprehensive implementation of vectorized GPU mask generation eliminating 864 GPU→CPU synchronizations per batch by replacing Python loops with batch tensor operations, achieving an expected 10-14x training speedup from 0.42 it/s to 4-6 it/s.

### Problem Identified

The GPUEdgeMaskGenerator had **864 GPU→CPU synchronizations per batch** caused by `.item()` calls in nested Python loops, creating a severe bottleneck that limited training to 0.42 it/s.

### Key Changes Implemented

#### 1. GPUEdgeMaskGenerator Vectorization (`gpu_edge_mask_generator.py`)

##### Added Methods:

- **`_build_vectorized_incidence_tensors()`** (lines 120-164)
  - Converts list-based incidence cache to padded tensors
  - Enables batch tensor indexing without Python loops
  - Pre-computes validity masks for efficient filtering

- **`generate_batch_masks_vectorized()`** (lines 254-350)
  - Zero `.item()` calls (previously 864 per batch)
  - Zero Python loops over samples
  - Uses batch tensor operations throughout
  - Key optimizations:
    - Concatenates all perturbation indices upfront
    - Uses `torch.repeat_interleave` for batch assignment
    - Applies scatter operations for mask updates
    - Filters padded positions with boolean indexing

#### 2. Trainer Optimization (`int_hetero_cell.py`)

##### Modified extraction of perturbation indices (lines 130-165):

- Removed `.item()` calls from ptr indexing (lines 137-138 removed)
- Vectorized validation checks (lines 144-154)
- Only uses `.item()` in error messages (not hot path)
- Uses tensor slicing with tensor indices (lines 160-164)

##### Updated to use vectorized method (lines 174-177):

- Changed from `generate_batch_masks()` to `generate_batch_masks_vectorized()`

#### 3. Experiment Configuration

##### Created `gh_hetero_cell_bipartite_dango_gi_lazy-ddp_082.slurm`:

- Job name: 006-082-HCPD-VECTORIZED
- References config 082
- Documents expected speedup in comments

##### Created `hetero_cell_bipartite_dango_gi_gh_082.yaml`:

- Added tags: `[vectorized, zero_syncs]`
- Documents optimization details in comments
- Keeps `init_masks_on_gpu: true`

#### 4. Testing Infrastructure

##### Created `test_vectorized_gpu_masks.py`:

- Verifies correctness: vectorized produces identical masks
- Benchmarks performance improvement
- Tests multiple batch sizes and perturbation patterns

### Expected Performance Impact

#### Before Optimization:

- **864 GPU→CPU syncs per batch**
- ~432-864ms overhead per batch
- Training speed: 0.42 it/s

#### After Optimization:

- **0 GPU→CPU syncs per batch**
- Expected mask generation: 10-20x faster
- Expected training speed: **4-6 it/s** (10-14x improvement)

### How to Run

1. **Test correctness locally:**

   ```bash
   python experiments/006-kuzmin-tmi/scripts/test_vectorized_gpu_masks.py
   ```

2. **Submit training job:**

   ```bash
   sbatch experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_082.slurm
   ```

3. **Monitor performance:**
   - Check WandB for it/s metrics
   - Look for "GPUEdgeMaskGenerator initialized" in logs
   - Verify no `.item()` warnings in profiler

### Technical Details

#### Why Not GPU Filtering (torch.isin approach)?

With only ~3 genes perturbed out of 6607 (<1% edges removed):

- Filtering cost: 2×E operations (`torch.isin` + gather)
- Masking cost: 1×E operations (multiply)
- Conv savings: negligible (99% edges remain)

The 10-20x speedup comes from removing `.item()` syncs, not from mask vs filter choice.

#### Memory Overhead

- Padded incidence tensors: ~30-40MB additional GPU memory
- Acceptable tradeoff for 10-20x speedup
- Pre-allocated, reused across batches

### Next Steps if Performance Still Below Target

If training is still below 10 it/s after this optimization:

1. **Profile remaining bottlenecks:**
   - Model forward pass (batched attention)
   - DDP communication patterns
   - Data loading pipeline

2. **Consider GPU filtering approach:**
   - If profiling shows mask multiplication is expensive
   - Implement side-by-side comparison

3. **Other optimizations:**
   - Reduce hidden dimensions
   - Optimize attention layers
   - Use torch.compile
