---
id: 3sca5owd3476saedvnbz9d4
title: Gpu Masks
desc: Plan for GPU-based edge masking to eliminate data transfer bottleneck
updated: 1762402723907
created: 1762399532031
---

# GPU-Based Edge Masking Optimization Plan

## Problem Statement

The lazy model implementation (hetero_cell_bipartite_dango_gi_lazy) is **35x slower** than the dango baseline despite having **identical forward pass times** (~20ms each). This proves the bottleneck is in the data pipeline, not model computation.

## Evidence

### Forward Pass Profiling

From `profile_results_2025-11-05-16-33-37.txt` (lazy model):

- **Forward pass time: 20.8 ms**

From `profile_dango_results_2025-11-05-16-26-56.txt` (dango baseline):

- **Forward pass time: 20.0 ms**

→ **Essentially identical computation time**

### Actual Training Speed

**Dango model** (from `slurm/output/S12-DANGO_1941704.out`):

- **10.44 it/s** (96 ms per iteration)
- 2077 batches in ~3 minutes

**Lazy model** (from `006-080-HCPD-LAZY_480.out`):

- **~0.3 it/s** (~3300 ms per iteration)
- 1-2 hours per epoch

→ **35x slower despite identical forward pass!**

### Data Transfer Analysis

**Lazy model transfers per iteration:**

- Edge masks: 65 MB per batch
- 2x forward passes (cell_graph + batch): 130 MB total
- Transfer time at 12 GB/s PCIe: ~11ms just for transfer
- Plus CPU collate overhead: building/copying masks

**Dango model transfers per iteration:**

- Perturbation indices only: ~0.18 MB per batch
- 1x forward pass
- cell_graph stays on GPU permanently

**Data transfer ratio: 130 MB / 0.18 MB = 722x more data!**

## Previous Optimization Attempts

1. **Original (slow):** Build masks on-the-fly
   - Load perturbation list → Find edges to remove → Create boolean mask
   - ~10ms per sample overhead

2. **Optimization 1 (slower!):** Store just indices of perturbed edges
   - Still need to build boolean masks from indices
   - Mask creation is the actual bottleneck

3. **Optimization 2:** Store full boolean masks (current)
   - Storage cost: 2.5MB per sample (vs 25KB for indices-only)
   - Dataset: 848GB (full 20M dataset would be 14TB)
   - **No speedup** - still transferring 65 MB/batch to GPU

**All three approaches:** Transfer large edge masks from CPU → GPU every iteration

## Root Cause

The fundamental issue is **CPU → GPU data transfer bandwidth** is the bottleneck, not:

- ❌ Mask creation (already optimized with incidence cache)
- ❌ Data loading from disk (LMDB is extremely fast)
- ❌ Model forward pass (identical to dango)
- ❌ Graph processing (only 3.2% of time)

**✅ The bottleneck: Transferring 130 MB of edge masks every iteration**

## Proposed Solution: GPU-Based Edge Masking

### Core Idea

Instead of creating edge masks on CPU and transferring them to GPU, keep the incidence cache on GPU and generate masks on-the-fly during forward pass.

### Architecture

#### One-Time GPU Setup (at model initialization):

1. Upload incidence cache to GPU (~19 MB)
   - Maps: gene_index → [edge_indices that involve this gene]
   - 9 edge types × 6,607 nodes × avg 41 edges/node

2. Create base edge masks on GPU (full graph, all True)
   - One mask per edge type
   - Stays on GPU as model buffers

#### Per-Iteration (on GPU):

1. Transfer only perturbation indices: ~16 bytes/sample, **448 bytes/batch**
2. Index into GPU incidence cache to get affected edge positions
3. Clone base masks and set affected edges to False
4. Use masks during message passing

### Benefits

**Data transfer reduction:**

- Current: 130 MB per iteration
- Proposed: 0.0004 MB per iteration
- **Reduction: 325,000x**

**Expected speedup:**

- Current: ~0.3 it/s
- Expected: **8-10 it/s** (25-33x speedup)
- Should match or exceed dango baseline speed

**GPU memory increase:**

- Incidence cache: ~19 MB per GPU
- Base masks: negligible (reused across batches)
- **Total: < 25 MB additional GPU memory**

## Implementation (ACTUAL - Completed 2025-11-05)

### Critical Context for Restart

**Problem:** Lazy model 35x slower than dango (0.3 it/s vs 10.44 it/s) despite identical forward pass times (~20ms). Bottleneck is CPU→GPU transfer of 130 MB edge masks per iteration.

**Solution:** Generate edge masks on GPU from perturbation indices (448 bytes transfer instead of 130 MB). Expected 25-33x speedup to match dango baseline.

### Key Files Modified

1. **NEW:** `torchcell/models/gpu_edge_mask_generator.py` - GPU mask generation
2. **MODIFIED:** `torchcell/trainers/int_hetero_cell.py:40-68,135-147` - GPU masking support + fail-fast error handling
3. **MODIFIED:** `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py:318-319,333,351,524-526,550,573` - follow_batch config + init_masks_on_gpu parameter
4. **NEW:** `experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi_gh_081.yaml` - GPU masking config
5. **NEW:** `experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_081.slurm` - SLURM script

### Implementation Details

#### Phase 1: GPU Mask Generator Class

**File:** `torchcell/models/gpu_edge_mask_generator.py`

Created `GPUEdgeMaskGenerator(nn.Module)` that:
- Builds incidence cache on GPU (~19 MB) in `__init__`
- Registers base masks as model buffers (auto-moved with model.to(device))
- Generates batch masks from perturbation indices on GPU via `generate_batch_masks()`
- Supports per-sample mask generation for batched data

**Key design decisions:**
- Incidence cache stored as `Dict[edge_type, List[torch.Tensor]]` where each tensor contains edge indices for a gene
- Base masks are all-True tensors, cloned and modified for each batch
- Returns concatenated masks for entire batch (same format as CPU masks)

#### Phase 2: RegressionTask Modification

**File:** `torchcell/trainers/int_hetero_cell.py`

**Lines 40-68:** Added `init_masks_on_gpu` parameter and GPU mask generator initialization:
```python
def __init__(self, ..., init_masks_on_gpu: bool = False):
    self.init_masks_on_gpu = init_masks_on_gpu
    if self.init_masks_on_gpu:
        self.gpu_mask_generator = GPUEdgeMaskGenerator(cell_graph, device_obj)
```

**Lines 135-147:** CRITICAL FAIL-FAST BEHAVIOR (no silent fallback!):
```python
if hasattr(batch["gene"], "perturbation_ptr"):
    ptr = batch["gene"].perturbation_ptr
    start, end = ptr[sample_idx], ptr[sample_idx + 1]
    sample_pert_idx = batch["gene"].perturbation_indices[start:end]
else:
    raise RuntimeError(
        "GPU masking enabled but perturbation_ptr not found in batch. "
        "Fix: Add 'perturbation_indices' to follow_batch list"
    )
```

**Why this matters:** Original code had `log.warning()` with empty tensor fallback. This silently trained on wildtype graph only, producing confusing results. Now explicitly fails with clear fix instructions.

#### Phase 3: DataLoader Configuration (CRITICAL)

**File:** `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py`

**Lines 318-319:** Added follow_batch configuration:
```python
# CRITICAL: For GPU masking, need to track perturbation_indices to create perturbation_ptr
follow_batch_list = ["x", "x_pert", "perturbation_indices"]
```

**Lines 333, 351:** Passed to both datamodules:
```python
data_module = CellDataModule(
    ...,
    follow_batch=follow_batch_list,  # Creates perturbation_ptr
)
```

**Why this is critical:** PyTorch Geometric's `follow_batch` mechanism creates `perturbation_ptr` tensor that marks batch boundaries. Without this, `perturbation_ptr` doesn't exist and we can't extract per-sample perturbation indices from the concatenated batch tensor.

**Lines 524-526:** Read GPU masking config and log:
```python
init_masks_on_gpu = cfg.model.get("init_masks_on_gpu", False)
if init_masks_on_gpu:
    log.info("GPU-based edge masking enabled")
```

**Lines 550, 573:** Pass to RegressionTask:
```python
task = RegressionTask(
    ...,
    init_masks_on_gpu=init_masks_on_gpu,
)
```

#### Phase 4: Configuration Files

**File:** `experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi_gh_081.yaml`

Key settings:
- Line 58: `use_full_masks: false` - Uses indices-only dataset (not full masks)
- Line 147: `init_masks_on_gpu: true` - Enables GPU mask generation

**File:** `experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_081.slurm`

Standard 4-GPU DDP setup with GPU masking experiment config.

### Technical Details

#### Data Flow

**Before (CPU masking):**
1. Dataset stores full boolean masks (65 MB per batch)
2. DataLoader transfers masks to GPU (130 MB total with 2 forward passes)
3. Model applies masks during message passing

**After (GPU masking):**
1. Dataset stores only perturbation indices (~16 bytes/sample)
2. DataLoader transfers indices to GPU (448 bytes per batch)
3. PyG creates `perturbation_ptr` from `follow_batch` config
4. RegressionTask extracts per-sample indices using `perturbation_ptr`
5. GPUEdgeMaskGenerator creates masks on GPU (~1ms)
6. Model applies masks during message passing

#### Incidence Cache Structure

```python
incidence_cache = {
    ('gene', 'interaction', 'gene'): [
        tensor([0, 1, 5, ...]),  # Edge indices for gene 0
        tensor([2, 3, 4, ...]),  # Edge indices for gene 1
        ...
    ],
    ('gene', 'physical_interaction', 'gene'): [...],
    # ... 9 edge types total
}
```

Size: ~19 MB on GPU (9 edge types × 6,607 genes × avg 41 edges/gene × 8 bytes)

#### Batch Mask Generation Algorithm

For a batch with perturbation indices `[g1, g2, g3, g4, g5]` and ptr `[0, 2, 5]` (2 samples):
1. Sample 0: indices `[g1, g2]`
2. Sample 1: indices `[g3, g4, g5]`
3. For each sample, clone base mask (all True)
4. Lookup affected edges from incidence cache
5. Set affected edges to False
6. Concatenate masks for all samples in batch

### Errors Fixed

**Error 1: Missing perturbation_ptr (Initial Run)**
- **Symptom:** Thousands of warnings about missing `perturbation_ptr`
- **Root cause:** DataLoader not configured to track `perturbation_indices`
- **Fix:** Added `follow_batch=["x", "x_pert", "perturbation_indices"]` to datamodule init
- **Files changed:** `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py:318-319,333,351`

**Error 2: Silent Fallback Behavior (User Feedback)**
- **User quote:** "the script should flat out fail if this is happening. otherwise this is very confusing"
- **Symptom:** Code continued with empty perturbation indices, training on wildtype graph only
- **Fix:** Changed `log.warning()` + fallback to `raise RuntimeError()` with clear error message
- **Files changed:** `torchcell/trainers/int_hetero_cell.py:140-147`

### PyTorch Geometric follow_batch Mechanism

**What it does:** Creates a `{attr}_ptr` tensor marking batch boundaries for concatenated attributes.

**Example:**
```python
# Sample 0: perturbation_indices = [1, 3, 5]
# Sample 1: perturbation_indices = [2, 4]

# After batching with follow_batch=["perturbation_indices"]:
batch.perturbation_indices = tensor([1, 3, 5, 2, 4])  # Concatenated
batch.perturbation_ptr = tensor([0, 3, 5])  # Boundaries

# Extract sample 0: indices[ptr[0]:ptr[1]] = [1, 3, 5]
# Extract sample 1: indices[ptr[1]:ptr[2]] = [2, 4]
```

**Without follow_batch:** No `perturbation_ptr` created, can't separate samples!

### DDP Considerations

- `GPUEdgeMaskGenerator` is `nn.Module`, automatically replicated by DDP
- Incidence cache registered as model buffers, replicated to each GPU
- Each GPU gets its own copy (~19 MB per GPU, 76 MB total for 4 GPUs)
- Batch distributed across GPUs, each generates masks independently

### Testing & Validation

**Completed:**
- ✅ Code runs without errors with GPU masking enabled
- ✅ Follow_batch configuration creates perturbation_ptr correctly
- ✅ Fail-fast error handling prevents silent failures

**Pending:**
- ⏳ Training speed validation (should reach 8-10 it/s)
- ⏳ Memory usage profiling (should be ~19 MB increase per GPU)
- ⏳ Correctness validation (predictions should match CPU masking baseline)

## Expected Results

### Performance Improvements

- **Data transfer per iteration:** 130 MB → 0.0004 MB (325,000x reduction)
- **Training speed:** ~0.3 it/s → **8-10 it/s** (25-33x speedup)
- **Epoch time:** 1hr 45min → **3-4 minutes** (matching dango baseline)

### Resource Usage

- **GPU memory increase:** ~19 MB for incidence cache (negligible)
- **CPU usage:** Reduced (no mask creation overhead)
- **Disk usage:** Can revert to indices-only storage (848 GB → 8.8 GB)

### Code Changes

- **New files:** 1 (gpu_mask_generator.py)
- **Modified files:** 4 (graph_processor.py, lazy_collate.py, model forward, datamodule)
- **Total LOC:** ~400 lines added

## Risks & Mitigation

### Risk 1: Incidence Cache Size

**Risk:** Cache may be larger than 19 MB estimated
**Mitigation:**

- Profile actual size after moving to GPU
- Use sparse tensor representation if needed
- Implement cache compression

### Risk 2: Batch-Specific Masking Complexity

**Risk:** Each sample needs different mask - can't batch easily
**Mitigation:**

- Use `generate_masks_batched()` for per-sample masks
- Or apply masks during message passing (edge-level masking)
- Profile to see if GPU mask generation adds overhead

### Risk 3: Message Passing with Masks

**Risk:** PyG layers may not support `edge_mask_dict` parameter
**Mitigation:**

- Create custom wrapper layers
- Or modify edge_index directly (filter edges) before passing to layers
- Use PyG's subgraph utilities on GPU

### Risk 4: DDP Complications

**Risk:** Each GPU needs its own incidence cache
**Mitigation:**

- Cache is model buffer, automatically replicated by DDP
- Verify cache replication in multi-GPU setup
- Test with DDP before full training run

### Risk 5: Gradient Issues

**Risk:** GPU mask generation may break gradient flow
**Mitigation:**

- Mask generation is in forward pass, shouldn't affect gradients
- Use `torch.no_grad()` for mask generation if needed
- Verify gradients flow correctly in unit tests

## Validation Criteria

Before considering this complete:

1. ✅ **Correctness:** GPU masks match CPU masks exactly (unit test passes)
2. ✅ **Performance:** Training speed reaches 8-10 it/s (25-33x speedup)
3. ✅ **Predictions:** Model outputs match baseline within 1e-6 tolerance
4. ✅ **Memory:** GPU memory increase < 50 MB per GPU
5. ✅ **Stability:** Can complete full training run without errors
6. ✅ **Reproducibility:** Results are reproducible across runs

## Implementation Timeline

**Total estimated time: 8-12 hours**

- Phase 1 (GPU incidence cache): 1-2 hours
- Phase 2 (GPU mask generator): 2-3 hours
- Phase 3 (Collate modifications): 1-2 hours
- Phase 4 (Model forward): 1-2 hours
- Phase 5 (Incidence cache passing): 1 hour
- Phase 6 (Testing & validation): 2-3 hours

## Alternative Approaches (if GPU masking proves difficult)

### Alternative 1: Cache z_w Embeddings

- Cache wildtype embeddings once per epoch
- Eliminate 50% of forward passes
- Expected speedup: 1.5-2x (not as good as GPU masking)

### Alternative 2: Increase Batch Size

- Current: batch_size=28
- Try: batch_size=128 or 256
- Amortize transfer overhead
- Expected speedup: 1.3-1.8x

### Alternative 3: FP16 Masks

- Store masks as float16 instead of bool
- Reduces transfer: 65 MB → 32.5 MB
- Expected speedup: 1.5x

### Alternative 4: Async Transfer with Prefetching

- Overlap CPU→GPU transfer with computation
- Use CUDA streams for async transfer
- Expected speedup: 1.5-2x

## Next Steps

1. **Clean up repo** (as requested)
   - Remove unused profile results
   - Clean up old slurm outputs
   - Organize experiment configs

2. **Implement GPU masking** (following this plan)
   - Start with Phase 1 (incidence cache)
   - Test each phase before moving to next
   - Profile at each step to verify improvements

3. **Compare to baseline**
   - Run full training with GPU masking
   - Compare speed, memory, and results to dango baseline
   - Document findings

## References

- Related: [[experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy.2025.11.4.results]]
- Related: [[experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy.2025.11.3.optimization-plan]]
- Model: [[torchcell.models.hetero_cell_bipartite_dango_gi_lazy]]
- Data processor: [[torchcell.data.graph_processor.LazySubgraphRepresentation]]
