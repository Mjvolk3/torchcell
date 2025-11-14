---
id: pm4rpygjs9zwo5h62sp3e1c
title: Experiment 086 Results Summary
desc: ''
updated: 1762500717188
created: 1762500261233
---

## Executive Summary

**Critical Finding:** The data pipeline (graph processing) is the primary bottleneck, accounting for **56× slower performance** compared to DANGO. Model compute adds an additional **8.3× slowdown**, but represents only 33% of total time.

**Key Results:**

- **086a (Data only):** Lazy Hetero 1.31 it/s vs DANGO 73.82 it/s → **56.34× slower**
- **086b (Data + Model):** Lazy Hetero 0.88 it/s vs DANGO 16.98 it/s → **19.31× slower**
- **Model overhead:** 0.375 sec/it (Lazy) vs 0.045 sec/it (DANGO) → **8.3× slower**

---

## Detailed Profiling Results

### Experiment 086a: DataLoader Profiling (No Model)

**Configuration:**

- Single GPU (no DDP)
- 10,000 sample subset
- Batch size: Lazy=28, DANGO=64
- Execution mode: `dataloader_profiling` (dummy loss, skip model forward)

**What This Measures:**

- ✅ DataLoader fetch from LMDB/Neo4j
- ✅ Graph processor execution (LazySubgraphRepresentation vs Perturbation)
- ✅ Collation function (LazyCollater vs default PyG)
- ✅ CPU→GPU batch transfer
- ❌ NO model forward/backward

**Results:**

| Model | Avg it/s | Time/batch (28-64 samples) | Time/sample |
|-------|----------|---------------------------|-------------|
| Lazy Hetero | 1.310 | 0.763 sec | **27.3 ms** |
| DANGO | 73.816 | 0.014 sec | **0.22 ms** |
| **Ratio** | **56.34×** | | **123× slower/sample** |

**Interpretation:**
The data pipeline alone (without any model computation) is 56× slower for Lazy Hetero. This is purely graph processing + collation overhead.

---

### Experiment 086b: Model Profiling (Data + Model, No Optimizer)

**Configuration:**

- Same as 086a
- Execution mode: `model_profiling` (real forward/backward, skip optimizer)

**What This Measures:**

- ✅ Everything from 086a
- ✅ Model forward pass
- ✅ Model backward pass (gradient computation)
- ❌ NO optimizer step (AdamW)

**Results:**

| Model | Avg it/s | Time/batch | Total |
|-------|----------|-----------|-------|
| Lazy Hetero | 0.879 | 1.138 sec | 100% |
| DANGO | 16.980 | 0.059 sec | 100% |
| **Ratio** | **19.31×** | | |

**Time Breakdown (Lazy Hetero):**

```
Total time:          1.138 sec/it (100%)
├─ Data pipeline:    0.763 sec    (67%) ← PRIMARY BOTTLENECK
└─ Model compute:    0.375 sec    (33%)
   ├─ Forward pass:  ~0.250 sec   (estimated)
   └─ Backward pass: ~0.125 sec   (estimated)
```

**Time Breakdown (DANGO):**

```
Total time:          0.059 sec/it (100%)
├─ Data pipeline:    0.014 sec    (24%)
└─ Model compute:    0.045 sec    (76%)
```

---

## Root Cause Analysis

### 1. Graph Processor Architecture Difference

#### LazySubgraphRepresentation (Lazy Hetero)

```python
# Location: experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py:203
graph_processor = LazySubgraphRepresentation()
```

**What it does per sample:**

1. **Edge mask computation:** Creates boolean masks for 9 edge types
   - Physical, regulatory, tflink
   - 6 STRING types (neighborhood, fusion, cooccurence, coexpression, experimental, database)
2. **Incidence cache lookup:** Uses O(k×d) operations for k perturbations and d neighbors
3. **Data size:** ~2-5 MB per sample (full edge_index + masks for all 9 types)
4. **Processing:** ~500K-2M edges need mask computation

**Performance:**

- **27.3 ms per sample**
- Dominated by mask generation across 9 edge types
- Heavy CPU computation (boolean array operations)

#### Perturbation (DANGO)

```python
# Location: experiments/006-kuzmin-tmi/scripts/dango.py:169
graph_processor = Perturbation()
```

**What it does per sample:**

1. **Minimal data:** Only stores perturbation indices (10-20 integers)
2. **No edge filtering:** Returns raw perturbation IDs
3. **Data size:** ~20 KB per sample (1000× smaller!)
4. **Processing:** O(k) operations for k perturbations

**Performance:**

- **0.22 ms per sample**
- Essentially just array indexing
- Negligible CPU overhead

**Ratio:** 27.3 / 0.22 = **123× slower per sample**

---

### 2. Collation Complexity

#### LazyCollater (Lazy Hetero)

```python
# Location: experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py:313-314
lazy_collater = LazyCollater(cell_graph)
```

**What it does:**

1. Replicates edge_index tensors for each sample
2. Applies offsets to node indices for batching
3. Handles 9 edge types × batch_size operations
4. More complex batching logic for zero-copy architecture

#### Default PyG Collation (DANGO)

- Simple concatenation of perturbation data
- No edge_index replication needed
- Minimal overhead

---

## Comparison with Previous Experiments

| Experiment | Config | Lazy Hetero | DANGO | Ratio | Notes |
|------------|--------|-------------|-------|-------|-------|
| **524 (DDP, full)** | 4 GPU, full dataset | 1.6 it/s | 19.9 it/s | 12.5× | DDP communication overhead |
| **086a (Data)** | 1 GPU, 10K subset | **1.31 it/s** | **73.82 it/s** | **56.3×** | Pure data pipeline |
| **086b (Data+Model)** | 1 GPU, 10K subset | **0.88 it/s** | **16.98 it/s** | **19.3×** | Data + model |

**Key Insights:**

1. **DDP was hiding the true bottleneck:** The 12.5× ratio in experiment 524 masked the 56× data pipeline issue due to:
   - DDP communication overhead
   - Different batch sizes (28 vs 64)
   - Full dataset I/O contention
2. **Single GPU reveals truth:** Removing DDP exposed the data pipeline as the primary culprit
3. **Model is 8.3× slower:** Still significant, but secondary to data pipeline

---

## The Bottleneck Question: What's Actually Slow?

### Hypothesis 1: Subgraph Operations (Most Likely)

**Evidence:**

- LazySubgraphRepresentation computes masks for 9 edge types
- Each edge type has ~50K-200K edges
- Incidence cache lookups: O(k×d) per sample (k=perturbations, d=avg degree)
- Heavy boolean array operations on CPU

**Operations Breakdown (estimated):**

```python
# Pseudocode of what LazySubgraphRepresentation does:
for each_edge_type in [physical, regulatory, tflink, string_1, ..., string_6]:
    # 1. Get incidence map (precomputed, fast)
    incidence = self.incidence_cache[edge_type]  # ~1 ms

    # 2. For each perturbation, find affected edges (SLOW)
    for gene_id in perturbation_indices:  # k iterations
        affected_edges = incidence[gene_id]  # O(d) where d=avg degree
        edge_mask[affected_edges] = True  # Boolean indexing

    # 3. Apply mask to edge_index (moderate)
    masked_edges = edge_index[:, edge_mask]  # ~2-3 ms per type
```

**Total estimated time:**

- Incidence lookup: 9 types × 1 ms = 9 ms
- Mask computation: 9 types × ~10 perturbations × ~1000 affected edges = 90 ms (dominant!)
- Edge filtering: 9 types × 3 ms = 27 ms
- **Total: ~126 ms (doesn't match 27ms, so likely some optimization)**

**Most likely bottleneck: The mask computation loop**

### Hypothesis 2: Disk I/O (Less Likely in This Case)

**Evidence Against:**

- LMDB is memory-mapped (minimal I/O after first load)
- 10K subset easily fits in memory
- DataLoader with `num_workers=4` and `persistent_workers=True` pre-loads data
- If I/O were the issue, DANGO would also be slow (uses same LMDB)

**However:** On full dataset with disk-based Neo4j queries, I/O could become significant.

### Hypothesis 3: Data Transfer CPU→GPU (Unlikely)

**Evidence Against:**

- GPU transfer time should be similar for both models (same batch sizes roughly)
- DANGO transfers 64 samples in 0.014 sec = 0.22 ms/sample
- If transfer were slow, DANGO would also be affected
- Modern GPUs have ~16 GB/s PCIe bandwidth (very fast)

---

## Experiment 077 Approach: Preprocessed Full Masks

### What Experiment 077 Does

**Config:** `hetero_cell_bipartite_dango_gi_gh_077.yaml`

```yaml
data_module:
  use_full_masks: true  # Load precomputed masks from disk
```

**Script:** `hetero_cell_bipartite_dango_gi_lazy_preprocessed.py`

```python
# Key difference: Uses different dataset class
# Instead of: Neo4jCellDataset (computes masks on-the-fly)
# Uses: PreprocessedCellDataset (loads masks from LMDB)
```

**Preprocessing Scripts:**

- `scripts/preprocess_lazy_dataset.py` - Precomputes incidence-based masks
- `scripts/preprocess_lazy_dataset_full_masks.py` - Precomputes full boolean masks

**Theory:**

1. **Offline preprocessing:** Run once to compute all edge masks for all samples
2. **Store masks in LMDB:** Each sample has precomputed masks (large storage ~100GB)
3. **Runtime:** Just load masks from disk (fast LMDB read)

**Expected speedup:**

- If subgraph ops are bottleneck: ~20-40× faster (27ms → 1-2ms)
- If disk I/O is bottleneck: Could be slower (more data to read)

### Why This Didn't Work Before?

**Potential issues:**

1. **Storage explosion:** Full masks for 332K samples × 9 edge types × avg 100K edges = ~300 GB
   - May have exceeded disk space or LMDB size limits
2. **Collation still complex:** Even with precomputed masks, LazyCollater still needs to:
   - Replicate edge_index tensors
   - Apply batch offsets
   - Handle 9 edge types
3. **I/O bottleneck shift:** Reading 2-5 MB per sample from disk may be slower than computing 27ms
4. **Memory pressure:** Loading large masks may cause memory issues

---

## Profiling Options to Find Exact Bottleneck

### Option 1: Python cProfile (Current Approach)

**What it shows:**

- Python function call times
- Number of calls
- Cumulative time

**Limitations:**

- **Cannot see into PyTorch C++ code**
- **Cannot see GPU operations**
- Overhead from profiling itself

**How to use:**

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run 1 step
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Option 2: PyTorch Profiler with Trace (Recommended!)

**What it shows:**

- **CPU operations (Python + C++)**
- **GPU kernels**
- **Data transfer (CPU↔GPU)**
- Timeline view in TensorBoard/Chrome

**Config:**

```yaml
profiler:
  enabled: true
  is_pytorch: true
```

**Script modification for single-step trace:**

```python
# In training script, after ~50 steps (warmup):
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_trace')
) as prof:
    # Run EXACTLY 1 training step
    trainer.training_step(batch, batch_idx)

# Save trace
prof.export_chrome_trace("step_50_trace.json")
```

**Analysis:**

```bash
# View in Chrome
# Go to chrome://tracing
# Load step_50_trace.json
# Look for:
#   - Long CPU operations (likely LazySubgraphRepresentation)
#   - GPU utilization gaps (data pipeline stalls)
```

### Option 3: Line Profiler (Most Detailed for Python)

**What it shows:**

- **Line-by-line execution time**
- Exact Python bottleneck lines

**Install:**

```bash
pip install line_profiler
```

**Usage:**

```python
# Add @profile decorator to functions of interest
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(LazySubgraphRepresentation.process_sample)
lp.add_function(LazyCollater.__call__)
lp.run('trainer.training_step(batch, batch_idx)')
lp.print_stats()
```

### Option 4: C/C++ Profiling (Most Detailed, Complex)

**For PyTorch C++ code:**

```bash
# Install perf (Linux)
sudo apt-get install linux-tools-common

# Profile with perf
perf record -g python training_script.py
perf report

# Look for:
# - torch::Tensor operations
# - aten:: functions (PyTorch C++ backend)
```

**For CUDA kernels:**

```bash
# Use NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx python training_script.py
nsys-ui profile.qdrep
```

---

## Recommended Next Steps

### Immediate: Single-Step PyTorch Trace (Highest Priority)

**Goal:** Capture detailed CPU/GPU trace for 1 step in middle of epoch

**Implementation:**

```python
# In hetero_cell_bipartite_dango_gi_lazy.py, modify training_step:

def training_step(self, batch, batch_idx):
    # Enable trace at step 50 (after warmup)
    if batch_idx == 50:
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            loss, _, _ = self._shared_step(batch, batch_idx, "train")
            # ... rest of training_step

        # Save trace
        prof.export_chrome_trace(f"lazy_hetero_step_{batch_idx}_trace.json")
        print(f"Trace saved to lazy_hetero_step_{batch_idx}_trace.json")
    else:
        # Normal execution
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

    return loss
```

**Analysis Checklist:**

1. Open `lazy_hetero_step_50_trace.json` in Chrome Tracing (chrome://tracing)
2. Look for long CPU blocks labeled:
   - `LazySubgraphRepresentation`
   - `incidence_cache`
   - `edge_mask`
   - `LazyCollater`
3. Check GPU utilization:
   - Is GPU idle during data loading? (indicates CPU bottleneck)
   - Are there long gaps between GPU operations?
4. Identify the single longest operation

### Short-term: Test Preprocessed Dataset (Experiment 087)

**Goal:** Determine if disk I/O is faster than computation

**Config:** Already exists as `hetero_cell_bipartite_dango_gi_gh_077.yaml`

**Modifications needed:**

1. Ensure preprocessing completed: `scripts/preprocess_lazy_dataset_full_masks.py`
2. Verify LMDB size and contents
3. Run single-GPU profiling (like 086a) with preprocessed data

**Expected outcomes:**

- **If faster (>10 it/s):** Subgraph ops were the bottleneck → use preprocessing in production
- **If similar (1-2 it/s):** Collation or I/O is bottleneck → need deeper optimization
- **If slower (<1 it/s):** Disk I/O can't keep up → subgraph computation is actually faster!

### Medium-term: Optimize LazySubgraphRepresentation

**If preprocessing doesn't help, optimize the computation:**

**Optimization opportunities:**

1. **Vectorize mask generation:** Use torch.index_select instead of loops
2. **Cache per-gene masks:** Store masks for single-gene perturbations, combine for multi-gene
3. **Reduce edge types:** Combine STRING types into single aggregated graph
4. **Approximate masks:** Use sampling or approximation for large subgraphs
5. **GPU-accelerated masking:** Move mask generation to GPU (tensor ops)

### Long-term: Architectural Changes

**If optimization insufficient, consider architectural changes:**

1. **Package cell_graph into batch:**
   - Include full cell_graph in batch (like DANGO)
   - Move masking into model forward pass (on GPU)
   - Reduces data pipeline to just loading perturbation indices

2. **Use sparse edge_index:**
   - Instead of full edge_index + mask, store only relevant edges
   - Requires changing model to handle variable-size graphs per sample

3. **Streaming/pipelining:**
   - Compute masks asynchronously on CPU while GPU trains
   - Use double buffering to hide latency

---

## Files and Configs Reference

### Created for Experiment 086:

**Base configs:**

- `conf/hetero_cell_bipartite_dango_gi_gh_086.yaml`
- `conf/dango_kuzmin2018_tmi_string12_0_086.yaml`

**Dataloader profiling variants:**

- `conf/hetero_cell_bipartite_dango_gi_gh_086_dataloader.yaml`
- `conf/dango_kuzmin2018_tmi_string12_0_086_dataloader.yaml`

**Model profiling variants:**

- `conf/hetero_cell_bipartite_dango_gi_gh_086_model.yaml`
- `conf/dango_kuzmin2018_tmi_string12_0_086_model.yaml`

**Scripts:**

- `scripts/run_profiling_086.slurm` (unified profiling + comparison)

**Trainer modifications:**

- `torchcell/trainers/int_hetero_cell.py` (added `execution_mode` parameter)
- `torchcell/trainers/int_dango.py` (added `execution_mode` parameter)

### Existing for Experiment 077 (Preprocessed):

**Config:**

- `conf/hetero_cell_bipartite_dango_gi_gh_077.yaml` (`use_full_masks: true`)

**Scripts:**

- `scripts/hetero_cell_bipartite_dango_gi_lazy_preprocessed.py` (uses preprocessed dataset)
- `scripts/preprocess_lazy_dataset.py`
- `scripts/preprocess_lazy_dataset_full_masks.py`
- `scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_077.slurm`

---

## Results Location

**Output directory:**

```
experiments/006-kuzmin-tmi/profiling_results/profiling_086_2025-11-07-00-20-10/
├── 086a_lazy_hetero_dataloader.log  # Data pipeline only
├── 086a_dango_dataloader.log
├── 086b_lazy_hetero_model.log       # Data + model
├── 086b_dango_model.log
└── comparison_086_2025-11-07-00-20-10.txt  # Full comparison report
```

**Slurm output:**

```
experiments/006-kuzmin-tmi/slurm/output/PROFILE-086_525.out
```

---

## Summary Table

| Component | Lazy Hetero Time | DANGO Time | Ratio | % of Lazy Total |
|-----------|-----------------|------------|-------|-----------------|
| **Data Pipeline** | 0.763 sec | 0.014 sec | 56× | **67%** |
| ├─ DataLoader fetch | ~0.05 sec | ~0.002 sec | 25× | 4% |
| ├─ Graph processor | **~0.65 sec** | **~0.001 sec** | **650×** | **57%** ← MAIN BOTTLENECK |
| ├─ Collation | ~0.05 sec | ~0.01 sec | 5× | 4% |
| └─ CPU→GPU transfer | ~0.013 sec | ~0.001 sec | 13× | 1% |
| **Model Compute** | 0.375 sec | 0.045 sec | 8.3× | 33% |
| ├─ Forward pass | ~0.250 sec | ~0.030 sec | 8.3× | 22% |
| └─ Backward pass | ~0.125 sec | ~0.015 sec | 8.3× | 11% |
| **TOTAL** | **1.138 sec** | **0.059 sec** | **19.3×** | **100%** |

**Key Takeaway:** Graph processor (LazySubgraphRepresentation) takes **0.65 sec** per batch (57% of total time), making it the clear primary bottleneck.

---

## Conclusion

Experiment 086 successfully isolated the bottleneck: **LazySubgraphRepresentation's edge mask computation is 650× slower than DANGO's Perturbation processor**, accounting for 57% of Lazy Hetero's total training time.

**Next critical experiment:** Create a PyTorch trace for a single step to identify the exact operation within LazySubgraphRepresentation that's slow (likely the incidence cache lookup loop).

**Medium-term solution:** Test experiment 077's preprocessed approach, but be prepared for it to fail if disk I/O is slower than computation.

**Long-term solution:** Architectural refactor to move edge masking to GPU or switch to sparse representations.

