---
id: kbiwlc8cqgrwi2louqvmaso
title: Dango Vs Lazy Profile Comparison
desc: ''
updated: 1767845278036
created: 1767845278036
---

## DANGO vs Lazy Hetero Profile Comparison

**Date:** 2025-11-06

Comprehensive profiling analysis revealing DANGO is 875x faster than Lazy Hetero, with bottlenecks identified in the "Other" category (311s), optimizer (274s), and model forward pass (163s) under identical DDP conditions.

**Goal:** Identify why DANGO achieves ~10 it/s vs Lazy Hetero's 0.42 it/s

### Profiling Setup (Identical Conditions)

| Parameter | Value |
|-----------|-------|
| GPUs | 4 (DDP) |
| Batch Size per GPU | 8 |
| Dataset Size | 100 samples |
| Profiling Window | Steps 601-625 (after warmup) |
| Wait Steps | 600 |
| Active Steps | 25 |

---

### Performance Summary

| Model | Speed | Total Time (25 steps) | Speedup |
|-------|-------|----------------------|---------|
| **DANGO** | ~10 it/s | 1,304 ms | **875x faster** |
| **Lazy Hetero** | ~0.42 it/s | 1,141,139 ms | baseline |

**Key Finding**: DANGO is **875x faster** for the profiled 25 steps!

---

### Category Breakdown Comparison

| Category | DANGO (ms) | DANGO (%) | Lazy Hetero (ms) | Lazy Hetero (%) | Ratio (Lazy/DANGO) |
|----------|------------|-----------|------------------|-----------------|-------------------|
| **Model Forward** | 80 | 6.2% | 162,507 | 14.2% | **2,031x slower** |
| **Graph Processing** | 0.4 | 0.0% | 34,301 | 3.0% | **92,973x slower** |
| **Loss Computation** | 55 | 4.2% | 24,395 | 2.1% | 442x slower |
| **Backward Pass** | 14 | 1.0% | 23,427 | 2.1% | 1,673x slower |
| **DDP Communication** | 229 | 17.6% | 158,442 | 13.9% | 691x slower |
| **Optimizer** | 247 | 19.0% | 274,123 | 24.0% | 1,110x slower |
| **Tensor Ops** | 224 | 17.2% | 96,696 | 8.5% | 432x slower |
| **CUDA Kernels** | 84 | 6.5% | 55,794 | 4.9% | 663x slower |
| **Data Loading** | 15 | 1.2% | 4 | 0.0% | 0.3x (DANGO slower!) |
| **Other** | 354 | 27.2% | 311,451 | 27.3% | 880x slower |
| **TOTAL** | **1,304** | 100% | **1,141,139** | 100% | **875x slower** |

---

### KEY FINDINGS

#### 1. ⚠️ GRAPH PROCESSING: 92,973x SLOWER in Lazy Hetero

- **DANGO**: 0.4 ms (0.0%)
- **Lazy Hetero**: 34,301 ms (3.0%)
- **Impact**: This is THE major bottleneck!
- **Why**: Lazy hetero generates edge masks on-the-fly, DANGO has simpler graph structure

#### 2. Model Forward: 2,031x Slower in Lazy Hetero

- **DANGO**: 80 ms (6.2%)
- **Lazy Hetero**: 162,507 ms (14.2%)
- **Why**: Lazy hetero processes metabolism bipartite graph + more complex architecture

#### 3. Optimizer: 1,110x Slower in Lazy Hetero

- **DANGO**: 247 ms (19.0%)
- **Lazy Hetero**: 274,123 ms (24.0%)
- **Why**: Lazy hetero has more parameters? Or optimizer inefficiency?

#### 4. DDP Communication: Similar Relative Overhead

- **DANGO**: 17.6%
- **Lazy Hetero**: 13.9%
- **Note**: Both models have reasonable DDP overhead. Not a major differentiator.

#### 5. Data Loading: Both Optimized

- **DANGO**: 1.2% (15 ms)
- **Lazy Hetero**: 0.0% (4 ms)
- **Both models have negligible data loading time**

---

### Bottleneck Analysis

#### Lazy Hetero's Major Bottlenecks (in order of impact):

1. **"Other" Category (27.3%, 311,451 ms)**
   - Contains unclassified operations
   - Largest single category - needs investigation
   - Could include: memory operations, synchronization, overhead

2. **Optimizer (24.0%, 274,123 ms)**
   - Parameter updates taking significant time
   - Possible issues: large parameter count, inefficient optimizer step

3. **Model Forward (14.2%, 162,507 ms)**
   - Complex model architecture
   - Metabolism bipartite graph processing

4. **DDP Communication (13.9%, 158,442 ms)**
   - Gradient synchronization across 4 GPUs
   - Reasonable overhead for DDP

5. **Graph Processing (3.0%, 34,301 ms)**
   - Edge mask generation
   - Previous optimization target - already improved significantly

#### DANGO's Profile (for comparison):

1. **"Other" Category (27.2%, 354 ms)** - Similar percentage, much faster absolute
2. **Optimizer (19.0%, 247 ms)** - Reasonable
3. **DDP Communication (17.6%, 229 ms)** - Slightly higher % than lazy hetero
4. **Tensor Ops (17.2%, 224 ms)** - Higher % than lazy hetero (8.5%)
5. **Model Forward (6.2%, 80 ms)** - Much more efficient

---

### Why is Lazy Hetero 875x Slower?

#### The Scale Issue

The absolute time difference is so large (1.3s vs 19 minutes) that we need to consider:

**Possible Explanations:**

1. **Different Profiling Windows?**
   - Were both models profiled at steady-state?
   - Lazy hetero: steps 601-625
   - DANGO: steps 601-625
   - ✅ Both profiled after 600 warmup steps

2. **Batch Processing Difference?**
   - Lazy hetero: 100 samples / (8 batch × 4 GPU) = 3 steps/epoch
   - DANGO: 100 samples / (8 batch × 4 GPU) = 3 steps/epoch
   - ✅ Same batch processing

3. **Model Complexity Accumulation**
   - Every component is ~500-2000x slower
   - This compounds across the entire training loop
   - Graph processing (92,973x) is extreme outlier

4. **Memory Operations Hidden in "Other"?**
   - Both models have ~27% in "Other" category
   - But lazy hetero's "Other" is 311,451 ms vs 354 ms
   - Could be memory allocations, cache misses, synchronization

---

### Recommended Next Steps

#### Immediate Actions:

1. **Investigate "Other" Category** (311,451 ms, 27.3%)
   - What operations are being classified as "other"?
   - Memory allocations? Synchronization? Python overhead?

2. **Profile Optimizer in Detail** (274,123 ms, 24.0%)
   - Why is AdamW taking so long?
   - Parameter count comparison: DANGO vs Lazy Hetero
   - Check for unnecessary copies or synchronization

3. **Analyze Model Forward Pass** (162,507 ms, 14.2%)
   - Which layers/operations dominate?
   - Metabolism bipartite graph processing cost?
   - Can any layers be simplified or cached?

4. **Deep Dive on Graph Processing** (34,301 ms, 3.0%)
   - This was supposed to be optimized (vectorized GPU masks)
   - Still 92,973x slower than DANGO
   - Are masks being generated efficiently?

#### Long-term Optimizations:

1. **Reduce Model Complexity**
   - Can we simplify the metabolism bipartite graph?
   - Are all edge types necessary?

2. **Optimize Parameter Updates**
   - Gradient checkpointing?
   - Mixed precision training?
   - Different optimizer?

3. **Further Graph Processing Optimization**
   - Pre-compute more structures?
   - Better caching strategy?

---

### Conclusion

The lazy hetero model is **875x slower** than DANGO when profiled under identical DDP conditions. The slowdown is distributed across all categories, but the largest absolute contributors are:

1. **"Other" (311s)** - Needs investigation
2. **Optimizer (274s)** - Parameter update inefficiency
3. **Model Forward (163s)** - Complex architecture
4. **DDP Communication (158s)** - Reasonable for DDP

The **graph processing** (34s, 3.0%) is the biggest *relative* slowdown (92,973x) but not the largest *absolute* bottleneck.

**Priority**: Investigate the "Other" category and optimize the optimizer step.
