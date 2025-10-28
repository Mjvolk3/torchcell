# SubgraphRepresentation Performance Optimization Plan

**Project**: TorchCell HeteroCell Model Optimization
**Target**: 100x speedup in dataset creation (from 44.38ms/sample to <1ms/sample)
**Date**: October 2024
**Current Performance**: 44.38ms/sample (SubgraphRepresentation) vs 0.42ms/sample (Perturbation baseline)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Directory Structure](#directory-structure)
3. [Key Files Reference](#key-files-reference)
4. [Completed Phases](#completed-phases)
5. [Current Phase: Optimize Gene Interactions](#current-phase-optimize-gene-interactions)
6. [Future Phases](#future-phases)
7. [Rollback Procedure](#rollback-procedure)
8. [Fresh Session Startup](#fresh-session-startup)

---

## Executive Summary

This document outlines a data-driven plan to optimize the SubgraphRepresentation class in TorchCell's graph processor module. The optimization targets dataset creation time, which is currently 100x slower than the simpler Perturbation processor.

**Key Strategy**: Use timing instrumentation to identify actual bottlenecks, then optimize based on measured data rather than theoretical complexity.

**Validation Method**: Equivalence testing ensures optimized code produces identical outputs to baseline.

**Current Status**: Timing instrumentation revealed `_process_gene_interactions` is the #1 bottleneck (62% of time). This is our next optimization target.

**Progress Report**: See `subgraph-optimization-progress-report.md` for detailed execution history.

---

## Directory Structure

```
# Core Implementation
torchcell/data/graph_processor.py                    # TARGET FILE - SubgraphRepresentation class
torchcell/profiling/timing.py                        # Timing instrumentation utility

# Testing and Benchmarking
tests/torchcell/data/test_graph_processor_equivalence.py
experiments/006-kuzmin-tmi/scripts/benchmark_graph_processors.py
experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm

# Test Data and References
/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/
├── reference_baseline.pkl                           # Baseline reference data
└── profiling_results/                               # Benchmark outputs

# SLURM Outputs
experiments/006-kuzmin-tmi/slurm/output/*.out       # Job outputs
```

---

## Key Files Reference

**Primary Target**: `torchcell/data/graph_processor.py`
- Class: `SubgraphRepresentation`
- Current Bottleneck: `_process_gene_interactions()` (8.38ms/call, 62% of time)

**Benchmarking**:
- Script: `experiments/006-kuzmin-tmi/scripts/benchmark_graph_processors.py`
- SLURM: `experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm`
- Run with: `TORCHCELL_DEBUG_TIMING=1` for timing data

**Testing**:
- Equivalence tests: `tests/torchcell/data/test_graph_processor_equivalence.py`
- Command: `pytest tests/torchcell/data/test_graph_processor_equivalence.py -xvs`

**Timing Instrumentation**:
- Utility: `torchcell/profiling/timing.py`
- Decorator: `@time_method`
- Enable: Set `TORCHCELL_DEBUG_TIMING=1` environment variable

---

## Completed Phases

### Phase 0: Setup and Baseline ✅

**Deliverables**:
- Created equivalence test suite
- Generated baseline reference data
- Established testing infrastructure

**Result**: All tests passing, baseline established

---

### Phase 1: Bipartite Subgraph Optimization ✅

**Target**: `_process_metabolism_bipartite()` method

**Optimization**: Replaced general `bipartite_subgraph()` with optimized filtering for all-metabolite case (100% of usage). Uses direct boolean masking instead of expensive gather/scatter operations.

**Result**:
- Optimization successful (0.70ms/call, only 5% of total time)
- No training speedup observed (profiler uses cached data)
- Dataset creation speedup confirmed in Phase 1.5 benchmark

---

### Phase 1.5: Benchmark Verification ✅

**Objective**: Definitively verify whether graph processing is the bottleneck

**Approach**: Direct benchmark comparing SubgraphRepresentation vs Perturbation processors

**Results**:
- SubgraphRepresentation: 44.38ms/sample
- Perturbation: 0.42ms/sample
- **104.52x SLOWER**

**Decision**: ✅ Graph processing IS the bottleneck - continue with optimizations

---

### Phase 2: Boolean Mask Indexing ❌ FAILED

**Attempted**: Replace `torch.isin()` with boolean mask indexing

**Expected**: 10-15% speedup (O(n) vs O(n×m) complexity)

**Result**: 3.15% SLOWDOWN (45.78ms vs 44.38ms baseline)

**Why It Failed**:
- Allocation overhead (~13k boolean values per call) exceeded algorithmic savings
- `torch.isin()` is highly optimized in PyTorch's C++/CUDA backend
- Big-O notation ignores constant factors - library implementations are heavily optimized

**Lesson**: Don't optimize blindly based on algorithmic complexity - always measure!

**Action**: Reverted changes, implemented timing instrumentation

---

### Phase 2.1: Timing Instrumentation ✅

**Objective**: Identify actual bottlenecks using measurement instead of theory

**Implementation**:
- Created `torchcell/profiling/timing.py` with `@time_method` decorator
- Added decorators to all SubgraphRepresentation methods
- Environment variable control: `TORCHCELL_DEBUG_TIMING=1`

**Timing Results** (1,152 calls):
```
Method                                       Calls    Total (ms)    Mean (ms)
-------------------------------------------------------------------------------
_process_gene_interactions                    1152      9649.85       8.3766  (62%)
_process_gene_info                            1152      2248.75       1.9520  (14%)
_process_reaction_info                        1152      2237.11       1.9419  (14%)
_process_metabolism_bipartite                 1152       804.50       0.6983  (5%)
_initialize_masks                             1152       233.53       0.2027  (1.5%)
```

**Key Finding**: `_process_gene_interactions` is the #1 bottleneck at 8.38ms/call (62% of time)

**Multiprocessing Limitation**: Timing only works with `num_workers=0` due to separate process spaces

---

### Phase 2.2: Incidence-Based Optimization (Path A) ❌ FAILED

**Attempted**: Use incidence cache to accelerate edge mask computation

**Strategy**: Pre-compute node-to-edge incidence mappings to find edges touching perturbed genes in O(k×d) instead of O(E)

**Implementation Attempts**:
1. Python set operations: 9.59ms/call (14% slower)
2. Pure tensor operations: 9.10ms/call (10% slower)

**Why It Failed**:
- Incidence cache successfully computed masks faster
- BUT: Still doing expensive operations afterward:
  - Tensor allocation: `kept_edges = edge_index[:, edge_mask]` (O(E') copy)
  - Node relabeling: `gene_map[kept_edges[0]]` (O(E') operations)
  - New tensor creation: `torch.stack([...])` per edge type
- Competing against PyTorch Geometric's C++/CUDA optimized `subgraph()` using Python-level operations
- **Core issue**: Wrong optimization layer - computing mask faster doesn't help if we still allocate new tensors

**Benchmark Results** (bench-processors_347):
- SubgraphRepresentation: 54.80ms/sample (_process_gene_interactions: 8.28ms)
- IncidenceSubgraphRepresentation: 55.62ms/sample (_process_gene_interactions: 9.10ms)
- **Still 0.82ms SLOWER**

**Key Insight**: The theory is sound, but the implementation is at the wrong architectural layer. We need to stop building new graphs entirely (Path B), not just optimize how we build them (Path A).

---

## Phase 3: LazySubgraphRepresentation (Gene Graphs) ✅ COMPLETE

**Status**: ✅ **11.8x speedup achieved on gene graph processing!**

**Implementation**: Created `LazySubgraphRepresentation` class

**Results**:
- Gene graph processing: 8.26ms → 0.69ms (**11.8x faster**)
- Total graph processing: 13.28ms → 5.64ms (2.35x faster)
- Overall per-sample: 54.98ms → 45.99ms (1.20x faster)
- Memory savings: ~2.7 MB per sample (93.7% reduction for edge tensors)

**What Was Optimized**:
- ✅ All gene-gene edge types (physical, regulatory, tflink, string12_0)
- ✅ Zero-copy edge references
- ✅ O(k×d) mask computation using incidence cache

**What's Still NOT Optimized**:
- ❌ Metabolism bipartite edges (reaction-metabolite)
- ❌ Gene-reaction (GPR) edges
- ❌ Stoichiometry attributes

See `subgraph-optimization-progress-report.md` for detailed benchmark results.

---

## Current Phase: Lazy All Edge Types (Phase 4)

**Phase 4**: Extend lazy approach to ALL edge types

**Strategy**: Apply zero-copy + mask architecture to metabolism bipartite and GPR edges

**Target**: Complete lazy transformation in `graph_processor.py`

### What's Left to Optimize

**Current State** (from `load_lazy_batch_006.py`):

```python
HeteroData(
  # ✅ OPTIMIZED - Zero-copy with masks
  (gene, physical, gene)={ edge_index=[2,144211], mask=[144211] }
  (gene, regulatory, gene)={ edge_index=[2,44310], mask=[44310] }

  # ❌ NOT OPTIMIZED - Still allocating/filtering
  (gene, gpr, reaction)={ hyperedge_index=[2,5450], pert_mask=[5450] }
  (reaction, rmr, metabolite)={
    hyperedge_index=[2,26325],
    stoichiometry=[26325]
  }
)
```

**Phase 4 Tasks**:

1. **Lazy GPR edges** (gene → reaction):
   - Return full hyperedge_index (reference)
   - Compute mask based on perturbed genes
   - ~0.5ms savings

2. **Lazy metabolism bipartite** (reaction → metabolite):
   - Return full hyperedge_index and stoichiometry (references)
   - Compute mask based on invalid reactions
   - ~0.6ms savings

**Expected Total Speedup**: Additional 1-2ms reduction in graph processing

### Phase 3 Success - What Was Achieved

**Current approach** (both SubgraphRepresentation and IncidenceSubgraphRepresentation):
```python
# Per batch, per edge type:
kept_edges = edge_index[:, edge_mask]           # O(E') copy ~2.4M edges
new_edge_index = torch.stack([                  # O(E') allocation
    gene_map[kept_edges[0]],                    # O(E') relabeling
    gene_map[kept_edges[1]]
])
integrated_subgraph[et].edge_index = new_edge_index  # Store new tensor
```

**Cost per batch:**
- Allocate new tensors: 9 edge types × 2.4M edges × 8 bytes = ~170MB
- Copy operations: 9 × O(E)
- Node relabeling: 9 × O(E)
- Competing against PyG's C++/CUDA with Python operations

**Perturbation processor** (97× faster):
```python
# One-time reference, never copied:
processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes
processed_graph["gene"].pert_mask = pert_mask      # Just mask (6K bools)
processed_graph["gene"].mask = ~pert_mask         # Just mask
# No edge_index, no tensor allocation, no copying
```

---

### Path B Strategy: Mask-Based Architecture

**Key Difference from Perturbation:**

| Aspect | Perturbation (Dango) | HeteroCell (Our Need) |
|--------|---------------------|----------------------|
| Message passing | On FULL unperturbed graph | On PERTURBED graph |
| Mask usage | Node masks for downstream only | Node + edge masks for MP |
| Edge handling | Model sees all edges | Must simulate edge removal |

**Proposed Architecture:**

```python
# Processor returns full graph + masks (no tensor allocation)
processed_graph[et].edge_index = cell_graph[et].edge_index        # Reference only
processed_graph[et].edge_alive_mask = compute_edge_mask(...)      # Boolean mask
processed_graph["gene"].x = cell_graph["gene"].x                  # Reference only
processed_graph["gene"].node_alive_mask = node_mask               # Boolean mask
processed_graph["gene"].pert_mask = ~node_mask                    # Boolean mask
```

**Edge mask computation using incidence cache:**
```python
# One-time cache build (O(E)):
incidence[edge_type][gene_idx] = tensor([edge_ids where gene appears])

# Per batch (O(k×d) where k=perturbed genes, d=degree):
edge_alive_mask = torch.ones(num_edges, dtype=bool)
for gene_idx in perturbed_genes:
    edge_alive_mask[incidence[edge_type][gene_idx]] = False
```

**Cost per batch:**
- Compute edge masks: O(k×d) ≈ 10 genes × 50 edges = 500 ops (vs 2.4M)
- Compute node masks: O(k) ≈ 10 ops
- Tensor allocation: 9 boolean masks × 2.4M bits ≈ 2.7MB (vs 170MB)
- **No edge_index copying, no node relabeling**

---

### Implementation Plan

**Phase 3.1: Modify Graph Processor**

Create new `MaskedGraphProcessor`:
1. Build incidence cache on initialization (one-time O(E))
2. In `process()`:
   - Return references to full `cell_graph` tensors (no copying)
   - Compute `node_alive_mask` (O(k))
   - Compute `edge_alive_mask` per edge type using incidence (O(k×d))
   - Attach masks to returned graph
3. Validate with equivalence test modified to compare after mask application

**Phase 3.2: Modify HeteroCell Model**

Update model forward pass to handle masks:

**Option A (Quick)**: Pre-filter edges on GPU before message passing
```python
for edge_type in batch.edge_types:
    edge_mask = batch[edge_type].edge_alive_mask
    edge_index_active = batch[edge_type].edge_index[:, edge_mask]
    # Use edge_index_active in message passing
```
- Cost: One-time O(E') filter per forward pass on GPU
- Still faster than CPU-side filtering + host→device transfer

**Option B (Optimal)**: Mask-aware message passing
```python
# Custom MessagePassing that applies edge_alive_mask internally
# Reuse same edge_index every batch, multiply messages by mask
```
- Cost: Zero edge filtering, mask applied during aggregation
- Requires custom layer implementation

**Phase 3.3: Validation Strategy**

1. Create test comparing:
   - Old: SubgraphRepresentation output
   - New: MaskedGraphProcessor + mask application
   - Verify identical after mask filtering
2. Benchmark both options A and B
3. Measure end-to-end training speedup

---

### Expected Performance

**Theoretical speedup:**
- Current: O(N+E) per batch × 9 edge types ≈ 22M ops
- Path B: O(k×d) per batch × 9 edge types ≈ 4.5K ops
- **~4800× reduction in preprocessing**

**Realistic expectations:**
- Current bottleneck: 8.28ms for `_process_gene_interactions`
- Target: <0.5ms (10-20× speedup)
- Overall: 54.80ms → 5-10ms per sample (5-10× total)

**Why not 100×?**
- Other operations still needed (phenotype, GPU transfer, etc.)
- Model changes (Option A) add some overhead back
- Conservative: Achieve 10-20× speedup total

---

## Rollback Procedure

If any optimization fails:

1. **STOP** - Do not proceed to next optimization
2. Check test output for specific failure
3. If quick fix not possible, revert:
   ```bash
   git checkout HEAD~1 torchcell/data/graph_processor.py
   ```
4. Re-run equivalence test to confirm rollback
5. Document issue before re-attempting

---

## Fresh Session Startup

When starting work on this optimization:

1. **Read this document** and the progress report
2. **Check current status**: Review git log to see what's been completed
3. **Verify environment**:
   ```bash
   cd /home/michaelvolk/Documents/projects/torchcell
   conda activate torchcell
   ```
4. **Test baseline**:
   ```bash
   pytest tests/torchcell/data/test_graph_processor_equivalence.py -xvs
   ```
5. **Review timing data**: Check latest benchmark output with timing enabled

---

## Important Notes

1. **Never skip equivalence testing** - This ensures correctness
2. **Always measure before and after** - No optimization without proof it helps
3. **Data-driven approach** - Let timing data guide optimization decisions
4. **Atomic commits** - Each optimization is independently reversible
5. **Document failures** - Failed optimizations teach us what NOT to do

---

**Document Version**: 2.0
**Last Updated**: October 2024
**Status**: Phase 2.1 complete, Phase 2 (revised) next
