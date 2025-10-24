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

## Current Phase: Optimize Gene Interactions

**Phase 2 (Revised)**: Data-driven optimization of `_process_gene_interactions`

**Target**: `torchcell/data/graph_processor.py:312-352`

**Current Performance**: 8.38ms/call (62% of total processing time)

**Goal**: Reduce to <4ms per call (2x speedup → 25-40% overall improvement)

### What `_process_gene_interactions` Does

```python
@time_method
def _process_gene_interactions(
    self, integrated_subgraph, cell_graph, gene_info
):
    # Process all gene-to-gene edge types (9 types total)
    for et in cell_graph.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            edge_index, _, edge_mask = pyg.utils.subgraph(
                subset=gene_info["keep_subset"],
                edge_index=cell_graph[et].edge_index,
                relabel_nodes=True,
                num_nodes=cell_graph["gene"].num_nodes,
                return_edge_mask=True,
            )
            integrated_subgraph[et].edge_index = edge_index
            integrated_subgraph[et].num_edges = edge_index.size(1)
            integrated_subgraph[et].pert_mask = ~edge_mask
```

### Bottleneck Analysis

- Loops over 9 edge types (physical, regulatory, tflink, 6x string12_0 channels)
- Each loop calls PyG's `subgraph()` which:
  1. Creates boolean mask for subset edges
  2. Filters edges with boolean indexing
  3. Relabels node indices (this is expensive!)
  4. Returns edge_index, edge_attr, edge_mask

### Optimization Opportunities

1. **Pre-compute gene mapping once** instead of 9 times
2. **Batch process edge types** instead of sequential loop
3. **Direct edge filtering** to avoid PyG's subgraph() overhead
4. **Reuse node mask** across all edge types

### Implementation Strategy

1. Pre-compute gene mapping tensor once (reuse across all 9 edge types)
2. Replace 9 sequential `subgraph()` calls with optimized batch processing
3. Test equivalence thoroughly
4. Benchmark with `TORCHCELL_DEBUG_TIMING=1` and `num_workers=0`
5. Only commit if measurable improvement achieved

---

## Future Phases

Based on timing data, if further optimization is needed:

- **Phase 3**: Optimize `_process_gene_info` and `_process_reaction_info` (14% each)
- **Phase 4**: Buffer reuse to reduce memory allocations
- **Phase 5**: Cache edge types (pre-filter gene-gene edges)

**Key Principle**: No optimization without timing data proving it's a bottleneck

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
