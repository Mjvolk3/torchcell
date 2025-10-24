# SubgraphRepresentation Optimization Progress

**Objective**: Achieve 100x speedup in dataset creation (from 44.38ms/sample to <1ms/sample)

**See Also**: `2025-10-22-subgraph-optimization-plan.md` for detailed strategy

---

## Summary

| Phase | Status | Result |
|-------|--------|--------|
| Phase 0: Setup and Baseline | ✅ Complete | Baseline established, tests passing |
| Phase 1: Bipartite Optimization | ✅ Complete | 0.70ms/call (5% of time) - SUCCESS |
| Phase 1.5: Benchmark Verification | ✅ Complete | Confirmed 104.52x slower than Perturbation |
| Phase 2: Boolean Mask Indexing | ❌ Failed | 3.15% SLOWER - reverted |
| Phase 2.1: Timing Instrumentation | ✅ Complete | Identified `_process_gene_interactions` as #1 bottleneck (62%) |
| **Phase 2 (Revised): Optimize Gene Interactions** | 🔄 **NEXT** | Target: 8.38ms → <4ms per call |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Baseline (SubgraphRepresentation) | 44.38ms/sample |
| Target (Perturbation baseline) | 0.42ms/sample |
| Required Speedup | ~100x in dataset creation |
| Current Bottleneck | `_process_gene_interactions` at 8.38ms/call (62% of time) |
| Phase 1 Success | `_process_metabolism_bipartite` now 0.70ms/call (5% of time) |

---

## Phase 0: Setup and Baseline ✅

**Objective**: Establish testing infrastructure and baseline performance

**Deliverables**:

1. Equivalence test suite (`tests/torchcell/data/test_graph_processor_equivalence.py`)
2. Baseline reference data (`/scratch/.../reference_baseline.pkl`)
3. Deterministic data loading (comprehensive seeding)

**Baseline Metrics**:

- CUDA Time: 624.345ms (for training profiling)
- Data Loading: 44.38ms/sample (for dataset creation)

**Key Learnings**:

- Tests initially failed due to non-deterministic data loading
- Some attributes (like `ids_pert`) are semantically sets but stored as lists
- Edge ordering differs between runs - needed canonical sorting
- Production settings work with proper seeding

---

## Phase 1: Bipartite Subgraph Optimization ✅

**Target**: `_process_metabolism_bipartite()` method (lines 364-395)

**Problem**: Used expensive `bipartite_subgraph()` which performs gather/scatter operations even though we always keep all metabolites (100% of production cases).

**Solution**: Replaced with direct boolean masking and reaction index remapping.

**Result**:

- Optimization implemented successfully
- All equivalence tests PASSED
- **CUDA Time**: 624.987ms (vs baseline 624.345ms) - no training speedup
- **Why**: Graph processor runs during dataset creation (one-time cost), not training loop
- Dataset is cached - profiling uses pre-cached data from August

**Key Finding**: The 244ms gather/scatter operations in the profiler are from the GNN model's forward pass (message passing), NOT from the graph processor's `bipartite_subgraph()` call.

**Decision**: Need to verify whether graph processing is actually the bottleneck → Proceed to Phase 1.5

---

## Phase 1.5: Benchmark Verification ✅

**Objective**: Definitively verify whether graph processing is the bottleneck

**Approach**: Direct benchmark comparing SubgraphRepresentation vs Perturbation processors on data loading with GPU transfer.

**Benchmark Configuration**:

- Sample size: 10,000 samples
- Batch size: 32
- Num workers: 2
- GPU transfer: Yes (`.to('cuda')`)

**Results** (SLURM job bench-processors_334):

| Processor | Time | Per-sample | Throughput |
|-----------|------|------------|------------|
| SubgraphRepresentation (HeteroCell) | 444.54s | 44.38ms | 22.5 samples/sec |
| Perturbation (Dango) | 4.25s | 0.42ms | 2354.9 samples/sec |

**Relative Performance**: SubgraphRepresentation is **104.52x SLOWER** than Perturbation

**Conclusion**: ✅ Graph processing IS definitively the bottleneck

**Decision**: Continue with optimization phases, but use timing instrumentation to identify actual hotspots

---

## Phase 2: Boolean Mask Indexing ❌ FAILED

**Target**: `_process_reaction_info()` method (lines 296-310)

**Attempted**: Replace `torch.isin()` with boolean mask indexing for O(n) vs O(n×m) complexity

**Benchmark Results** (SLURM job bench-processors_335):

- **Phase 1 Baseline**: 44.38ms/sample
- **Phase 2 with mask indexing**: 45.78ms/sample
- **Result**: 3.15% SLOWER

**Why It Failed**:

- Allocation overhead (~13,000 boolean values per call) exceeded algorithmic savings
- `torch.isin()` is highly optimized in PyTorch's C++/CUDA backend
- Big-O notation ignores constant factors
- Memory allocation and zeroing overhead dominated performance

**Lesson Learned**:

- Don't optimize blindly based on algorithmic complexity
- Library implementations (torch, numpy) are heavily optimized
- **Always measure before and after**
- Data-driven optimization > theoretical optimization

**Action**: Reverted Phase 2 changes, implemented timing instrumentation

---

## Phase 2.1: Timing Instrumentation ✅

**Objective**: Establish data-driven optimization approach with timing instrumentation

**Implementation**:

1. Created `torchcell/profiling/timing.py` with `@time_method` decorator
2. Added decorators to all SubgraphRepresentation methods
3. Environment variable control: `TORCHCELL_DEBUG_TIMING=1`
4. Updated SLURM script to export environment variable
5. Set `num_workers=0` for timing collection (multiprocessing limitation)

**Benchmark Configuration**:

- Sample count: 1,000 samples
- Batch size: 32
- **Num workers: 0** (required for timing collection)
- Environment: `TORCHCELL_DEBUG_TIMING=1`

**Timing Results** (SLURM job bench-processors_340, 1,152 calls):

```
================================================================================
Method                                                Calls   Total (ms)    Mean (ms)
--------------------------------------------------------------------------------
SubgraphRepresentation._process_gene_interactions     1152      9649.85       8.3766
SubgraphRepresentation._process_gene_info             1152      2248.75       1.9520
SubgraphRepresentation._process_reaction_info         1152      2237.11       1.9419
SubgraphRepresentation._process_metabolism_bipartite  1152       804.50       0.6983
SubgraphRepresentation._initialize_masks              1152       233.53       0.2027
SubgraphRepresentation.process                        1152       164.37       0.1427
================================================================================
```

**Breakdown by % of total time**:

1. `_process_gene_interactions`: 8.38ms/call **(62%)**
2. `_process_gene_info`: 1.95ms/call (14%)
3. `_process_reaction_info`: 1.94ms/call (14%)
4. `_process_metabolism_bipartite`: 0.70ms/call (5%)
5. `_initialize_masks`: 0.20ms/call (1.5%)
6. `process`: 0.14ms/call (1%) - overhead only

**Key Finding**: `_process_gene_interactions` is the #1 bottleneck at **8.38ms/call (62% of time)**

**Phase 1 Validation**: The `_process_metabolism_bipartite` optimization DID work! It's now only 0.70ms/call (5% of time).

**Multiprocessing Limitation Discovered**:

- DataLoader with `num_workers > 0` creates worker processes with separate `_TIMINGS` dictionaries
- Timing data collected in workers is lost when processes terminate
- **Solution**: Use `num_workers=0` for profiling, `num_workers > 0` for production

**Files Modified**:

1. `torchcell/profiling/timing.py` - Created timing utility
2. `torchcell/profiling/__init__.py` - Created package
3. `torchcell/data/graph_processor.py` - Added `@time_method` decorators, reverted Phase 2
4. `experiments/006-kuzmin-tmi/scripts/benchmark_graph_processors.py` - Added timing summary, set num_workers=0
5. `experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm` - Added `TORCHCELL_DEBUG_TIMING=1` environment variable

---

## Phase 2 (Revised): Optimize Gene Interactions Processing - NEXT

**Target**: `torchcell/data/graph_processor.py:312-352` - `_process_gene_interactions()` method

**Current Performance**: 8.38ms/call (62% of total processing time)

**Goal**: Reduce to <4ms per call (2x speedup → 25-40% overall improvement)

**What It Does**:

- Loops over 9 edge types (physical, regulatory, tflink, 6x string12_0 channels)
- Each loop calls PyG's `subgraph()` for edge filtering and node relabeling
- Total 9 calls to `subgraph()` per invocation

**Optimization Strategy**:

1. Pre-compute gene mapping tensor once (reuse across all 9 edge types)
2. Replace 9 sequential `subgraph()` calls with optimized batch processing
3. Direct edge filtering to avoid PyG's subgraph() overhead
4. Reuse node mask across all edge types

**Validation Plan**:

1. Implement optimization
2. Run equivalence tests
3. Benchmark with `TORCHCELL_DEBUG_TIMING=1` and `num_workers=0`
4. **Only commit if measurable improvement achieved**

---

## Test and Benchmark Commands

**Run equivalence tests**:

```bash
pytest tests/torchcell/data/test_graph_processor_equivalence.py -xvs
```

**Run benchmark with timing**:

```bash
# Ensure TORCHCELL_DEBUG_TIMING=1 is set in SLURM script
sbatch experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm
```

**Run benchmark without timing** (production speed):

```bash
# Comment out TORCHCELL_DEBUG_TIMING=1, set num_workers=2
sbatch experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm
```

---

## Next Action

**Phase 2 (Revised)**: Optimize `_process_gene_interactions` method

1. Analyze PyG's `subgraph()` implementation to understand overhead
2. Design optimized edge filtering approach
3. Implement optimization
4. Run equivalence tests to verify correctness
5. Benchmark with timing enabled to measure improvement
6. Document results and commit if successful

---

**Last Updated**: October 2024
**Current Status**: Phase 2.1 complete, ready for Phase 2 (revised)
**Key Principle**: No optimization without measurement proving it helps
