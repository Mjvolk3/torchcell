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
| Phase 2.2: Incidence Optimization (Path A) | ❌ Failed | 9.10ms vs 8.28ms baseline (10% SLOWER) |
| **Phase 3: LazySubgraphRepresentation (Gene Graphs)** | ✅ **Complete** | **11.8x speedup on gene graphs! (8.38ms → 0.71ms)** |
| **Phase 4: Lazy All Edge Types** | 🔄 **NEXT** | Extend to metabolism bipartite and GPR edges |

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

## Phase 2.2: Incidence-Based Optimization (Path A) ❌ FAILED

**Target**: `torchcell/data/graph_processor.py` - `_process_gene_interactions()` method

**Current Performance**: 8.38ms/call (62% of total processing time)

**Goal**: Reduce to <4ms per call using incidence cache

**Strategy**: Pre-compute node-to-edge incidence mappings to find edges touching perturbed genes in O(k×d) instead of O(E)

**Implementation Attempts**:

1. **Python set operations** (bench-processors_346):
   - IncidenceSubgraphRepresentation: 9.59ms/call
   - SubgraphRepresentation (baseline): 8.40ms/call
   - **Result: 14% SLOWER**

2. **Pure tensor operations** (bench-processors_347):
   - IncidenceSubgraphRepresentation: 9.10ms/call
   - SubgraphRepresentation (baseline): 8.28ms/call
   - **Result: 10% SLOWER**

**Why It Failed**:

The incidence cache successfully computed edge masks faster, BUT we were still doing expensive operations afterward:

```python
# Still expensive even with faster mask computation:
kept_edges = edge_index[:, edge_mask]        # O(E') tensor copy ~2.4M edges
new_edge_index = torch.stack([               # O(E') allocation
    gene_map[kept_edges[0]],                 # O(E') node relabeling
    gene_map[kept_edges[1]]
])
integrated_subgraph[et].edge_index = new_edge_index  # New tensor storage
```

**Root Cause Analysis**:

- Competing against PyTorch Geometric's C++/CUDA optimized `subgraph()` using Python-level operations
- Per batch cost: 9 edge types × 2.4M edges × 8 bytes ≈ 170MB tensor allocation
- **Core issue**: Wrong optimization layer - computing mask faster doesn't help if we still allocate new tensors every batch

**Benchmark Comparison**:

| Processor | Per-sample | _process_gene_interactions | Result |
|-----------|------------|---------------------------|---------|
| SubgraphRepresentation | 54.80ms | 8.28ms | Baseline |
| IncidenceSubgraphRepresentation | 55.62ms | 9.10ms | **0.82ms SLOWER** |
| Perturbation (Dango) | 0.53ms | N/A | 104× faster |

**Key Insight**:

Perturbation is 104× faster because it doesn't build filtered graphs at all - it returns full graph + masks. The theory behind incidence optimization is sound, but the implementation must be at the architectural level (Path B), not just optimizing the mask computation within the current architecture (Path A).

**Files Modified**:

1. `torchcell/data/graph_processor.py` - Added `IncidenceSubgraphRepresentation` class
2. `experiments/006-kuzmin-tmi/scripts/benchmark_graph_processors.py` - Added 3-way comparison

**Lesson Learned**:

Algorithmic optimization at the wrong architectural layer yields no benefit. To achieve real speedup, we need to change the architecture to stop building filtered graphs entirely (Path B).

---

## Phase 3: LazySubgraphRepresentation (Gene Graphs) ✅

**Target**: Gene-gene edge types (9 edge types: physical, regulatory, tflink, string12_0 channels)

**Strategy**: Zero-copy architecture with boolean masks instead of filtered tensors

**Implementation**: Created `LazySubgraphRepresentation` class in `torchcell/data/graph_processor.py`

**Key Features**:

```python
# Returns FULL edge_index (reference to cell_graph)
integrated_subgraph[et].edge_index = cell_graph[et].edge_index  # Zero-copy
integrated_subgraph[et].num_edges = num_edges                    # FULL count
integrated_subgraph[et].mask = edge_mask                         # Boolean mask only (True=keep)

# Uses incidence cache for O(k×d) mask computation
edge_mask = torch.ones(num_edges, dtype=torch.bool)
for node_idx in perturbed_nodes:
    edge_mask[incidence_cache[et][node_idx]] = False
```

**Benchmark Results** (SLURM job bench-lazy-vs-subgraph_355):

| Metric | SubgraphRep | LazySubgraphRep | Speedup |
|--------|-------------|-----------------|---------|
| **_process_gene_interactions** | **8.26ms** | **0.69ms** | **11.8x ⭐** |
| _process_gene_info | 1.95ms | 1.96ms | 1.0x |
| _process_reaction_info | 1.93ms | 1.82ms | 1.06x |
| _process_metabolism_bipartite | 0.68ms | 0.66ms | 1.03x |
| **Total graph processing** | **13.28ms** | **5.64ms** | **2.35x** |
| **Overall per-sample** | **54.98ms** | **45.99ms** | **1.20x** |

**Key Achievement**: Gene graph processing went from **8.26ms → 0.69ms** (91.6% reduction)

**Memory Savings**:
- Per sample: ~2.7 MB (93.7% reduction for edge tensors)
- Edge tensors: Zero-copy references (0 bytes allocated)
- Edge masks: ~0.2 MB boolean masks (vs ~2.9 MB int64 tensors)

**Why Only 1.20x Overall?**

The remaining 40ms per sample is spent on:
- DataLoader overhead
- GPU transfer (`.to('cuda')`)
- Batch collation
- Node embedding lookups
- Dataset infrastructure

**What's Still NOT Optimized**:

Looking at the current data structure:

```python
HeteroData(
  (gene, physical, gene)={ mask=[144211] }           # ✅ Lazy (zero-copy)
  (gene, regulatory, gene)={ mask=[44310] }          # ✅ Lazy (zero-copy)
  (gene, gpr, reaction)={ hyperedge_index=[2,5450] } # ❌ Still filtered
  (reaction, rmr, metabolite)={                      # ❌ Still filtered
    hyperedge_index=[2,26325],
    stoichiometry=[26325]
  }
)
```

**Files Created**:

1. `torchcell/data/graph_processor.py` - Added `LazySubgraphRepresentation` class (lines 1211-1828)
2. `torchcell/scratch/load_lazy_batch_006.py` - Inspection script
3. `torchcell/scratch/compare_lazy_vs_subgraph.py` - Equivalence verification
4. `experiments/006-kuzmin-tmi/scripts/benchmark_lazy_vs_subgraph.py` - Benchmark script
5. `torchcell/profiling/timing.py` - Added `print_comparison_table()` function

**Equivalence Verification**:

- ✅ Node data identical to SubgraphRepresentation
- ✅ Edge indices recoverable: `filtered = full_edge_index[:, mask]` then relabel
- ✅ Both edge types (physical, regulatory) match after relabeling

**Next Step**: Phase 4 - Extend lazy approach to ALL edge types (metabolism bipartite, GPR)

---

## Phase 4: Lazy All Edge Types - NEXT

**Target**: Extend lazy approach to metabolism bipartite and GPR edges

**Current Status**: Gene-gene edges optimized (✅), but metabolism still uses tensor allocation:

```python
# Current implementation (EXPENSIVE):
final_edge_index = torch.stack([
    reaction_map[reaction_indices[edge_mask]],  # Relabeling + filtering
    metabolite_indices[edge_mask]
])
final_edge_attr = stoichiometry[edge_mask]       # Copy edge attributes

# Proposed lazy alternative (CHEAP):
integrated_subgraph[et].hyperedge_index = hyperedge_index  # Reference (zero-copy)
integrated_subgraph[et].stoichiometry = stoichiometry      # Reference (zero-copy)
integrated_subgraph[et].mask = edge_mask                   # Boolean mask only
```

**Expected Savings**:
- Current: 0.68ms per sample (5% of graph processing)
- Target: ~0.1ms per sample
- Memory: Additional ~1-2 MB saved per sample

**Strategy**: Apply same zero-copy approach to all remaining edge types:

1. **Gene-Reaction (GPR) edges**: Already partially optimized, may need full lazy treatment
2. **Reaction-Metabolite edges**: Needs zero-copy + mask implementation
3. **Stoichiometry attributes**: Return references instead of filtered copies

**Model Architecture Changes Required**:

After completing lazy graph processing, the HeteroCell model will need mask-aware message passing. Proposed general pattern:

```python
def masked_mp_layer(
    x_batch,            # [B, N, F_in]
    edge_index_full,    # [2, E] (src, dst) long - SHARED across batches
    edge_alive_mask,    # [B, E] bool - per-experiment masks
    W,                  # [F_in, F_out] learnable
):
    # 1. Gather source features for all edges (batched)
    x_src = x_batch[:, edge_index_full[0], :]  # [B, E, F_in]

    # 2. Transform messages
    msg = x_src @ W  # [B, E, F_out]

    # 3. Apply per-experiment edge masks
    msg = msg * edge_alive_mask.unsqueeze(-1).to(msg.dtype)  # Zero out deleted edges

    # 4. Scatter to destinations (using torch_scatter)
    # ... (see full implementation in discussion)
```

This pattern works for ALL edge types (gene-gene, gene-reaction, reaction-metabolite) without special cases.

**Key Difference from Perturbation**:

| Aspect | Perturbation (Dango) | HeteroCell (Our Need) |
|--------|---------------------|----------------------|
| Message passing | On FULL unperturbed graph | On PERTURBED graph |
| Mask usage | Node masks for downstream only | **Node + edge masks for MP** |
| Edge handling | Model sees all edges | **Must simulate edge removal** |

**Why We Need Edge Masks**:

- Perturbation: Does MP on full graph, uses node masks only for pooling/prediction
- HeteroCell: Needs MP on perturbed graph → requires edge masks to filter during MP
- Incidence cache principle still applies for efficient edge mask computation

**Proposed Architecture**:

```python
# Processor returns full graph + masks (no tensor allocation)
processed_graph[et].edge_index = cell_graph[et].edge_index        # Reference only
processed_graph[et].edge_alive_mask = compute_edge_mask(...)      # Boolean mask
processed_graph["gene"].x = cell_graph["gene"].x                  # Reference only
processed_graph["gene"].node_alive_mask = node_mask               # Boolean mask
processed_graph["gene"].pert_mask = ~node_mask                    # Boolean mask
```

**Expected Speedup**:

- Current: O(N+E) per batch × 9 edge types ≈ 22M ops
- Path B: O(k×d) per batch × 9 edge types ≈ 4.5K ops
- Target: 54.80ms → 5-10ms per sample (5-10× speedup)

**Implementation Plan**:

1. **Phase 3.1**: Create `MaskedGraphProcessor` with incidence cache
2. **Phase 3.2**: Modify HeteroCell model to handle edge masks during message passing
3. **Phase 3.3**: Benchmark and validate end-to-end speedup

**Validation Strategy**:

1. Test that MaskedGraphProcessor + mask application = SubgraphRepresentation output
2. Benchmark processor alone and end-to-end training
3. Only commit if measurable improvement achieved

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
