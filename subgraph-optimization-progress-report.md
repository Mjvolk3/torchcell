# SubgraphRepresentation Optimization Progress

**Objective**: Achieve 2x CUDA speedup (from 624ms to 312ms or better)

---

## Phase 0: Setup and Baseline - COMPLETE

**Baseline CUDA Time**: 624.345ms

### Completed Tasks

1. Created stable profiling configuration
   - File: `experiments/006-kuzmin-tmi/conf/profiling_stable_config.yaml`
   - Production settings: batch_size=32, num_workers=2, pin_memory=true

2. Created SLURM profiling script
   - File: `experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm`

3. Built equivalence test suite
   - File: `tests/torchcell/data/test_graph_processor_equivalence.py`
   - Handles unordered attributes (sets stored as lists)
   - Canonical edge sorting using PyG's sort_edge_index()
   - Tests single instance and batch data

4. Generated baseline reference data
   - File: `torchcell/scratch/save_reference_baseline.py`
   - Saved to: `/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/reference_baseline.pkl`

5. Made data loading deterministic
   - Modified: `torchcell/scratch/load_batch_005.py`
   - Added comprehensive seeding (random, numpy, torch, cudnn)
   - Tests pass with production settings (num_workers=2, pin_memory=true)

6. Ran baseline profiling
   - Profile saved: `/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005/profiling_results/baseline_profile.txt`
   - Key bottleneck identified: aten::gather operations consume 242.590ms (38.86% of CUDA time)

7. All tests passing
   - Command: `pytest tests/torchcell/data/test_graph_processor_equivalence.py -xvs`

8. Committed baseline work
   - Commit 1: Profiling infrastructure for dango models
   - Commit 2: SubgraphRepresentation optimization baseline setup

### Key Notes from Phase 0

- Tests initially failed due to non-deterministic data loading - fixed with comprehensive seeding
- Some attributes like `ids_pert` are semantically sets but stored as lists - needed set-based comparison
- Edge ordering differs between runs - needed canonical sorting with PyG's sort_edge_index()
- Batch data has nested lists - needed special handling for list-of-lists comparison
- Production settings work with proper seeding - no need to force num_workers=0

---

## Phase 1: Bipartite Subgraph Optimization - IN PROGRESS

**Expected CUDA Time**: ~380ms (1.6x speedup)
**Target**: Reduce aten::gather operations from 242ms to ~120ms

### Target for Optimization

File: `torchcell/data/graph_processor.py`
Method: `_process_metabolism_bipartite()` (lines 364-395)

Problem: Uses expensive bipartite_subgraph() which performs gather/scatter operations even though we always keep all metabolites (100% of production cases).

Solution: Replace with direct boolean masking and reaction index remapping.

### Tasks Remaining

1. Implement fast path optimization
2. Run equivalence tests
3. Profile and measure speedup
4. Save reference as `reference_opt1_bipartite.pkl`
5. Commit Phase 1

###

 Phase 1 Results

**Implemented**: Fast path optimization completed
**Tests**: All equivalence tests PASSED
**CUDA Time**: 624.987ms (vs baseline 624.345ms)
**Speedup**: None measurable in training time

### Key Finding

The optimization is valid but targets dataset creation, not training:
- Graph processor runs during dataset creation/caching (one-time cost)
- Dataset is cached - profiling uses pre-processed data from August
- Gather/scatter operations in profile (244ms) are from GNN model's message passing, NOT from bipartite_subgraph()
- Optimization improves dataset creation time but not cached training time

### Conclusion

The optimization plan misidentified the bottleneck. The 244ms gather operations are from the model's forward pass (GNN message passing via index_select), not from the graph processor's bipartite_subgraph() call.

To see the actual benefit, we would need to:
1. Clear the cache and measure dataset creation time
2. Or profile dataset creation specifically
3. Or identify optimizations in the GNN model itself

**Decision**: Before continuing with remaining phases, we need to definitively verify whether graph processing is the actual bottleneck. Proceeding with Phase 1.5 - Benchmark Verification.

---

## Phase 1.5: Benchmark Verification - COMPLETE

**Objective**: Definitively verify whether graph processing is the bottleneck by directly benchmarking graph processors on data loading.

### Approach

Created benchmark script that loops through dataset with two different graph processors:
1. **SubgraphRepresentation** (HeteroCell model - with Phase 1 optimization)
2. **Perturbation** (Dango model)

### Benchmark Configuration

- **HeteroCell**: `graphs=[physical, regulatory, tflink, string12_0_*]`, `incidence_graphs=[metabolism_bipartite]`, `node_embeddings=[learnable]`
- **Dango**: `graphs=[string12_0_*]`, no incidence graphs, no node embeddings
- Sample size: 10,000 samples
- Batch size: 32
- Num workers: 2
- Include GPU transfer (`.to('cuda')`) to simulate real training
- Measure wall time for complete data loading loop

### Results (SLURM job bench-processors_334)

**SubgraphRepresentation (HeteroCell)**:
- Total time: 444.54s (10,016 samples)
- Per-sample time: 44.38ms
- Throughput: 22.5 samples/sec

**Perturbation (Dango)**:
- Total time: 4.25s (10,016 samples)
- Per-sample time: 0.42ms
- Throughput: 2354.9 samples/sec

**Relative Performance**: SubgraphRepresentation is **104.52x SLOWER** than Perturbation

### Conclusion

✅ **Graph processing IS the bottleneck** - SubgraphRepresentation is 100x slower than Perturbation

**Decision**: Continue with Phase 2-5 optimizations as planned

**Important Note**:
- These optimizations target dataset creation (one-time cost), not cached training
- To see training benefit, must clear cache and regenerate dataset
- Expected improvement: ~2x speedup in dataset creation time
- Profiling shows no training speedup because data is pre-cached from August

### Files Created

1. `experiments/006-kuzmin-tmi/scripts/benchmark_graph_processors.py` - Benchmark script
2. `experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm` - SLURM submission script

---

## Phase 2: Optimize Mask Indexing - READY TO START

**Expected Impact**: 10-15% speedup in edge filtering operations
**Target**: Replace O(n×m) `torch.isin()` with O(n) boolean indexing

### Target for Optimization

File: `torchcell/data/graph_processor.py`
Method: `_process_reaction_info()` (around lines 296-298)

Problem: Uses expensive `torch.isin()` for edge filtering, which is O(n×m) complexity.

Solution: Replace with boolean mask indexing for O(n) complexity.

### Implementation

Current code:
```python
edge_mask = torch.isin(
    reaction_indices, torch.where(valid_with_genes_mask)[0]
) & torch.isin(gene_indices, gene_info["keep_subset"])
```

Optimized code:
```python
# Create boolean masks for O(1) lookup instead of O(n×m) torch.isin
valid_reactions_mask = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)
valid_reactions_mask[torch.where(valid_with_genes_mask)[0]] = True

keep_genes_mask = torch.zeros(cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device)
keep_genes_mask[gene_info["keep_subset"]] = True

# Direct indexing - O(n) instead of O(n×m)
edge_mask = valid_reactions_mask[reaction_indices] & keep_genes_mask[gene_indices]
```

### Tasks

1. Implement boolean mask optimization in `_process_reaction_info()`
2. Run equivalence tests
3. Save reference as `reference_opt2_mask_indexing.pkl`
4. Run benchmark to measure speedup
5. Commit Phase 2

---

## Phases 3-5: Pending

- Phase 3: Optimize buffer reuse (reduce memory allocations)
- Phase 4: Eliminate device transfers (check device before transfer)
- Phase 5: Cache edge types (pre-filter gene-gene edges)

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Baseline Data Loading | 44.38ms/sample (SubgraphRepresentation) |
| Target | 0.42ms/sample (match Perturbation) |
| Required Speedup | ~100x in dataset creation |
| Current Progress | Phase 0 complete, Phase 1 complete, Phase 1.5 complete |

---

## Test and Benchmark Commands

**Run equivalence tests**:
```bash
pytest tests/torchcell/data/test_graph_processor_equivalence.py -xvs
```

**Run benchmark**:
```bash
sbatch experiments/006-kuzmin-tmi/scripts/benchmark_processors.slurm
```

**Run profiling**:
```bash
sbatch experiments/006-kuzmin-tmi/scripts/profile_single_model.slurm
```

---

## Next Action

Implement Phase 2 mask indexing optimization in `torchcell/data/graph_processor.py`.
