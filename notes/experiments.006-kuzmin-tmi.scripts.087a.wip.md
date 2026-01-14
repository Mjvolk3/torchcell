---
id: oaf8eh2vh0ouhl1eqff8inc
title: Experiment 087a Work in Progress
desc: NeighborSubgraphRepresentation DataLoader Profiling
updated: 1762512228453
created: 1762512228453
---

## Experiment 087a: NeighborSubgraphRepresentation DataLoader Profiling

### Objective

Test if k-hop neighborhood sampling (`NeighborSubgraphRepresentation`) is faster than full-graph masking (`LazySubgraphRepresentation`) for data loading.

**Key Question**: Can we reduce the 56× data pipeline slowdown (Experiment 086a) by only processing k-hop neighborhoods instead of the full 6607-node graph?

### Expected Benefits

- **Smaller subgraphs**: 2-hop neighborhoods contain ~500-1000 nodes vs 6607 full graph
- **Reduced processing**: Only compute masks/edges for relevant neighborhood
- **Faster iteration**: Should approach DANGO speed if graph processing is the bottleneck

### Baseline (Experiment 086a)

- **LazySubgraphRepresentation**: 1.31 it/s (full graph + masks)
- **DANGO (Perturbation)**: 73.82 it/s (minimal data)
- **Ratio**: 56.34× slower (data pipeline only, no model)

### Implementation

#### 1. Core Graph Processor

**File**: `torchcell/data/graph_processor.py`

**Added**: `NeighborSubgraphRepresentation` class (line ~2360)

**Key features**:

- Inherits from `GraphProcessor`
- Configurable `num_hops` parameter (default: 2)
- Uses `torch_geometric.utils.k_hop_subgraph()` to find k-hop neighbors
- Unions neighborhoods across all 9 edge types
- Preserves original node indices (no relabeling)
- Handles metabolism (GPR and RMR edges)

**Import added**:

```python
from torch_geometric.utils import k_hop_subgraph  # Line 12
```

#### 2. Configuration Files

**File**: `experiments/006-kuzmin-tmi/conf/neighbor_subgraph_gh_087_dataloader.yaml`

**Purpose**: Configure experiment 087a for dataloader profiling only (skip model)

**Key settings**:

```yaml
defaults:
  - hetero_cell_bipartite_dango_gi_gh_086  # Inherit from 086
  - _self_

wandb:
  tags: [experiment_087a, dataloader_profiling, neighbor_subgraph, single_gpu, 2hop]

graph_processor:
  type: "neighbor_subgraph"
  num_hops: 2

regression_task:
  execution_mode: "dataloader_profiling"  # Skip model forward
```

**Note**: The config references a `graph_processor` section that may need to be handled differently in the actual training script.

#### 3. Profiling Script

**File**: `experiments/006-kuzmin-tmi/scripts/profile_neighbor_subgraph_087a.py`

**Purpose**: Standalone profiling script that measures dataloader speed

**Configuration**:

- `NUM_HOPS = 2`
- `BATCH_SIZE = 28` (same as 086a)
- `NUM_WORKERS = 4`
- `SUBSET_SIZE = 10000` (same as 086a)
- `MAX_STEPS = 100`

**Process**:

1. Initialize genome, graph, and metabolism
2. Create dataset with `NeighborSubgraphRepresentation(num_hops=2)`
3. Apply normalization transform
4. Create subset datamodule (10K samples)
5. Warmup for 10 batches
6. Profile for 100 steps
7. Calculate average it/s
8. Compare with 086a baseline

**Output**: Prints comparison table and speedup metrics

#### 4. SLURM Submission Script

**File**: `experiments/006-kuzmin-tmi/scripts/run_profiling_087a.slurm`

**Job configuration**:

- Job name: `PROFILE-087a`
- GPU: 1 (single GPU, no DDP)
- Memory: 250GB
- Time: 2 hours
- CPUs: 16

**Process**:

1. Run `profile_neighbor_subgraph_087a.py`
2. Save output to `profiling_results/profiling_087a_<timestamp>/087a_neighbor_subgraph_dataloader.log`
3. Extract metrics (it/s)
4. Calculate speedup vs LazySubgraph (086a)
5. Calculate slowdown vs DANGO (086a)
6. Generate summary report

**Output files**:

```
profiling_results/profiling_087a_<timestamp>/
├── 087a_neighbor_subgraph_dataloader.log  # Full log
└── summary_<timestamp>.txt                # Summary report
```

### Files Created/Modified

**Production code**:

1. `torchcell/data/graph_processor.py` - Added `NeighborSubgraphRepresentation` class

**Experiment files**:
2. `experiments/006-kuzmin-tmi/conf/neighbor_subgraph_gh_087_dataloader.yaml`
3. `experiments/006-kuzmin-tmi/scripts/profile_neighbor_subgraph_087a.py`
4. `experiments/006-kuzmin-tmi/scripts/run_profiling_087a.slurm`

**Scratch file** (for initial testing):
5. `torchcell/scratch/load_neigbor_batch_006.py` - Initial prototype

### Known Issues / TODO

#### Issue 1: Graph Processor Instantiation

**Problem**: The config specifies `graph_processor.type` but the training script may not support dynamic graph processor selection.

**Current config approach**:

```yaml
graph_processor:
  type: "neighbor_subgraph"
  num_hops: 2
```

**Likely needed**: Modify the training script to check config and instantiate the correct processor:

```python
if wandb.config.get("graph_processor", {}).get("type") == "neighbor_subgraph":
    num_hops = wandb.config.graph_processor.get("num_hops", 2)
    graph_processor = NeighborSubgraphRepresentation(num_hops=num_hops)
else:
    graph_processor = LazySubgraphRepresentation()
```

**Status**: ❌ **Not implemented** - profiling script handles this directly, but config-based approach needs work

#### Issue 2: Import Dependencies

**Potential issue**: The profiling script imports `NeighborSubgraphRepresentation` from `graph_processor.py`, but we added it by appending to the file which might have formatting issues.

**Check needed**: Verify the class was added correctly to `graph_processor.py` without syntax errors.

**Status**: ⚠️ **Needs verification**

#### Issue 3: COOLabelNormalizationTransform Import

**File**: `profile_neighbor_subgraph_087a.py` line 25

**Issue**: Incorrect import path caused ImportError.

**Original (incorrect)**:

```python
from torchcell.transforms.regression_to_classification import COOLabelNormalizationTransform
```

**Fixed**:

```python
from torchcell.transforms.coo_regression_to_classification import COOLabelNormalizationTransform
```

**Status**: ✅ **Fixed** (2025-11-07)

#### Issue 4: Follow Batch Configuration

**File**: `profile_neighbor_subgraph_087a.py` line ~183

```python
follow_batch=["perturbation_indices"],
```

**Potential issue**: NeighborSubgraphRepresentation returns smaller subgraphs with different structure. The `follow_batch` parameter may need adjustment or the collation function may need to handle this differently.

**Status**: ⚠️ **Needs testing**

#### Issue 5: Performance Bottleneck - Inefficient Loops

**File**: `torchcell/data/graph_processor.py` NeighborSubgraphRepresentation class

**Problem**: Job 527 hung during warmup. Investigation revealed 4 major performance bottlenecks with inefficient Python loops using `.item()` calls:

1. **Edge filtering** (lines 2439-2443): Looped through millions of edges one-by-one
2. **Perturbed mask creation** (lines 2431-2433): Looped through nodes
3. **GPR edge filtering** (lines 2458-2462): Looped through hyperedges
4. **RMR edge filtering** (lines 2477-2479): Looped through hyperedges

**Impact**: With 9 edge types and millions of edges, a single batch could take hours.

**Fix applied** (2025-11-07):

1. **Edge filtering** - Vectorized with `torch.isin()`:

```python
# Before: for loop with .item() calls
# After:
src_mask = torch.isin(edge_index[0], subset_nodes)
dst_mask = torch.isin(edge_index[1], subset_nodes)
mask = src_mask & dst_mask
```

2. **Perturbed mask** - Direct indexing:

```python
# Before: for loop
# After:
perturbed_mask = self.masks["gene"]["perturbed"][subset_nodes]
```

3. **GPR filtering** - Vectorized:

```python
# Before: for loop with .item() calls
# After:
gpr_mask = torch.isin(gpr_hyperedge_index[0], subset_nodes)
included_reaction_indices = gpr_hyperedge_index[1, gpr_mask].unique()
```

4. **RMR filtering** - Vectorized:

```python
# Before: for loop with .item() calls
# After:
rmr_mask = torch.isin(rmr_hyperedge_index[0], included_reaction_indices)
```

**Expected speedup**: 100-1000× faster for graph processing operations

**Status**: ✅ **Fixed** (2025-11-07)

### How to Run

```bash
cd /home/michaelvolk/Documents/projects/torchcell
sbatch experiments/006-kuzmin-tmi/scripts/run_profiling_087a.slurm
```

**Expected output location**:

```
experiments/006-kuzmin-tmi/profiling_results/profiling_087a_<timestamp>/
experiments/006-kuzmin-tmi/slurm/output/PROFILE-087a_<job_id>.out
```

### Expected Results

**Best case** (>5× speedup):

- NeighborSubgraph: ~7-10 it/s
- Speedup: 5-8× vs LazySubgraph
- Conclusion: K-hop sampling successfully reduces overhead

**Good case** (2-5× speedup):

- NeighborSubgraph: ~3-6 it/s
- Speedup: 2-5× vs LazySubgraph
- Conclusion: Meaningful improvement, worth exploring further

**Poor case** (<2× speedup):

- NeighborSubgraph: ~1.5-2.5 it/s
- Speedup: <2× vs LazySubgraph
- Conclusion: Bottleneck is elsewhere (collation, I/O, etc.)

### Experiment 087a Results (Job 528)

**2-hop NeighborSubgraph (vectorized)**:

- **Speed**: 2.479 it/s
- **Speedup vs LazySubgraph**: 1.89×
- **Slowdown vs DANGO**: 29.78×
- **Status**: ⚠️ Moderate improvement

**Conclusion**: Vectorization fixed the hang, but 1.89× speedup is below target (hoped for >5×).

### Multi-Hop Comparison (Updated Experiment Design)

**Objective**: Compare 1-hop, 2-hop, and 3-hop to find optimal speed/memory/context tradeoff.

**Updated Scripts** (2025-11-07):

1. **Profiling script enhancements** (`profile_neighbor_subgraph_087a.py`):
   - Added `argparse` for `--num-hops` parameter
   - Added memory tracking:
     - CPU memory per batch (via `psutil`)
     - GPU memory per batch (via `torch.cuda.max_memory_allocated()`)
     - Average subgraph size (nodes/edges per sample)
   - Added batch size estimation (60GB GPU available for data)

2. **SLURM script redesign** (`run_profiling_087a.slurm`):
   - Runs 3 sequential experiments: 1-hop, 2-hop, 3-hop
   - Extracts metrics from all logs
   - Generates comprehensive comparison table:

     ```
     | Hop | it/s | Speedup | GPU Mem | Est. Max Batch | Nodes/sample | Edges/sample |
     ```

   - Calculates effective throughput (samples/sec = it/s × max_batch_size)

**Key Insight**: Smaller neighborhoods may enable larger batch sizes, potentially offsetting slower per-batch speed with higher throughput.

**Next run**: Job 529+ will test all three hop counts and provide data-driven recommendation.

### Next Steps

**After multi-hop results**:

1. **If 1-hop with large batch has best throughput**:
   - Test model with 1-hop to verify acceptable performance
   - Increase batch size to maximize GPU utilization

2. **If 2-hop is optimal**:
   - Continue with current settings
   - Consider PyG C++ NeighborSampler for 2-3× additional speedup

3. **If 3-hop is needed**:
   - Reduce batch size accordingly
   - May need preprocessing or other optimizations

4. **If all show <3× improvement**:
   - Consider switching to PyG NeighborLoader architecture
   - Explore preprocessing approach (save subgraphs to disk)
   - Test hybrid DANGO + graph features approach

### Previous Errors (Fixed)

**Before running, verify**:

1. ✅ **Syntax check**: Verify `graph_processor.py` compiles without errors

   ```bash
   python -c "from torchcell.data.graph_processor import NeighborSubgraphRepresentation; print('OK')"
   ```

2. ⚠️ **Import check**: Verify all imports in profiling script work

   ```bash
   python -c "from experiments.006-kuzmin-tmi.scripts.profile_neighbor_subgraph_087a import main; print('OK')"
   ```

3. ⚠️ **Config validation**: Ensure YAML config is valid

   ```bash
   python -c "import yaml; yaml.safe_load(open('experiments/006-kuzmin-tmi/conf/neighbor_subgraph_gh_087_dataloader.yaml')); print('OK')"
   ```

4. ⚠️ **Data loading test**: Run a quick manual test

   ```bash
   python torchcell/scratch/load_neigbor_batch_006.py
   ```

### Debugging Strategy

**If SLURM job fails**:

1. Check SLURM output file for errors:

   ```bash
   cat experiments/006-kuzmin-tmi/slurm/output/PROFILE-087a_*.out
   ```

2. Check specific error types:
   - **Import errors**: Fix import paths
   - **CUDA OOM**: Reduce batch size or num_hops
   - **Syntax errors**: Fix graph_processor.py formatting
   - **DataLoader errors**: Check collation function compatibility

3. Test locally first (without SLURM):

   ```bash
   python experiments/006-kuzmin-tmi/scripts/profile_neighbor_subgraph_087a.py
   ```

### Comparison Table

| Component | LazySubgraph (086a) | NeighborSubgraph (087a) | Expected Change |
|-----------|---------------------|-------------------------|-----------------|
| **Graph size** | 6607 nodes | ~500-1000 nodes | 85-90% smaller |
| **Edge processing** | All 9 edge types, full graph | All 9 types, k-hop only | 85-95% fewer edges |
| **Mask computation** | O(k×d) for all nodes | O(k×d) for subset | ~10× faster |
| **Memory per sample** | ~2-5 MB | ~0.2-0.5 MB | 80-90% smaller |
| **it/s (target)** | 1.31 it/s | >5 it/s | >3× faster |

### References

- **Experiment 086 summary**: `notes/experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy.2025.11.07.experiment-086-results-summary.md`
- **Scratch prototype**: `torchcell/scratch/load_neigbor_batch_006.py`
- **PyG k_hop_subgraph docs**: <https://pytorch-geometric.readthedocs.io/en/2.5.3/modules/utils.html#torch_geometric.utils.k_hop_subgraph>
