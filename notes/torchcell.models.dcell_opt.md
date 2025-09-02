---
id: cshdbmlwu123va68dg6pr16
title: Dcell_opt
desc: ''
updated: 1748034112517
created: 1747191393026
---

## 2025.08.29 - Model Update for Torch Compile Optimization

### Problem Analysis

The current DCell implementation in `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py` experiences significant graph breaks when using `torch.compile` due to:

1. **Dictionary Lookups with Tensor Keys** (Lines 56-57, 303, 310, 358):
   - `self.subsystems = nn.ModuleDict()` - String keys from tensor values
   - `self.subsystems[str(term_idx_int)]` - Causes graph break at line 303
   - `self.parent_to_children.get(term_idx, [])` - Dynamic dictionary lookup at line 358
   - `term_activations[child_idx]` - Runtime dictionary access

2. **Tensor to Python Scalar Conversions** (Lines 292, 326, 240):
   - `term_idx_int = int(term_idx)` - Graph break from .item() replacement
   - `root_term_idx = int(root_terms[0])` - Materialization forcing
   - `rows_per_sample = int(ptr[1] - ptr[0])` - Multiple conversions

3. **Dynamic Control Flow**:
   - Iterating through `stratum_terms` with variable content
   - Different GO terms have different numbers of children/genes
   - Hierarchical processing with data-dependent paths

### Current Performance Impact

From `/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py` (lines 409-439):

- torch.compile is attempted with `mode="default"` and `dynamic=True`
- Multiple recompilation warnings indicate hitting `config.recompile_limit`
- Graph breaks reduce potential speedup from 10x to ~2-3x
- Model runs at ~0.03 it/s (33 seconds per iteration) with graph breaks

### Proposed Solution: DCellOpt with Tensorized Architecture

Create `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell_opt.py` with the following optimizations:

#### 1. Replace ModuleDict with ModuleList

**Current Implementation** (dcell.py lines 56-57, 178):

```python
self.subsystems = nn.ModuleDict()
self.linear_heads = nn.ModuleDict()
# Later...
self.subsystems[str(term_idx)] = subsystem
```

**Optimized Implementation**:

```python
class DCellOpt(nn.Module):
    def __init__(self, hetero_data: HeteroData, ...):
        # Pre-allocate ModuleList with fixed size
        self.num_go_terms = hetero_data["gene_ontology"].num_nodes  # 2655 terms
        
        # Use ModuleList with None placeholders
        self.subsystems = nn.ModuleList([None] * self.num_go_terms)
        self.linear_heads = nn.ModuleList([None] * self.num_go_terms)
        
        # Track which indices have modules
        self.has_subsystem = torch.zeros(self.num_go_terms, dtype=torch.bool)
        
        # Build modules
        for term_idx in range(self.num_go_terms):
            if self._term_needs_subsystem(term_idx):
                self.subsystems[term_idx] = DCellSubsystem(input_dim, output_dim)
                self.linear_heads[term_idx] = nn.Linear(output_dim, self.output_size)
                self.has_subsystem[term_idx] = True
```

#### 2. Tensorize Hierarchy Representations

**Current Implementation** (dcell.py lines 50, 112-122):

```python
self.parent_to_children = self._build_parent_to_children()  # Dict[int, List[int]]
# Dictionary lookup in forward:
children = self.parent_to_children.get(term_idx, [])
```

**Optimized Implementation**:

```python
def _build_tensorized_hierarchy(self):
    # Create adjacency matrix for parent-child relationships
    self.parent_child_adj = torch.zeros(self.num_go_terms, self.num_go_terms, dtype=torch.bool)
    
    for parent, children in self.parent_to_children.items():
        for child in children:
            self.parent_child_adj[parent, child] = True
    
    # Pre-compute max children for padding
    self.max_children = max(len(children) for children in self.parent_to_children.values())
    
    # Create padded children tensor
    self.children_indices = torch.full(
        (self.num_go_terms, self.max_children), 
        -1,  # Use -1 for padding
        dtype=torch.long
    )
    self.num_children = torch.zeros(self.num_go_terms, dtype=torch.long)
    
    for parent, children in self.parent_to_children.items():
        num_children = len(children)
        self.children_indices[parent, :num_children] = torch.tensor(children)
        self.num_children[parent] = num_children
```

#### 3. Convert Stratum Processing to Tensor Operations

**Current Implementation** (dcell.py lines 45, 286):

```python
self.stratum_to_terms = hetero_data["gene_ontology"].stratum_to_terms  # Dict
stratum_terms = self.stratum_to_terms[stratum]  # Dictionary lookup
```

**Optimized Implementation**:

```python
def _build_stratum_masks(self):
    # Create binary masks for each stratum
    self.max_stratum = max(self.stratum_to_terms.keys())
    self.stratum_masks = torch.zeros(
        self.max_stratum + 1, 
        self.num_go_terms, 
        dtype=torch.bool
    )
    
    for stratum, terms in self.stratum_to_terms.items():
        for term in terms:
            self.stratum_masks[stratum, term] = True
    
    # Pre-compute terms per stratum for efficient iteration
    self.stratum_term_lists = []
    for stratum in range(self.max_stratum + 1):
        terms = torch.where(self.stratum_masks[stratum])[0]
        self.stratum_term_lists.append(terms)
```

#### 4. Optimized Forward Pass

**Current Implementation** (dcell.py lines 260-344):

- Uses dictionary lookups for subsystems and activations
- Iterates through terms with Python control flow
- Multiple tensor-to-scalar conversions

**Optimized Implementation**:

```python
def forward(self, cell_graph: HeteroData, batch: HeteroData) -> Tuple[torch.Tensor, Dict[str, Any]]:
    batch_size = batch["gene"].batch.max() + 1  # Keep as tensor longer
    device = batch["gene"].x.device
    
    # Pre-allocate ALL term activations
    max_output_dim = max(self.term_output_dims.values())
    all_activations = torch.zeros(
        batch_size, 
        self.num_go_terms, 
        max_output_dim,
        device=device
    )
    activation_mask = torch.zeros(
        batch_size,
        self.num_go_terms,
        dtype=torch.bool,
        device=device
    )
    
    # Pre-allocate linear outputs
    linear_outputs_tensor = torch.zeros(
        batch_size,
        self.num_go_terms,
        self.output_size,
        device=device
    )
    
    # Process strata in descending order (leaves to root)
    for stratum_idx in range(self.max_stratum, -1, -1):
        # Get all terms in this stratum using pre-computed tensor
        stratum_terms = self.stratum_term_lists[stratum_idx]
        
        if len(stratum_terms) == 0:
            continue
        
        # Process all terms in parallel where possible
        for term_idx in stratum_terms:
            term_idx_int = term_idx.item() if torch.is_tensor(term_idx) else term_idx
            
            # Skip if no subsystem
            if not self.has_subsystem[term_idx_int]:
                continue
            
            # Prepare input using tensorized operations
            term_input = self._prepare_term_input_optimized(
                term_idx_int, 
                batch, 
                all_activations,
                activation_mask
            )
            
            # Direct indexing - no dictionary lookup!
            subsystem = self.subsystems[term_idx_int]
            linear_head = self.linear_heads[term_idx_int]
            
            # Forward through subsystem
            term_output = subsystem(term_input)
            
            # Store in pre-allocated tensor
            output_dim = term_output.size(1)
            all_activations[:, term_idx_int, :output_dim] = term_output
            activation_mask[:, term_idx_int] = True
            
            # Linear output
            linear_output = linear_head(term_output)
            linear_outputs_tensor[:, term_idx_int, :] = linear_output
    
    # Extract root prediction (stratum 0)
    root_term_idx = self.stratum_term_lists[0][0].item()
    predictions = linear_outputs_tensor[:, root_term_idx, :].squeeze(-1)
    
    # Convert tensors back to dictionary for compatibility
    # This happens AFTER all computation, minimizing graph breaks
    linear_outputs = {}
    for term_idx in range(self.num_go_terms):
        if activation_mask[0, term_idx]:  # Check if term was processed
            linear_outputs[f"GO:{term_idx}"] = linear_outputs_tensor[:, term_idx, :].squeeze(-1)
    
    linear_outputs["GO:ROOT"] = predictions
    
    outputs = {
        "linear_outputs": linear_outputs,
        "all_activations_tensor": all_activations,  # Keep tensor version
        "activation_mask": activation_mask,
        "stratum_outputs": {}
    }
    
    return predictions, outputs
```

#### 5. Optimized Term Input Preparation

**Current Implementation** (dcell.py lines 346-376):

- Dictionary lookups for children
- Dynamic list building

**Optimized Implementation**:

```python
def _prepare_term_input_optimized(
    self, 
    term_idx: int, 
    batch: HeteroData,
    all_activations: torch.Tensor,
    activation_mask: torch.Tensor
) -> torch.Tensor:
    batch_size = all_activations.size(0)
    device = all_activations.device
    
    # Get children using pre-computed indices
    children_indices = self.children_indices[term_idx]  # [max_children]
    num_children = self.num_children[term_idx]
    
    # Extract child activations using advanced indexing
    if num_children > 0:
        valid_children = children_indices[:num_children]
        child_activations = all_activations[:, valid_children, :]  # [batch, num_children, max_dim]
        
        # Mask and flatten
        child_mask = activation_mask[:, valid_children]  # [batch, num_children]
        masked_child_acts = child_activations * child_mask.unsqueeze(-1)
        
        # Flatten children dimension
        child_input = masked_child_acts.reshape(batch_size, -1)
    else:
        child_input = torch.zeros(batch_size, 1, device=device)
    
    # Get gene states (keep existing optimized version)
    gene_states = self._extract_gene_states_for_term(term_idx, batch)
    
    # Concatenate
    return torch.cat([child_input, gene_states], dim=1)
```

### Memory Analysis

For 2655 GO terms with batch_size=400:

1. **Activation Storage**:
   - `all_activations`: 400 × 2655 × 256 × 4 bytes = ~1.1 GB
   - `activation_mask`: 400 × 2655 × 1 byte = ~1 MB
   - `linear_outputs_tensor`: 400 × 2655 × 1 × 4 bytes = ~4 MB

2. **Model Storage**:
   - ModuleList overhead: ~2655 pointers (negligible)
   - Hierarchy tensors: ~2655² × 1 byte = ~7 MB for adjacency
   - Children indices: 2655 × max_children × 8 bytes = ~0.5 MB

3. **Total Additional Memory**: ~1.2 GB per batch
   - Acceptable on modern GPUs (V100 has 32GB, A100 has 40-80GB)

### Expected Benefits

1. **Compilation Improvements**:
   - Eliminate dictionary lookup graph breaks
   - Reduce recompilations from dynamic shapes
   - Enable better kernel fusion
   - Potential 3-5x speedup over current implementation

2. **Maintained Functionality**:
   - Same model accuracy (no algorithmic changes)
   - Compatible with existing training pipeline
   - Backward compatible output format

3. **Trade-offs**:
   - +1.2 GB memory per batch
   - Slightly more complex initialization
   - Some wasted computation on padded/empty slots

### Implementation Plan

1. **Phase 1: Create dcell_opt.py**
   - Copy base structure from dcell.py
   - Implement ModuleList conversion
   - Add tensorized hierarchy structures

2. **Phase 2: Optimize Forward Pass**
   - Replace dictionary operations with tensor indexing
   - Implement batched stratum processing
   - Minimize Python scalar conversions

3. **Phase 3: Integration**
   - Update `/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py` to support DCellOpt
   - Add config flag `model.use_optimized: true`
   - Ensure checkpoint compatibility

4. **Phase 4: Testing**
   - Verify numerical equivalence with original DCell
   - Benchmark torch.compile performance
   - Profile memory usage

### Configuration Changes

In `experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi.yaml`:

```yaml
model:
  use_optimized: true  # Enable DCellOpt
  compile_mode: "default"  # or "reduce-overhead" for more aggressive optimization
  tensorize_hierarchy: true
  preallocate_activations: true
```

### Validation Approach

1. **Correctness Testing**:

   ```python
   # Compare outputs between DCell and DCellOpt
   with torch.no_grad():
       out_original, _ = dcell_original(cell_graph, batch)
       out_optimized, _ = dcell_opt(cell_graph, batch)
       assert torch.allclose(out_original, out_optimized, rtol=1e-5)
   ```

2. **Performance Benchmarking**:

   ```python
   # Measure graph breaks
   import torch._dynamo as dynamo
   dynamo.reset()
   dynamo.config.verbose = True
   
   # Count recompilations
   with dynamo.optimize("inductor"):
       for i in range(100):
           _ = model(cell_graph, batch)
   ```

3. **Memory Profiling**:

   ```python
   import torch.profiler as profiler
   with profiler.profile(activities=[profiler.ActivityType.CUDA],
                         profile_memory=True) as prof:
       _ = model(cell_graph, batch)
   print(prof.key_averages().table(sort_by="cuda_memory_usage"))
   ```

### Key Insights from Analysis

1. **Dynamic Shapes Are OK**: With `torch.compile(dynamic=True)`, variable batch sizes and gene counts are handled efficiently. The issue is dictionary lookups, not dynamic tensors.

2. **GO Term Count is Fixed**: All 2655 GO terms are known at initialization, making pre-allocation feasible.

3. **Sparse but Bounded**: Not all GO terms have subsystems, but we know the maximum count, allowing efficient masking.

4. **Hierarchy is Static**: The GO term relationships don't change during training, perfect for tensorization.

### References

- Original DCell: `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py`
- Training Script: `/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py`
- torch.compile docs: <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- PyTorch ModuleList: <https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html>
