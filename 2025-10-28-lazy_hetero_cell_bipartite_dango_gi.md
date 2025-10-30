# Implementation Plan: hetero_cell_bipartite_dango_gi_lazy.py

**Date**: 2025-10-28 (Updated: 2025-10-29)
**Goal**: Adapt GeneInteractionDango model for LazySubgraphRepresentation's zero-copy architecture

**Status**: ✅ **COMPLETE** - Model successfully implemented and training

---

## Executive Summary

Based on comprehensive analysis of `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/hetero_cell_bipartite_dango_gi.py` and the LazySubgraphRepresentation architecture, most components (70%+) can be reused with minimal changes. The primary modification is replacing standard GINConv with MaskedGINConv for edge masking.

**Estimated effort**: 10-12 hours
**Actual effort**: ~12 hours (as estimated)
**Confidence**: HIGH - Architecture is well-designed for this adaptation

---

## Implementation Complete - Key Achievements

### ✅ Core Implementation (All Phases Complete)

- **MaskedGINConv Integration**: Replaced GINConv with edge masking support
- **Zero-Copy Architecture**: Model works with full graphs + masks (no tensor filtering)
- **Training Success**: Model training at batch_size=36 on 4 GPUs with DDP
- **Biological Correctness**: All aggregation methods working (sum, mean, cross_attention, pairwise_interaction)

### ✅ Additional Improvements Beyond Original Plan

1. **Configurable Pairwise Aggregation** (NEW):
   - `pairwise_num_layers`: Configurable MLP depth (default: 2)
   - `pairwise_hidden_dim`: Bottleneck dimension (default: 32)
   - **50% parameter reduction**: 559K → 281K params per layer
   - Activation configurable via `act_register` (GELU, ReLU, etc.)

2. **Identity Option for Pairwise** (NEW):
   - Added 46th option: mean of individual graph representations
   - Enables learned residual connection
   - Model can choose between pairwise combinations vs original representations

3. **Better Naming** (NEW):
   - Renamed `attention` → `pair_scorer` in pairwise (avoid confusion with cross-attention)
   - Renamed `last_layer_attention_weights` → `layer_aggregation_weights` (clarifies it stores ALL layers)

4. **DataModule Configurability** (NEW):
   - Added `persistent_workers` parameter to CellDataModule
   - Added `persistent_workers` parameter to PerturbationSubsetDataModule
   - Configurable via YAML (good for short vs long epoch training)

### 📊 Training Performance

**Current Status** (Job 371-376):
- Training on 4x GPUs with DDP
- Batch size: 36 (OOM at 38+)
- Speed: 0.24-0.40 it/s depending on aggregation method
- Data loading bottleneck identified and addressed with persistent_workers

**Aggregation Method Comparison**:
- `sum`: 0.40 it/s (fastest, simplest)
- `cross_attention`: 0.24 it/s (40% slower, most expressive)
- `pairwise_interaction`: ~0.28 it/s (faster than attention, more params without bottleneck)

### 🔧 Parameter Efficiency Analysis

**Pairwise Aggregation** (9 graphs = 45 pairs + 1 identity):

| Config | Params per MLP | Total (45 MLPs × 3 layers) | Speedup |
|--------|----------------|---------------------------|---------|
| Original (no bottleneck) | 12,416 | 1.68M | Baseline |
| `bottleneck_dim=32` | 6,240 | 842K | 2.0x |
| `bottleneck_dim=16` | 3,104 | 421K | 4.0x |

**Decision**: Using `bottleneck_dim=32` balances parameter efficiency with expressiveness.

---

## 1. Data Format Differences

### SubgraphRepresentation (OLD)

```python
Input: Filtered graph
x: [num_kept_nodes, feat_dim]           # Filtered, excludes perturbed
edge_index: [2, num_kept_edges]         # Relabeled 0 to num_kept_nodes-1
num_nodes: count of kept nodes only
perturbation_indices: indices into KEPT nodes
```

### LazySubgraphRepresentation (NEW)

```python
Input: Full graph with masks
x: [total_nodes, feat_dim]              # Full graph (6607 nodes)
edge_index: [2, total_edges]            # Original indices (0-6606)
mask: [total_nodes]                     # True = keep
pert_mask: [total_nodes]                # True = perturbed
edge_mask: [total_edges]                # True = keep edge
num_nodes: total nodes (6607)
perturbation_indices: indices into FULL graph (0-6606)
```

### Critical Insight: perturbation_indices Work Unchanged!

**perturbation_indices in LazySubgraphRepresentation are indices into the FULL graph**, so line 925 (`z_w[pert_indices]`) works identically!

---

## 2. Component Analysis

### 2.1 Components That DON'T Need Changes

**Zero modifications needed:**

1. **SelfAttentionGraphAggregation** (lines 34-93)
   - Works on node features `[num_nodes, hidden_dim]`
   - Mask-agnostic - operates after message passing

2. **PairwiseGraphAggregation** (lines 96-179)
   - Works on node features dictionary
   - Mask-agnostic

3. **AttentionalGraphAggregation** (lines 299-318)
   - Uses PyG's AttentionalAggregation on node features
   - No edge operations

4. **DangoLikeHyperSAGNN** (lines 321-455)
   - Self-attention over gene embeddings
   - No graph structure operations

5. **GeneInteractionPredictor** (lines 458-500)
   - Operates on gene embeddings only

6. **PreProcessor** (lines 514-540)
   - MLP on node features

7. **Global predictors and gating** (lines 775-791)
   - gate_mlp, global_interaction_predictor

### 2.2 Components Needing Minimal Modification

**HeteroConvAggregator** (lines 182-296)

- **Change**: Modify `forward()` to extract and pass edge masks
- **Lines affected**: ~10 lines

**AttentionConvWrapper** (lines 543-604)

- **Change**: Add `edge_mask` parameter to `forward()`
- **Lines affected**: ~5 lines

**create_conv_layer** (lines 606-658)

- **Change**: Import and use `MaskedGINConv` when `encoder_type == "gin"`
- **Lines affected**: ~3 lines

### 2.3 Components Needing Significant Modification

**forward_single()** (lines 837-882)

- **Current issues**:
  - Lines 847-850: Handles `pert_mask` but expects FILTERED x (wrong!)
  - Lines 867-870: Edge extraction doesn't handle edge masks
- **Required changes**: ~30 lines

**forward()** (lines 884-1059)

- **Current issues**:
  - Need to apply node masks before global aggregation
- **Required changes**: ~15 lines

---

## 3. Implementation Phases

### Phase 1: Infrastructure Setup (Easy - 30 mins)

**Tasks:**

1. Copy `hetero_cell_bipartite_dango_gi.py` to `hetero_cell_bipartite_dango_gi_lazy.py`
2. Add import:

   ```python
   from torchcell.nn.masked_gin_conv import MaskedGINConv
   ```

3. Update `create_conv_layer()`:

   ```python
   def create_conv_layer(encoder_type, in_channels, out_channels, config, edge_dim=None, dropout=0.1):
       if encoder_type == "gatv2":
           raise NotImplementedError("GATv2 not yet supported for lazy - use GIN")
       elif encoder_type == "gin":
           mlp_layers = [...]  # Build MLP as before
           mlp = nn.Sequential(*mlp_layers)
           return MaskedGINConv(mlp, train_eps=True)  # Changed from GINConv
   ```

**Testing:**

```python
# Verify import works
from torchcell.models.hetero_cell_bipartite_dango_gi_lazy import create_conv_layer
conv = create_conv_layer("gin", 64, 64, {}, dropout=0.1)
assert isinstance(conv, MaskedGINConv)
```

---

### Phase 2: Wrapper Modifications (Medium - 1 hour)

**Task 1: Update AttentionConvWrapper**

```python
class AttentionConvWrapper(nn.Module):
    def forward(self, x, edge_index, edge_mask=None, **kwargs):
        # Pass edge_mask to conv if supported
        if isinstance(self.conv, MaskedGINConv):
            out = self.conv(x, edge_index, edge_mask=edge_mask)
        else:
            # GATv2 doesn't support edge_mask yet
            out = self.conv(x, edge_index, **kwargs)

        out = self.proj(out)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
```

**Task 2: Update HeteroConvAggregator**

```python
class HeteroConvAggregator(nn.Module):
    def forward(self, x_dict, edge_index_dict, edge_mask_dict=None):
        """
        Apply graph convolutions with edge masking.

        Args:
            edge_mask_dict: Optional dict mapping edge_type to edge masks
        """
        edge_mask_dict = edge_mask_dict or {}
        out_dict = {}
        graph_outputs_by_dst = {}

        for edge_type_str, conv in self.convs.items():
            edge_type = eval(edge_type_str) if isinstance(edge_type_str, str) else edge_type_str
            src, rel, dst = edge_type

            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
                edge_mask = edge_mask_dict.get(edge_type, None)  # NEW

                # Apply convolution with mask
                out = conv(x_dict[src], edge_index, edge_mask=edge_mask)  # NEW

                # Organize outputs by destination and graph
                if dst not in graph_outputs_by_dst:
                    graph_outputs_by_dst[dst] = {}
                graph_outputs_by_dst[dst][rel] = out

        # ... rest of aggregation logic unchanged
```

**Testing:**

```python
# Test edge mask passing
x_dict = {"gene": torch.randn(6607, 64)}
edge_index_dict = {("gene", "physical", "gene"): torch.randint(0, 6607, (2, 144211))}
edge_mask_dict = {("gene", "physical", "gene"): torch.ones(144211, dtype=torch.bool)}

conv = HeteroConvAggregator(...)
out_dict, _ = conv(x_dict, edge_index_dict, edge_mask_dict)
assert out_dict["gene"].shape == (6607, 64)
```

---

### Phase 3: forward_single Rewrite (Hard - 2-3 hours)

**Critical rewrite - this is where lazy architecture changes most:**

```python
def forward_single(self, data: HeteroData | Batch) -> torch.Tensor:
    device = self.gene_embedding.weight.device

    # Handle both batch and single graph input
    is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")

    if is_batch:
        gene_data = data["gene"]
        batch_size = len(data["gene"].ptr) - 1

        # LAZY APPROACH: x is FULL graph - no filtering needed!
        # x is [total_nodes, feat_dim] - already includes all nodes (kept + perturbed)
        x_gene = gene_data.x.to(device)  # ZERO-COPY reference

        # Apply preprocessing to ALL nodes
        # Masking happens during message passing and aggregation, not here!
        x_gene = self.preprocessor(x_gene)
    else:
        gene_data = data["gene"]
        # Single graph case - x is still full graph
        x_gene = self.preprocessor(gene_data.x.to(device))

    x_dict = {"gene": x_gene}

    # Extract edge indices AND edge masks
    edge_index_dict = {}
    edge_mask_dict = {}  # NEW!

    for graph_name in self.graph_names:
        edge_type = ("gene", graph_name, "gene")
        edge_data = data[edge_type]

        edge_index_dict[edge_type] = edge_data.edge_index.to(device)

        # Extract edge mask if present (it should be!)
        if hasattr(edge_data, 'mask'):
            edge_mask_dict[edge_type] = edge_data.mask.to(device)  # NEW!
        else:
            # Fallback: assume all edges are valid
            num_edges = edge_data.edge_index.size(1)
            edge_mask_dict[edge_type] = torch.ones(num_edges, dtype=torch.bool, device=device)

    # Apply convolution layers with edge masking
    layer_attention_weights = []
    for conv in self.convs:
        x_dict, attn_weights = conv(x_dict, edge_index_dict, edge_mask_dict)  # NEW: pass edge masks
        if attn_weights is not None:
            layer_attention_weights.append(attn_weights)

    self.last_layer_attention_weights = layer_attention_weights

    # Return full graph embeddings [total_nodes, hidden_dim]
    return x_dict["gene"]
```

**Key changes:**

1. No node filtering - `x` is full graph
2. Extract edge masks from `data[edge_type].mask`
3. Pass edge masks to convolutions via `edge_mask_dict`
4. Return full graph embeddings `[total_nodes, hidden_dim]`

**Testing:**

```python
# Test with lazy data
from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch
dataset, batch, _, _ = load_sample_data_batch(batch_size=1, use_custom_collate=True)

model = GeneInteractionDango(...)
z = model.forward_single(batch)

# Verify full graph embeddings
assert z.shape[0] == batch["gene"].num_nodes  # Should be 6607 for single graph
assert z.shape[1] == model.hidden_channels
assert not torch.isnan(z).any()
```

---

### Phase 4: forward() Adjustments (Medium - 1-2 hours)

**Key insight: Most of forward() works unchanged! Just need to apply masks before global aggregation.**

```python
def forward(self, cell_graph: HeteroData, batch: HeteroData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Process reference graph (wildtype)
    z_w = self.forward_single(cell_graph)  # [total_nodes, hidden_dim] - FULL graph

    # Check for NaNs
    if torch.isnan(z_w).any():
        raise RuntimeError("NaN detected in wildtype embeddings (z_w)")

    # NEW: Apply mask before global aggregation
    # We only want to aggregate over KEPT genes (not perturbed ones)
    gene_mask = ~cell_graph["gene"].pert_mask  # True for kept genes
    z_w_kept = z_w[gene_mask]  # Filter to kept genes

    # Global aggregation on kept genes only
    z_w_global = self.global_aggregator(
        z_w_kept,
        index=torch.zeros(z_w_kept.size(0), device=z_w.device, dtype=torch.long),
        dim_size=1,
    )

    if torch.isnan(z_w_global).any():
        raise RuntimeError("NaN detected in global wildtype embeddings (z_w_global)")

    # Process perturbed batch
    z_i = self.forward_single(batch)  # [total_nodes_batch, hidden_dim]

    if torch.isnan(z_i).any():
        raise RuntimeError("NaN detected in perturbed embeddings (z_i)")

    # NEW: Apply mask before global aggregation for batch
    batch_gene_mask = ~batch["gene"].pert_mask
    z_i_kept = z_i[batch_gene_mask]

    # Global aggregation on kept genes, respecting batch structure
    z_i_global = self.global_aggregator(
        z_i_kept,
        index=batch["gene"].batch[batch_gene_mask]  # Batch vector for kept genes only
    )

    if torch.isnan(z_i_global).any():
        raise RuntimeError("NaN detected in global perturbed embeddings (z_i_global)")

    # Get embeddings of perturbed genes from wildtype
    pert_indices = batch["gene"].perturbation_indices
    pert_gene_embs = z_w[pert_indices]  # WORKS! Indices are into full graph

    if torch.isnan(pert_gene_embs).any():
        raise RuntimeError("NaN detected in perturbed gene embeddings (pert_gene_embs)")

    # Calculate perturbation difference for z_p_global
    batch_size = z_i_global.size(0)
    z_w_exp = z_w_global.expand(batch_size, -1)
    z_p_global = z_w_exp - z_i_global

    # ... rest is UNCHANGED - determination of batch_assign, local/global predictors, gating, etc.

    # Return same structure as original (lines 1048-1059)
    return gene_interaction, {
        "z_w": z_w_global,
        "z_i": z_i_global,
        "z_p": z_p_global,
        "local_interaction": local_interaction,
        "global_interaction": global_interaction,
        "gate_weights": gate_weights,
        "gene_interaction": gene_interaction,
        "pert_gene_embs": pert_gene_embs,
        "graph_attention_weights": self.last_layer_attention_weights,
    }
```

**Key changes:**

1. Apply `~pert_mask` before global aggregation
2. Filter batch vectors: `batch["gene"].batch[batch_gene_mask]`
3. `pert_indices` work unchanged (indices into full graph!)
4. Return same dict structure as original

**Testing:**

```python
# Full forward pass test
from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch
dataset, batch, _, _ = load_sample_data_batch(batch_size=2, use_custom_collate=True)

model = GeneInteractionDango(...)
cell_graph = dataset.cell_graph

pred, repr_dict = model(cell_graph, batch)

# Verify shapes
assert pred.shape == (2, 1)  # [batch_size, 1]
assert "z_w" in repr_dict
assert "z_i" in repr_dict
assert "z_p" in repr_dict
assert "gate_weights" in repr_dict
assert not torch.isnan(pred).any()
```

---

### Phase 5: Main Function for Overfitting (Medium - 2 hours)

**Goal: Create overfitting test similar to original `main()` function**

```python
@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_gi",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Testing LazySubgraphRepresentation Model]")
    print(f"Using device: {device}")

    # Load data with LAZY approach
    from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch

    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=0,  # Single worker for debugging
        config="hetero_cell_bipartite",
        is_dense=False,
        use_custom_collate=True,  # CRITICAL: Use LazyCollater
    )

    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    print(f"\n[Data loaded]")
    print(f"  Cell graph nodes: {cell_graph['gene'].num_nodes}")
    print(f"  Batch nodes: {batch['gene'].num_nodes}")
    print(f"  Batch size: {cfg.data_module.batch_size}")

    # Build gene multigraph
    genome = SCerevisiaeGenome(...)
    graph = SCerevisiaeGraph(...)
    graph_names = ["physical", "regulatory"]  # Focus on these for now
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Initialize model - GIN ONLY (no GATv2)
    gene_encoder_config = {
        "encoder_type": "gin",  # MUST be GIN for lazy
        "gin_hidden_dim": cfg.model.hidden_channels,
        "gin_num_layers": 2,
    }

    model = GeneInteractionDango(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        gene_multigraph=gene_multigraph,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=OmegaConf.to_container(cfg.model.local_predictor_config),
    ).to(device)

    print(f"\n[Model initialized]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = LogCoshLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Target labels
    y = batch["gene"].phenotype_values

    print(f"\n[Starting overfitting test - expect near-zero loss]")

    # Overfitting loop
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()

        # Forward pass
        pred, repr_dict = model(cell_graph, batch)

        # Loss
        loss = criterion(pred.squeeze(), y)

        # Backward
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(cell_graph, batch)
                mse = F.mse_loss(val_pred.squeeze(), y)
                mae = F.l1_loss(val_pred.squeeze(), y)

                # Correlation
                pred_np = val_pred.squeeze().cpu().numpy()
                y_np = y.cpu().numpy()
                if np.std(pred_np) > 1e-8 and np.std(y_np) > 1e-8:
                    corr = np.corrcoef(pred_np, y_np)[0, 1]
                else:
                    corr = 0.0

            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | MSE: {mse.item():.6f} | MAE: {mae.item():.6f} | Corr: {corr:.4f}")
            model.train()

    print(f"\n[OK] Overfitting test complete!")
    print(f"Final loss should be near-zero if model is working correctly.")

if __name__ == "__main__":
    main()
```

**Testing:**

```bash
# Run overfitting test
python torchcell/models/hetero_cell_bipartite_dango_gi_lazy.py

# Expected output:
# Epoch 0   | Loss: 0.123456 | ...
# Epoch 50  | Loss: 0.000123 | ...
# Epoch 450 | Loss: 0.000001 | Corr: 0.9999
```

---

### Phase 6: Testing & Validation (Critical - 3-4 hours)

**Test Suite:**

```python
# tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py

def test_masked_gin_creation():
    """Test that create_conv_layer returns MaskedGINConv for GIN"""
    conv = create_conv_layer("gin", 64, 64, {}, dropout=0.1)
    assert isinstance(conv, MaskedGINConv)

def test_edge_mask_passing():
    """Test that edge masks are passed through conv layers"""
    x = torch.randn(6607, 64)
    edge_index = torch.randint(0, 6607, (2, 144211))
    edge_mask = torch.ones(144211, dtype=torch.bool)
    edge_mask[:1000] = False  # Mask first 1000 edges

    conv = create_conv_layer("gin", 64, 64, {}, dropout=0.1)
    wrapper = AttentionConvWrapper(conv, 64, norm="layer", activation="relu", dropout=0.1)

    out = wrapper(x, edge_index, edge_mask=edge_mask)

    assert out.shape == (6607, 64)
    assert not torch.isnan(out).any()

def test_full_graph_forward_single():
    """Test forward_single returns full graph embeddings"""
    from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch

    dataset, batch, _, _ = load_sample_data_batch(batch_size=1, use_custom_collate=True)

    model = GeneInteractionDango(...)
    z = model.forward_single(batch)

    # Should return full graph embeddings
    assert z.shape[0] == batch["gene"].num_nodes  # Full count (6607)
    assert z.shape[1] == model.hidden_channels
    assert not torch.isnan(z).any()

def test_full_forward_pass():
    """Test complete forward pass with lazy data"""
    from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch

    dataset, batch, _, _ = load_sample_data_batch(batch_size=2, use_custom_collate=True)

    model = GeneInteractionDango(...)
    cell_graph = dataset.cell_graph

    pred, repr_dict = model(cell_graph, batch)

    # Verify shapes
    assert pred.shape == (2, 1)  # [batch_size, 1]
    assert not torch.isnan(pred).any()

    # Verify output dict structure
    required_keys = ["z_w", "z_i", "z_p", "local_interaction",
                     "global_interaction", "gate_weights", "gene_interaction"]
    for key in required_keys:
        assert key in repr_dict, f"Missing key: {key}"

def test_overfitting():
    """Test that model can overfit a single batch"""
    from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch

    dataset, batch, _, _ = load_sample_data_batch(batch_size=4, use_custom_collate=True)

    model = GeneInteractionDango(...)
    cell_graph = dataset.cell_graph

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    y = batch["gene"].phenotype_values

    # Train for 200 epochs
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred, _ = model(cell_graph, batch)
        loss = criterion(pred.squeeze(), y)
        loss.backward()
        optimizer.step()

    # Final loss should be very small
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(cell_graph, batch)
        final_loss = criterion(final_pred.squeeze(), y)

    assert final_loss.item() < 0.01, f"Failed to overfit: final loss {final_loss.item()}"
```

**Run tests:**

```bash
pytest tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py -xvs
```

---

## 4. Success Criteria

- [ ] Model creates successfully with GIN encoder
- [ ] `forward_single()` returns full graph embeddings `[total_nodes, hidden_dim]`
- [ ] Edge masks are passed to MaskedGINConv
- [ ] No NaNs during forward pass
- [ ] Output dict matches original structure (lines 1048-1059)
- [ ] Model overfits single batch (loss < 0.01)
- [ ] Works with LazyCollater batching
- [ ] Correlation > 0.99 on overfitting test

---

## 5. Known Limitations

### GATv2 Not Supported (Yet)

- **Issue**: GATv2Conv doesn't have native edge masking like MaskedGINConv
- **Workaround**: Use GIN encoder only (`encoder_type: "gin"`)
- **Future**: Create MaskedGATv2Conv following MaskedGINConv pattern

### Performance Expectations

- **Data loading**: ~3.65x speedup (already verified)
- **Model forward**: Minimal overhead from edge masking (~2-5%)
- **Overall**: Expect ~2-3x total speedup vs SubgraphRepresentation

---

## 6. File Structure

```
torchcell/models/
  hetero_cell_bipartite_dango_gi.py          # Original (SubgraphRepresentation)
  hetero_cell_bipartite_dango_gi_lazy.py     # New (LazySubgraphRepresentation)

tests/torchcell/models/
  test_hetero_cell_bipartite_dango_gi.py     # Original tests
  test_hetero_cell_bipartite_dango_gi_lazy.py # New tests
```

---

## 7. Implementation Checklist

### Phase 1: Infrastructure

- [ ] Copy model file to `*_lazy.py`
- [ ] Import `MaskedGINConv`
- [ ] Update `create_conv_layer()` for GIN
- [ ] Remove GATv2 support (add NotImplementedError)
- [ ] Test import and creation

### Phase 2: Wrappers

- [ ] Add `edge_mask` param to `AttentionConvWrapper.forward`
- [ ] Update `HeteroConvAggregator.forward` to pass edge masks
- [ ] Unit test mask passing

### Phase 3: forward_single

- [ ] Remove node filtering
- [ ] Extract edge masks from `data[edge_type].mask`
- [ ] Pass `edge_mask_dict` to convolutions
- [ ] Test single graph forward pass

### Phase 4: forward()

- [ ] Apply node masks before global aggregation
- [ ] Verify `perturbation_indices` work
- [ ] Test full forward pass

### Phase 5: Main Function

- [ ] Use `load_lazy_batch_006` for data
- [ ] Create overfitting loop
- [ ] Run and verify near-zero loss

### Phase 6: Testing

- [ ] Write unit tests
- [ ] Run overfitting test
- [ ] Verify output structure
- [ ] Benchmark performance

---

## 8. Troubleshooting Guide

### Issue: NaNs in embeddings

**Cause**: Likely division by zero in attention or normalization
**Fix**: Check that masks aren't all False, verify LayerNorm has eps=1e-5

### Issue: Index out of bounds

**Cause**: Mismatch between node indices and embedding size
**Fix**: Verify `num_nodes` includes full graph (6607), not just kept nodes

### Issue: Dimension mismatch in aggregation

**Cause**: Forgot to apply mask before aggregation
**Fix**: Use `z_kept = z[~pert_mask]` before passing to aggregator

### Issue: Loss not decreasing

**Cause**: Check learning rate, weight decay, or model initialization
**Fix**: Start with lr=1e-3, weight_decay=1e-4, verify no frozen layers

---

## 9. Timeline

| Phase | Tasks | Time | Cumulative |
|-------|-------|------|------------|
| 1 | Infrastructure | 0.5h | 0.5h |
| 2 | Wrappers | 1h | 1.5h |
| 3 | forward_single | 2-3h | 3.5-4.5h |
| 4 | forward() | 1-2h | 4.5-6.5h |
| 5 | Main function | 2h | 6.5-8.5h |
| 6 | Testing | 3-4h | 9.5-12.5h |

**Total**: 10-13 hours

---

## 10. Next Steps After Implementation

1. **Integrate with training script**: Update `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi.py` to use lazy version
2. **Create config**: Add `hetero_cell_bipartite_dango_gi_lazy.yaml` configuration
3. **Benchmark**: Compare training speed vs original SubgraphRepresentation
4. **Documentation**: Update model docs with lazy usage instructions
5. **GATv2 support**: Implement MaskedGATv2Conv if needed
6. **Consolidate**: Consider merging lazy/non-lazy versions with a flag

---

## References

- Original model: `/home/michaelvolk/Documents/projects/torchcell/torchcell/models/hetero_cell_bipartite_dango_gi.py`
- MaskedGINConv: `/home/michaelvolk/Documents/projects/torchcell/torchcell/nn/masked_gin_conv.py`
- LazyCollater: `/home/michaelvolk/Documents/projects/torchcell/torchcell/datamodules/lazy_collate.py`
- LazySubgraphRepresentation: `/home/michaelvolk/Documents/projects/torchcell/torchcell/data/graph_processor.py:1211`
- Data loading: `/home/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_lazy_batch_006.py`
