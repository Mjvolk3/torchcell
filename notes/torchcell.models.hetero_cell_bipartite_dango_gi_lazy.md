---
id: 1vosfycny2mob4267evmi4u
title: Hetero_cell_bipartite_dango_gi_lazy
desc: ''
updated: 1762400278956
created: 1762333527498
---

## 2025.11.05 - Simplifying Model

### Plan: Make Local Predictor Optional

**Goal**: Save 16.7% compute time and ~200K parameters by making local predictor optional via config.

**Config Change** (add to `local_predictor_config`):

```yaml
local_predictor_config:
  use_local_predictor: false  # NEW: Set to false to skip local predictor entirely
  num_attention_layers: 2     # Ignored when use_local_predictor=false
  num_heads: 8                # Ignored when use_local_predictor=false
  combination_method: "concat" # Ignored when use_local_predictor=false
```

**Model Changes** (`torchcell/models/hetero_cell_bipartite_dango_gi_lazy.py`):

1. **Line ~870** - Read config flag:

```python
local_predictor_config = local_predictor_config or {}
self.use_local_predictor = local_predictor_config.get("use_local_predictor", True)
self.combination_method = local_predictor_config.get("combination_method", "gating")
```

2. **Line ~951** - Conditional initialization (NEVER CREATE if disabled):

```python
# Gene interaction predictor (optional - only create if enabled)
if self.use_local_predictor:
    self.gene_interaction_predictor = GeneInteractionPredictor(
        hidden_dim=hidden_channels,
        num_heads=local_predictor_config.get("num_heads", 4),
        num_layers=local_predictor_config.get("num_attention_layers", 2),
        dropout=dropout,
        activation=activation,
    )
else:
    self.gene_interaction_predictor = None
```

3. **Line ~973** - Conditional gate MLP:

```python
# MLP for gating (only if using gating AND local predictor enabled)
if self.combination_method == "gating" and self.use_local_predictor:
    self.gate_mlp = nn.Sequential(...)
else:
    self.gate_mlp = None
```

4. **Line ~1225** - Skip local predictor computation:

```python
# Get gene interaction predictions using local predictor (if enabled)
if self.use_local_predictor:
    local_interaction = self.gene_interaction_predictor(pert_gene_embs, batch_assign)
    # NaN checks...
    if torch.isnan(local_interaction).any():
        raise RuntimeError("NaN detected in local interaction predictions")
else:
    local_interaction = None
```

5. **Line ~1267** - Handle global-only case:

```python
# Combine predictions based on configuration
if not self.use_local_predictor:
    # Global only mode - weight is 1.0
    gene_interaction = global_interaction
    gate_weights = torch.ones(batch_size, 1, device=global_interaction.device)

elif self.combination_method == "gating":
    # Existing gating logic...
    pred_stack = torch.cat([global_interaction, local_interaction], dim=1)
    gate_logits = self.gate_mlp(pred_stack)
    gate_weights = F.softmax(gate_logits, dim=1)
    weighted_preds = pred_stack * gate_weights
    gene_interaction = weighted_preds.sum(dim=1, keepdim=True)

elif self.combination_method == "concat":
    # Existing concat logic...
    gene_interaction = 0.5 * global_interaction + 0.5 * local_interaction
    gate_weights = torch.ones(batch_size, 2, device=global_interaction.device) * 0.5
```

6. **Line ~1320** - Conditional return dict:

```python
return_dict = {
    "z_w": z_w_global,
    "z_i": z_i_global,
    "z_p": z_p_global,
    "global_interaction": global_interaction,
    "gene_interaction": gene_interaction,
    "gate_weights": gate_weights,
    "pert_gene_embs": pert_gene_embs,
    "layer_aggregation_weights": self.layer_aggregation_weights,
}
# Only include local_interaction if predictor was used
if self.use_local_predictor:
    return_dict["local_interaction"] = local_interaction

return gene_interaction, return_dict
```

7. **Line ~1337** - Update num_parameters property:

```python
counts = {
    "gene_embedding": count_params(self.gene_embedding),
    "preprocessor": count_params(self.preprocessor),
    "convs": count_params(self.convs),
    "global_aggregator": count_params(self.global_aggregator),
    "global_interaction_predictor": count_params(self.global_interaction_predictor),
}

# Only count if modules exist
if self.use_local_predictor and self.gene_interaction_predictor is not None:
    counts["gene_interaction_predictor"] = count_params(self.gene_interaction_predictor)
if self.gate_mlp is not None:
    counts["gate_mlp"] = count_params(self.gate_mlp)

counts["total"] = sum(counts.values())
return counts
```

**Expected Benefits**:

- Speed: ~17% faster forward pass (saves 11ms from 67ms → ~56ms)
- Parameters: ~200K fewer (1.78M → ~1.58M)
- Simplicity: Test if local predictor is necessary for performance
- Backward compatible: Defaults to `True` (existing behavior preserved)

**Test Configs to Create**:

1. `profile_v2.yaml` - 2 GIN layers, 4 attention heads (with local predictor)
2. `profile_v3.yaml` - Global only (use_local_predictor: false)
