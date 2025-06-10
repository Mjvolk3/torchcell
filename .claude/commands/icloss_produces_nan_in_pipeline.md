# ICLoss NaN Issue in Training Pipeline

## Summary

The GeneInteractionDango model experiences NaN values during training when using ICLoss in the Lightning training pipeline, but works fine with LogCosh loss and also works fine with ICLoss when run standalone (outside the Lightning pipeline).

## Evidence of the Issue

### 1. Initial Error

- Model produces NaN in wildtype embeddings (z_w) immediately on first batch
- Error: `RuntimeError: NaN detected in wildtype embeddings (z_w)`
- All 6607 gene embeddings become NaN (shape: [6607, 64])
- Occurs even with reduced learning rate (1e-6 � 1e-4)

### 2. Configuration Attempts

- Reduced graphs from 9 to 2 (physical, regulatory only)
- Set `lambda_supcr: 0` to disable SupCR component
- Issue persists even with these changes

### 3. Key Observation

- **Works with LogCosh loss**: Model trains successfully
- **Works with ICLoss in standalone**: When running the model's main() function directly, ICLoss works fine
- **Fails with ICLoss in Lightning pipeline**: NaN occurs immediately in first forward pass

### 4. Standalone Success Output

```
Using ICLoss with lambda_dist=0.1, lambda_supcr=0
Starting training:
  ICLoss components: mse=0.1364, dist=0.0148, supcr=0.0000
Epoch 1/500
ICLOSS Loss: 0.151205
Correlation: -0.4731
```

## Root Cause Analysis

The issue is related to device handling and initialization order in the Lightning pipeline:

1. **Model initialization**: Embeddings initialized on CPU with std=0.1 and 0.01 offset
2. **Lightning setup**: Model moved to device, but embedding offset computation may not be tracked
3. **First forward pass**: Device mismatch or untracked computation causes NaN
4. **ICLoss specific**: Creates more complex computational graph that exposes the issue

## Key Files Involved

### 1. Model File

- **Path**: `/torchcell/models/hetero_cell_bipartite_dango_gi.py`
- **Key sections**:
  - Lines 345-353: Embedding initialization with offset
  - Lines 475-511: forward_single method
  - Lines 512-687: Main forward method with NaN checks

### 2. Training Script

- **Path**: `/experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi.py`
- **Purpose**: Main training script using Lightning

### 3. Configuration

- **Path**: `/experiments/005-kuzmin2018-tmi/conf/hetero_cell_bipartite_dango_gi.yaml`
- **Key settings**:
  - `loss: icloss`
  - `lambda_dist: 0.1`
  - `lambda_supcr: 0`
  - `lr: 1e-4`

### 4. Trainer

- **Path**: `/torchcell/trainers/int_hetero_cell.py`
- **Key sections**:
  - Lines 71-81: forward method with device handling
  - Lines 91-141: _shared_step with ICLoss handling

### 5. Loss Functions

- **Path**: `/torchcell/losses/isomorphic_cell_loss.py`
- **Contains**: ICLoss implementation
- **Path**: `/torchcell/losses/multi_dim_nan_tolerant.py`
- **Contains**: SupCR, WeightedSupCRCell, WeightedDistLoss

## Fixes Applied

### 1. Numerical Stability in SupCR

- Modified `compute_similarity` in `multi_dim_nan_tolerant.py`:

  ```python
  # Add small epsilon to prevent division by zero
  norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=self.eps)
  normed = embeddings / norms
  ```

- Increased eps from 1e-7 to 1e-5 in WeightedSupCRCell

### 2. Embedding Initialization

- Changed std from 0.02 to 0.1
- Added 0.01 offset to prevent zero embeddings

## Recommended Solutions

### 1. Remove Embedding Offset from Init

The most likely fix is removing the offset addition from initialization:

```python
# Remove this line from __init__:
# self.gene_embedding.weight.add_(0.01)

# Add offset in forward if needed:
# x_gene = self.gene_embedding(gene_idx) + 0.01
```

### 2. Ensure Device Consistency

Add device synchronization in the model:

```python
def _ensure_device_consistency(self):
    device = next(self.parameters()).device
    if hasattr(self, 'gene_embedding'):
        self.gene_embedding = self.gene_embedding.to(device)
```

### 3. Fix Lightning Strategy

Change in config:

```yaml
trainer:
  strategy: single_device  # Instead of 'auto'
```

### 4. Alternative: Warmup Strategy

Start with very small or zero lambda values:

```yaml
lambda_dist: 0.01
lambda_supcr: 0.0
```

## Conclusion

The issue is specific to how Lightning handles model initialization and device movement when using ICLoss. The complex computational graph created by ICLoss (especially the z_p embeddings) exposes a device synchronization or initialization tracking issue that doesn't occur with simpler losses like LogCosh.
