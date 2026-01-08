---
id: cb8p03hr8jzzpk5evu5kejw
title: False Torchmetrics Bug Bc Wrong Dataset Path
desc: ''
updated: 1767753681826
created: 1767753651345
---

## Problem Statement

When evaluating a trained CellGraphTransformer checkpoint using [[equivariant_cell_graph_transformer_eval|experiments.010-kuzmin-tmi.scripts.equivariant_cell_graph_transformer_eval]], we observed ~13% metrics discrepancy between DDP training (4 GPU) and single-GPU eval:

| Metric | DDP Training (4 GPU) | Single-GPU Eval | Delta |
|--------|----------------------|-----------------|-------|
| val Pearson | 0.4619 | 0.5215 | +13% |
| val MSE | 0.00316 | 0.00273 | -14% |

Both additive (MSE) and non-additive (Pearson) metrics showed improvement in single-GPU eval.

## Hypotheses Investigated

We suspected a TorchMetrics DDP synchronization failure:

1. **Metrics created before DDP init**: `MetricCollection` objects created in `__init__` of `RegressionTask` might not detect the distributed environment when `torch.distributed.is_initialized()` is called
2. **Improper cross-rank aggregation**: Each GPU computing local metrics on ~1/4 of data, then incorrectly averaging (`avg(local_r) ≠ global_r` for Pearson)
3. **`sync_on_compute` timing**: Default is `True`, but requires DDP to be initialized

## Fix Attempt: Lightning Hooks

Added `on_fit_start` hook to `torchcell/trainers/int_transformer_cell.py` to configure metrics AFTER DDP initialization:

```python
def on_fit_start(self):
    """Configure TorchMetrics for DDP after distributed init."""
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() > 1:
        for stage in ["train", "val", "test"]:
            for metric in getattr(self, f"{stage}_metrics").values():
                metric.sync_on_compute = True
```

## Test Matrix

| Config | GPU Count | Purpose |
|--------|-----------|---------|
| `equivariant_cell_graph_transformer_cabbi_metrics_1gpu_test_gh_005.yaml` | 1 | Baseline |
| `equivariant_cell_graph_transformer_cabbi_metrics_4gpu_test_gh_004.yaml` | 4 | DDP comparison |

SLURM scripts created for each configuration, with and without the hook fix.

## Resolution: Dataset Path Mismatch

**The actual root cause was NOT a TorchMetrics bug.**

The eval script was loading a **different dataset** than training used. Once the dataset path was corrected:

- Metrics matched within floating-point precision (4th-9th decimal place disagreement)
- No hook was needed - TorchMetrics works correctly out of the box

### Lesson Learned

> Before investigating complex distributed computing issues (DDP sync, TorchMetrics configuration, Lightning hooks), **verify your data paths first**.

---

## Verification Results (Correct Dataset Path)

## Inference No Hook

### Inference No Hook - 1 GPU

#### Inference No Hook - 1 GPU - Slurm Output

│       val/gene_interaction/MSE       │         0.003405288327485323          │

#### Inference No Hook - 1 GPU - Wandb Logging

```
group: gilahyper-747_76f89452ed5d08aed3028c6bf9214923ba1265c998d44d24c1b6f1a70271aa44 
0.0034052832052111626
```

Mismatch occurs at 9th digit.

`0.003405283`

#### Inference No Hook - 1 GPU - Saved Model

```
models/checkpoints/gilahyper-747_76f89452ed5d08aed3028c6bf9214923ba1265c998d44d24c1b6f1a70271aa44/t34kenj1-best-mse-epoch=07-val/gene_interaction/MSE=0.0034.ckpt
```

#### Inference No Hook - 1 GPU - Val Sample

Mathpix OCR screenshot data on images.

Inference

```
Training Results
Loss: PointDistGraphReg
Epochs: 0
Target 0
MSE=3.405e-03, $\mathrm{n}=1000$
Pearson=0.305, Spearman=0.308
```

Training

```
Training Results
Loss: PointDistGraphReg
Epochs: 7
Target 0
MSE=3.405e-03, $\mathrm{n}=1000$
Pearson=0.305, Spearman=0.309
```

These are same which is good, but confusing considering previous we were having issues with no hook. Maybe the issue was related to dataset path. We could have been loading the wrong data.

#### Inference No Hook - 1 GPU - Slurm Output Pearson

```
group: gilahyper-747_76f89452ed5d08aed3028c6bf9214923ba1265c998d44d24c1b6f1a70271aa44 
0.30499735474586487
```

#### Inference No Hook - 1 GPU - Wandb Logging Pearson

│     val/gene_interaction/Pearson     │          0.3049542009830475           │

Disagreement at 5th digit. These match for all practical purposes.

`0.30495`

#### Inference No Hook - 1 GPU - Conclusion

Everything seems to match fine.

### Inference No Hook - 4 GPU

#### Inference No Hook - 4 GPU - Slurm Output

│       val/gene_interaction/MSE       │         0.0033936810214072466         │

#### Inference No Hook - 4 GPU - Wandb Logging

```
group: gilahyper-746_7611b85f469d398c835b378142115a2c0619d2a6ea9fc2dd4f50fc949eae98a6 
0.00339402724057436
```

Mismatch occurs at 6th digit. These match.

`0.003394`

#### Inference No Hook - 4 GPU - Saved Model

```
models/checkpoints/gilahyper-746_7611b85f469d398c835b378142115a2c0619d2a6ea9fc2dd4f50fc949eae98a6/c0i006av-best-mse-epoch=18-val/gene_interaction/MSE=0.0034.ckpt
```

#### Inference No Hook - 4 GPU - Val Sample

Difficult to tell since the metrics are split across 4 different gpus. Seems plausible that they could be aligned.

#### Inference No Hook - 4 GPU - Wandb Logging Pearson

```
group: gilahyper-746_7611b85f469d398c835b378142115a2c0619d2a6ea9fc2dd4f50fc949eae98a6 
0.32310670614242554
```

#### Inference No Hook - 4 GPU - Slurm Output Pearson

│     val/gene_interaction/Pearson     │          0.32318881154060364          │

Mismatch occurs at 5th digit. These match fine.

`0.32318`

#### Inference No Hook - 4 GPU - Conclusion

Everything seems to match fine. The observed issues were related to dataset path mismatch.
