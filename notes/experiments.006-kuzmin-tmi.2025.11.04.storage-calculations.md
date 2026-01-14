---
id: oi6f3x44ccqpwqjyefxq9op
title: Storage Calculations
desc: ''
updated: 1767845278027
created: 1767845278027
---

## Corrected Storage Calculations for Full Masks

This document corrects an initial 13TB storage estimate for full graph masks down to the actual 820GB requirement, demonstrating that uint8 mask storage is feasible and fits comfortably within available scratch space.

### Initial Error

My original estimate of 13TB was incorrect. I was calculating ~43MB per sample, which was likely for full graph structures, not just masks.

### Correct Calculation

#### Data Dimensions

- **Edges**: ~2.45M total across all graph types
- **Nodes**: ~16.5K (6607 genes + 7122 reactions + 2806 metabolites)
- **Samples**: 332,313

#### Storage as uint8 (1 byte per boolean)

```
Edge masks: 2,450,000 × 332,313 × 1 byte = 814.2 GB
Node masks: 16,535 × 332,313 × 1 byte = 5.5 GB
Total: ~820 GB
```

### Storage Availability

- **Available on /scratch**: 4.9TB
- **Required**: 820GB
- **Verdict**: ✅ Fits comfortably (using only 17% of available space)

### Optimization Details

#### Storage Format

- Masks stored as `torch.uint8` (1 byte per boolean)
- Converted back to `torch.bool` on loading
- No compression overhead

#### Expected Performance

- **Storage**: ~2.5MB per sample (820GB total)
- **Loading**: <0.1ms (direct deserialization)
- **Training**: 0.38+ it/s (matching or exceeding on-the-fly)

### Implementation Files

1. **`preprocess_lazy_dataset_full_masks.py`**: Preprocessing script with uint8 storage
2. **`neo4j_preprocessed_cell_full_masks.py`**: Dataset loader with uint8→bool conversion
3. **`hetero_cell_bipartite_dango_gi_lazy_full_masks.py`**: Training script
4. **`gh_hetero_cell_bipartite_dango_gi_lazy-ddp_074_full_masks.slurm`**: Job submission

### Usage

```bash
# Step 1: Run preprocessing (~50 minutes, creates 820GB LMDB)
cd /home/michaelvolk/Documents/projects/torchcell
python experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset_full_masks.py

# Step 2: Submit training job
sbatch experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_074_full_masks.slurm
```

### Comparison

| Method | Storage | Loading Speed | Training Speed |
|--------|---------|---------------|----------------|
| Compressed indices | 3GB | 15ms (reconstruction) | 0.24 it/s |
| Full masks (uint8) | 820GB | <0.1ms (direct) | 0.38+ it/s |
| On-the-fly | Minimal | 10ms (compute) | 0.38 it/s |

### Conclusion

With the corrected storage calculation (820GB instead of 13TB), the full masks approach is feasible on your system and should provide optimal performance.
