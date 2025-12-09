---
id: eum97l77licyhifqb4x3oth
title: MMLI 018 Config Justification
desc: >-
  Justification for MMLI 018 configuration as complementary experiment to CABBI
  017
updated: 1753238150996
created: 1753231391117
---

## MMLI 018 Configuration Justification

### Experiment Design Philosophy

MMLI 018 is designed to **replicate CABBI 017's success** while addressing computational slowdown observed with large buffer sizes. Key changes:

1. Matches CABBI 017 parameters exactly (bandwidth=2.0, lambda_dist=0.5, etc.)
2. Only two modifications: reduced buffer size and adjusted batch configuration for MMLI node
3. Tests if smaller buffer (2048) maintains performance while improving training speed

### Key Differences from CABBI 017 → MMLI 018

#### 1. Buffer Size: 4096 → 2048 (ONLY MAJOR CHANGE)

**Justification**:

- Addresses computational slowdown observed in CABBI 017
- Reduces KDE computation cost by 50%
- Still 5.3x larger than original 384 - maintains good distribution coverage
- Tests if performance holds with more efficient buffer size

#### 2. Node Configuration (MMLI constraints)

- `devices`: 6 → 4 (MMLI node limitation)
- `batch_size`: 4 → 8 (adjusted for MMLI memory)
- `grad_accumulation`: 8 → 4 (maintains reasonable effective batch size of 128)

**All other parameters match CABBI 017 exactly:**

- `dist_bandwidth`: 2.0 (wide kernel for standardized data)
- `lambda_dist`: 0.5 (strong distribution matching)
- `lambda_supcr`: 0.01 (meaningful contrastive signal)
- Temperature schedule: 0.2 → 0.05 with cosine
- Learning rate: 5e-4 peak with 3 cycles
- Weight decay: 1e-12 (minimal)
- Adaptive weighting: 10 epoch warmup, stable at 40

### Expected Outcomes

1. **Training Speed**: 2x faster loss computation with half buffer size
2. **Performance**: Should match CABBI 017 closely (within 2-3%)
3. **Tail Capture**: Still expect good tail distribution matching with bandwidth=2.0
4. **Validation**: If successful, shows 2048 buffer is sufficient for this dataset

### Effective Batch Size Calculation

- Per GPU: 8 samples
- Gradient accumulation: 4 steps
- DDP gather: 4 GPUs
- **Total**: 8 × 4 × 4 = 128 samples/update

### Buffer Dynamics

- Buffer size: 2048
- Gathered samples per forward: 32 (8 × 4 GPUs)
- Buffer fills in: 2048 ÷ 32 = 64 forward passes
- Complete refresh every: 64 ÷ 4 = 16 gradient updates
- 2x faster KDE computation vs 4096 buffer

### Hypothesis

This minimal-change experiment tests whether:

1. The computational slowdown in CABBI 017 is primarily due to large buffer KDE costs
2. A 2048 buffer provides sufficient distribution coverage for tail capture
3. All other CABBI 017 improvements (bandwidth, lambda weights, etc.) remain effective
