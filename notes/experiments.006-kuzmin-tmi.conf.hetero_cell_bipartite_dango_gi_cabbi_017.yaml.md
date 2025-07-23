---
id: eg242o1grrw41qvy7y50eyf
title: CABBI 017 Config Justification
desc: 'Justification for configuration changes from CABBI 016 to 017'
updated: 1753223931817
created: 1753223699774
---

## CABBI 017 Configuration Changes Justification

### Problem Statement

Previous runs (CABBI 016, MMLI 015) showed:

- MSE-only converges faster and captures more variance
- DistLoss not effectively capturing distribution tails
- No overfitting (train ≈ val curves overlap completely)
- Model predictions concentrated around mean, missing high/low gene interactions

### Key Changes from CABBI 016 → 017

#### 1. Buffer Size: 384 → 4096

**Justification**:

- Previous buffer (384) only captures 0.1% of 365K training samples
- Poor KDE estimation of true distribution, especially tails
- 4096 samples (~1.1% of data) provides much better distribution coverage
- Fills in ~170 forward passes vs 16, acceptable staleness for better coverage

#### 2. Distribution Bandwidth: 0.1 → 2.0

**Justification**:

- Data is standardized (mean=0, std=1)
- Bandwidth=0.1 means kernels only cover ±0.3σ - way too narrow
- Bandwidth=2.0 covers ±6σ range, capturing full distribution including tails
- Critical for DistLoss to work on standardized data

#### 3. Lambda Weights

- `lambda_dist`: 0.1 → 0.5 (5x increase)
- `lambda_supcr`: 0.001 → 0.01 (10x increase)

**Justification**:

- Previous weights let MSE dominate, preventing distribution matching
- Higher dist weight forces model to match full distribution shape
- Higher supcr weight creates meaningful contrastive signal for embeddings

#### 4. Temperature Schedule

- `init_temperature`: 1.0 → 0.2
- `final_temperature`: 0.1 → 0.05

**Justification**:

- Lower temperature = tighter clusters in embedding space
- Start low (0.2) for immediate contrastive learning
- End very low (0.05) for tight clusters that discriminate tail samples

#### 5. Learning Rate Schedule

- `max_lr`: 1e-4 → 5e-4 (5x increase)
- `min_lr`: 1e-8 → 1e-7
- `warmup_steps`: 1 → 5

**Justification**:

- No overfitting means significant headroom to push harder
- Higher LR enables faster convergence
- Quick warmup (5 steps) to reach peak LR faster

#### 6. Weight Decay: 1e-9 → 1e-12

**Justification**:

- Train/val curves identical = no overfitting
- Minimal regularization allows model more capacity
- Nearly removes L2 penalty to maximize expressiveness

#### 7. Adaptive Weighting Schedule

- `warmup_epochs`: 30 → 10
- `stable_epoch`: 30 → 40 (was reaching 0.9 at epoch 30)
- `max_epochs` in loss_config: 30 → 150 (match trainer)

**Justification**:

- Previous config reached buffer_weight=0.9 at epoch 30 and stayed there
- New schedule: quick warmup, reach 0.9 by epoch 40
- Properly aligned with 150 epoch training duration

#### 8. Other Optimizations

- `min_samples_for_dist/supcr`: 64 → 128 (more stable estimates)
- `gamma` (LR decay): 0.8 → 0.85 (less aggressive decay)
- `temp_schedule`: "exponential" → "cosine" (smoother transition)

### Expected Outcomes

1. **Better tail capture**: Wider bandwidth + larger buffer + higher lambda_dist
2. **Faster convergence**: Higher LR + aggressive warmup
3. **Stronger embeddings**: Lower temperature + higher lambda_supcr
4. **Full model capacity**: Minimal weight decay + no overfitting headroom

### Effective Batch Size Calculation

- Per GPU: 4 samples
- Gradient accumulation: 8 steps  
- DDP gather: 6 GPUs
- **Total**: 4 × 8 × 6 = 192 samples/update

### Buffer Dynamics

- Buffer size: 4096
- Gathered samples per forward: 24 (4 × 6 GPUs)
- Buffer fills in: 4096 ÷ 24 = 171 forward passes
- Complete refresh every: 171 ÷ 8 = 21 gradient updates
- Acceptable staleness for distribution estimation
