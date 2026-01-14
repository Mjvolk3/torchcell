---
id: aqvwx9b7hrzu3dv0f1orvlb
title: Point_dist_graph_reg
desc: ''
updated: 1768423367955
created: 1767976523838
---

## Mathematical Formulation

The `PointDistGraphReg` loss is a modular composite loss combining three components:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{point}} \cdot \mathcal{L}_{\text{point}} + \lambda_{\text{dist}} \cdot \mathcal{L}_{\text{dist}} + \lambda_{\text{graph}} \cdot \mathcal{L}_{\text{graph}}
$$

### 1. Point Estimator Loss (Required)

The point estimator minimizes prediction error using either:

**LogCosh Loss** (default):
$$
\mathcal{L}_{\text{point}} = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(\hat{y}_i - y_i))
$$

**MSE Loss**:
$$
\mathcal{L}_{\text{point}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

where:

- $\hat{y}_i$ = predicted value for sample $i$
- $y_i$ = target value for sample $i$
- $N$ = batch size

### 2. Distribution Loss (Optional)

Encourages the distribution of predictions to match the distribution of targets:

**Dist Loss** (kernel density-based):
$$
\mathcal{L}_{\text{dist}} = D_{\text{KDE}}(p(\hat{y}) \parallel p(y))
$$

**Wasserstein Loss** (optimal transport):
$$
\mathcal{L}_{\text{dist}} = W_p(p(\hat{y}), p(y))
$$

where:

- $p(\hat{y})$ = empirical distribution of predictions
- $p(y)$ = empirical distribution of targets
- $D_{\text{KDE}}$ = kernel density estimation-based divergence
- $W_p$ = $p$-Wasserstein distance

**Buffered Variant**: Both distribution losses support buffering, where the loss is computed against a moving buffer of historical samples:

$$
\mathcal{L}_{\text{dist}} = \alpha \cdot D(p(\hat{y}_{\text{batch}}) \parallel p(y_{\text{batch}})) + (1-\alpha) \cdot D(p(\hat{y}_{\text{buffer}}) \parallel p(y_{\text{buffer}}))
$$

where $\alpha$ controls the balance between current batch and historical buffer.

### 3. Graph Regularization Loss (Optional)

Regularizes model attention weights using graph structure:

$$
\mathcal{L}_{\text{graph}} = \mathcal{R}_{\text{graph}}(\mathbf{A}, \mathcal{G})
$$

where:

- $\mathbf{A}$ = attention weights from model
- $\mathcal{G}$ = biological graph structure (e.g., PPI, regulatory networks)
- $\mathcal{R}_{\text{graph}}$ = graph-based regularization term (defined in model)

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\lambda_{\text{point}}$ | 1.0 | Weight for point estimator loss |
| $\lambda_{\text{dist}}$ | 0.1 | Weight for distribution loss |
| $\lambda_{\text{graph}}$ | 1.0 | Weight for graph regularization |
| `buffer_size` | 256 | Size of historical sample buffer |
| `min_samples_for_dist` | 64 | Minimum samples required for Dist loss |
| `min_samples_for_wasserstein` | 224 | Minimum samples required for Wasserstein loss |

### Normalized Contributions

The loss also tracks normalized contributions to analyze component importance:

$$
\text{norm}_{\text{weighted}, k} = \frac{\lambda_k \cdot \mathcal{L}_k}{\mathcal{L}_{\text{total}}}
$$

$$
\text{norm}_{\text{unweighted}, k} = \frac{\mathcal{L}_k}{\sum_{j} \mathcal{L}_j}
$$

where $k \in \{\text{point}, \text{dist}, \text{graph}\}$

### DDP Synchronization

When using Distributed Data Parallel (DDP), samples are gathered across GPUs for distribution loss computation:

$$
\mathcal{L}_{\text{dist}} = D\left(p\left(\bigcup_{r=0}^{R-1} \hat{y}^{(r)}\right) \parallel p\left(\bigcup_{r=0}^{R-1} y^{(r)}\right)\right)
$$

where $R$ is the number of GPU ranks, ensuring all ranks maintain identical buffer contents.

## Key Question to Answer - Does Graph Reg Help

The `PointDistGraphReg` loss is a modular composite loss combining three components:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{point}} \cdot \mathcal{L}_{\text{point}} + \lambda_{\text{dist}} \cdot \mathcal{L}_{\text{dist}} + \lambda_{\text{graph}} \cdot \mathcal{L}_{\text{graph}}
$$

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{point}} \cdot \mathcal{L}_{\text{point}} + \lambda_{\text{dist}} \cdot \mathcal{L}_{\text{dist}} + \cancel{\lambda_{\text{graph}} \cdot \mathcal{L}_{\text{graph}}}
$$

We were able to show under a certain model config, $\lambda_{graph}=0$ prevents the model from learning.
