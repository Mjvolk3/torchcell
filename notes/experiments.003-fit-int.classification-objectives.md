---
id: 618l4iz33lp9xt1qq4exbww
title: Classification Objectives
desc: ''
updated: 1736362837645
created: 1731867245942
---
Facilitated with ChatGPT while referencing papers.

- Standard Regression (MSE):

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

- Hard Classification (Cross Entropy):

$$
L_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik}\log(\hat{y}_{ik})
$$

$y_{ik}$ is 1 if sample i belongs to class k, 0 otherwise

- Soft Labels (Smoothed Cross Entropy):

$$
L_{soft} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K s_{ik}\log(\hat{y}_{ik})
$$

$s_{ik}$ is the soft label distribution (e.g., Gaussian centered at true class)

- Ordinal Entropy Regularization (from the paper):

$$
L_{total} = L_{MSE} + \lambda_d L_d + \lambda_t L_t
$$

$$
L_d = -\frac{1}{M(M-1)}\sum_{i=1}^M\sum_{i\neq j} w_{ij}||z_i^c - z_j^c||^2
$$

$$
L_t = \frac{1}{N}\sum_{i=1}^N ||z_i - z_i^c||^2
$$

$$
w_{ij} = ||y_i - y_j||^2
$$

Key terms:

- $y_i$: true value
- $ŷ_i$: predicted value
- $z_i$: feature representation of sample $i$
- $z_i^c$: center of features for samples with similar values to $i$
- $w_{ij}$: weight based on distance between true values
- $λ_d, λ_t$: hyperparameters for diversity and tightness terms
- $K$: number of classes
- $N$: batch size
- $M$: number of samples used for entropy estimation

The ordinal entropy terms:

- $L_d$ (diversity): Pushes feature representations apart while respecting ordinal relationships through $w_{ij}$
- $L_t$ (tightness): Pulls features closer to their centers
- $w_{ij}$: Larger weights for pairs with larger differences in true values

The paper argues combining MSE with these regularization terms gives better results than either pure regression or classification because it:

1. Maintains continuous/ordinal nature of regression
2. Encourages higher entropy representations like classification
3. Explicitly preserves ordinal relationships in feature space

Let me break down each loss term in detail:

Main Regression Loss (MSE):

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

- $N$: number of graphs in batch
- $y_i$: true regression value for graph i
- $ŷ_i$: predicted regression value for graph i

Diversity Loss:

$$
L_d = -\frac{1}{M(M-1)}\sum_{i=1}^M\sum_{i\neq j} w_{ij}||z_i^c - z_j^c||^2
$$

$$
w_{ij} = ||y_i - y_j||^2
$$

- $M$: number of samples used (could be batch size or queue size)
- $z_i^c$: feature center for graph $i$ (or the graph's feature itself if precise labels)
- $w_{ij}$: weight based on difference in regression values
- $||z_i^c - z_j^c||^2$: squared Euclidean distance between feature representations
- Negative sign: because we want to maximize distances, weighted by label differences

Tightness Loss:

$$
L_t = \frac{1}{N}\sum_{i=1}^N ||z_i - z_i^c||^2
$$

- $z_i$: pooled representation of graph $i$
- $z_i^c$: center of features for graphs with similar regression values
- $||z_i - z_i^c||^2$: squared distance to center
- Pulls similar graphs' representations together

Total Loss:

$$
L_{total} = L_{MSE} + \lambda_d L_d + \lambda_t L_t
$$

- $λ_d$: weight for diversity term
- $λ_t$: weight for tightness term
- These balance the competing objectives

In your graph context:

- Each $z_i$ is a pooled graph representation
- $y_i$ is your regression target per graph
- The loss encourages pooled graphs to be:
  - Close together if regression values are close
  - Far apart if regression values are different
  - Distances proportional to differences in regression values

Ordinal Cross Entropy Loss

$$
L_{OCE} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^{K-1} [y_{ik}\log(\hat{p}_{ik}) + (1-y_{ik})\log(1-\hat{p}_{ik})]
$$

- $y_{ik} = 1$ if $y_i > k, 0$ otherwise (ordinal encoding)
- $p̂_{ik}$ = probability that $y_i > k$
- $K-1$ thresholds for $K$ classes

***

### Standard Regression (MSE):

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

## Regression to classification:

### Standard Classification (Cross Entropy):

$$
L_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik}\log(\hat{y}_{ik})
$$

- Bad Idea: Doesn't capture ordinality

## Regression to classification:

### Soft Label Classification:

$$
L_{soft} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K s_{ik}\log(\hat{y}{ik})
$$

- Where $s_{ik}$ is soft assignment (e.g., Gaussian centered at true bin)
- Preserves ordinal relationship through soft bin assignments

***

### Ordinal Classification (Binary Cross Entropy per threshold):

$$
L_{OCE} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^{K-1} [y_{ik}\log(\hat{p}_{ik}) + (1-y_{ik})\log(1-\hat{p}_{ik})]
$$

- Where $y_{ik} = 1$ if $y_i > k$, and $\hat{p}_{ik} = \sigma(f(x_i) - b_k)$ for monotonicity
- Preserves ordinal relationship through architecture

***

### Regression with Ordinal Entropy Regularization:

$$L_{total} = L_{MSE} + \lambda_d L_d + \lambda_t L_t$$

Where:

$$
L_d = -\frac{1}{M(M-1)}\sum_{i=1}^M\sum_{i\neq j} ||y_i - y_j||^2 \cdot ||z_i^c - z_j^c||^2
$$

- push apart if different

$$
L_t = \frac{1}{N}\sum_{i=1}^N ||z_i - z_i^c||^2
$$

- Pull together if similar

- Preserves ordinal relationship through regularization. Contrastive regression loss.

***

## Bin Sizing

- Can bin with equal proportion or equal width.
- std of dmf: 0.0424
- 1.6 (max fitness) / 0.0424 $\approx$ 38 bins (minimal binning)
