---
id: c6mxcgxw2g1s43yboegupsl
title: Alternative Regression Objectives
desc: ''
updated: 1736363970348
created: 1736361792094
---
## Summaries of Losses

### Quantile Loss (Multiple Quantiles):

Description:
This loss computes a separate quantile loss for each quantile in a specified list (e.g., $[0.1,0.2$, $\ldots, 0.9]$ ) and sums them. It is robust to NaN values in the targets, excluding them from the computation.

Equation:
Given a set of quantiles $\mathbf{q}=\left[q_1, q_2, \ldots, q_M\right]$, the total quantile loss is:

$$
L_{\text {Quantile }}=\sum_{m=1}^M \frac{1}{N_m} \sum_{i=1}^{N_m}\left[q_m \cdot \max \left(y_i-\hat{y}_i, 0\right)+\left(1-q_m\right) \cdot \max \left(\hat{y}_i-y_i, 0\right)\right]
$$

Where:

- $M$ is the number of quantiles in $\mathbf{q}$.
- $\quad N_m$ is the number of valid (non-NaN) target-prediction pairs for the $m$-th quantile.
- $q_m$ is the $m$-th quantile in $\mathbf{q}$.

### Dist Loss

$$
L_{\mathrm{Dist}}=\operatorname{MSE}(\tilde{y}, \hat{y})
$$

Where:

- $\tilde{y}$ are the pseudo-labels (KDE-sampled).
- $\hat{y}$ are the sorted predictions.

$$
L_{\text {Total }}=L_{\mathrm{MSE}}+\lambda \cdot L_{\text {Dist }}
$$

## Alternative Regression Loss:

### Multiple Quantile Loss

Given a set of quantiles $\mathbf{q} = [q_1, q_2, \ldots, q_M]$, the total quantile loss is:

$$
L_{\text{Quantile}} = \sum_{m=1}^M \frac{1}{N} \sum_{i=1}^N \left[q_m \cdot \max(y_i - \hat{y}_i, 0) + (1 - q_m) \cdot \max(\hat{y}_i - y_i, 0)\right]
$$

Where:
- $M$ is the number of quantiles in $\mathbf{q}$.
- $N$ is the total number of samples.
- $q_m$ is the $m$-th quantile in $\mathbf{q}$.

***

### Dist Loss

$$
L_{\mathrm{Dist}}=\operatorname{MSE}(\tilde{y}, \hat{y})
$$

Where:

- $\tilde{y}$ are the pseudo-labels (KDE-sampled).
- $\hat{y}$ are the sorted predictions.

$$
L_{\text {Total }}=L_{\mathrm{MSE}}+\lambda \cdot L_{\text {Dist }}
$$

***