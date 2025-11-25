---
id: tuiln1fgbwzdsumobnm0fjy
title: Troubleshoot Gpu Utilization
desc: ''
updated: 1761091742312
created: 1761083796340
---

## 2025.10.21 - GPU Utilization Improvement

I suspect that GPU utilization is low due to subgraphing ops and batching. [2025.10.21 - Experiment 058](#20251021---experiment-058---sum-over-graphs-with-dist-loss-regularization) utilization looks high which I believe is mainly due to larger batch size. I don't think it has anything to do with different GPU.

### 2025.10.21 - Experiment 055 - Sum Over Graphs with Wasserstein Regularization

![](./assets/images/experiments.006.hetero-cell-bipartite-dango-gi.troubleshoot-gpu-utilization.md.006.055.gpu-utilization.png)

Utilization: 38%
GPU: RTX 6000 Ada

### 2025.10.21 - Experiment 056 - Sum Over Graphs with Wasserstein Regularization

![](./assets/images/experiments.006.hetero-cell-bipartite-dango-gi.troubleshoot-gpu-utilization.md.006.056.gpu-utilization.png)

Utilization: 38%
GPU: RTX 6000 Ada

### 2025.10.21 - Experiment 058 - Sum Over Graphs with Dist Loss Regularization

![](./assets/images/experiments.006.hetero-cell-bipartite-dango-gi.troubleshoot-gpu-utilization.md.006.058.gpu-utilization.png)

Utilization: 48%
GPU: A100 80GB
