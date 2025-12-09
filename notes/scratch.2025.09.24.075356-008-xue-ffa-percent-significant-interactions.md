---
id: wsvlerb7ohtsjmngbxrr54v
title: 075356 008 Xue Ffa Percent Significant Interactions
desc: ''
updated: 1760708432188
created: 1758718438261
---
| Model          | Formula                                                              | P-value Method     | Significant |
|----------------|----------------------------------------------------------------------|--------------------|-------------|
| Multiplicative | $f_{ij} - f_i f_j$                                                   | t-test (raw scale) | 34%         |
| Additive       | $f_{ij} - (f_i + f_j - 1)$                                           | t-test (raw scale) | 37%         |
| Model A        | $s_{g,r} = \log(y_{g,r} + \delta) - \log(y_{ref,r} + \delta)$        | t-test (log scale) | 66%         |
| Model B        | $\log(E[y]) = \beta_0 + \sum \beta_i x_i + \sum \gamma_{ij} x_i x_j$ | Wald test          | 69%         |
