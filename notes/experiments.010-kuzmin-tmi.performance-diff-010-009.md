---
id: p0uasulfwb48lqo653kzlw0
title: Performance Diff 010 009
desc: ''
updated: 1768413941659
created: 1768413926762
---
## Overview

- Models on wandb tagged with `inf_1`
- Data taken directly from wandb scatter plots

## 010 Models

```
\begin{aligned}
&\text { MSE=3.163e-03, n=37673 }\\
&\text { Pearson=0.462, Spearman=0.425 }
\end{aligned}
```

```
\begin{gathered}
\mathrm{MSE}=3.244 \mathrm{e}-03, \mathrm{n}=37673 \\
\text { Pearson }=0.452, \text { Spearman }=0.422
\end{gathered}
```

```
\begin{aligned}
\mathrm{MSE} & =3.259 \mathrm{e}-03, \mathrm{n}=37673 \\
\text { Pearson } & =0.447, \text { Spearman }=0.416
\end{aligned}
```

## 009 Models

```
\begin{aligned}
\text { MSE } & =3.192 \mathrm{e}-03, \mathrm{n}=29915 \\
\text { Pearson } & =0.399, \text { Spearman }=0.310
\end{aligned}
```

```
\begin{aligned}
\text { MSE } & =3.192 \mathrm{e}-03, \mathrm{n}=29915 \\
\text { Pearson } & =0.399, \text { Spearman }=0.310
\end{aligned}
```

```
\begin{aligned}
\text { MSE } & =3.238 \mathrm{e}-03, \mathrm{n}=29915 \\
\text { Pearson } & =0.389, \text { Spearman }=0.306
\end{aligned}
```

## Analysis Table

| Dataset | Strain Description | Total (N) | Train (N) | Val (N) | Val  MSE ($\times 10^{-3}$) | Val Pearson                | Val Spearman               |
|---------|--------------------|-----------|-----------|---------|-----------------------------|----------------------------|----------------------------|
| 010     | All                | 376,732   | 301,386   | 37,673  | $3.222 \pm 0.042$           | $\mathbf{0.454 \pm 0.006}$ | $\mathbf{0.421 \pm 0.004}$ |
| 009     | Deletion only      | 299,146   | 239,317   | 29,915  | $3.207 \pm 0.022$           | $0.396 \pm 0.005$          | $0.309 \pm 0.002$          |

**Key observations:**

- MSE is nearly identical between datasets
- Pearson correlation is $\sim$15% higher for 010 ($0.454$ vs $0.396$)
- Spearman correlation is $\sim$36% higher for 010 ($0.421$ vs $0.309$)
- The larger Pearson-Spearman gap in 009 ($0.087$) vs 010 ($0.033$) suggests rank-order differences matter more in the deletions-only dataset
