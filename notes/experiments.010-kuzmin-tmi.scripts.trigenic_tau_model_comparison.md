---
id: 22ak4tyr1qjszrjy3e37b0o
title: Trigenic_tau_model_comparison
desc: ''
updated: 1775940196456
created: 1775915387399
---
Source data and provenance for the trigenic τ model-comparison bar chart produced by `experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py`.

![trigenic_tau_model_comparison](assets/images/010-kuzmin-tmi/trigenic_tau_model_comparison_2026-04-13-00-57-40.png)

## 2026.04.11 - Source Data

Ours TorchCell Graph Regularized Transformer

| Val Pearson | Val Spearman |
| :--- | :--- |
| $\mathbf{0.454} \pm \mathbf{0.006}$ | $\mathbf{0.421} \pm \mathbf{0.004}$ |

Dango Repro Best (3 replicates)

```text
0.36759
0.36708
0.36637
```

DCell - use the val pearson shown (3 replicates)

```text
0.17321017384529114
0.1550033837556839
0.14192065596580505
```

GEM (Yeast9) fitness pearson - deterministic modeling, so no SE

```text
Pearson r = 0.0006
```

## 2026.06.04 - Error-Bar Provenance (which whiskers are real)

Caveat recorded after auditing `trigenic_tau_model_comparison.py`. The four error bars are **not all the same statistical quantity**, so be careful when presenting them side by side.

| Model | Mean | Error bar | Underlying data | Error type |
| :--- | :--- | :--- | :--- | :--- |
| Yeast9 | 0.0006 | 0.0 | single deterministic value | legitimately zero (no replicates by nature) |
| DCell | 0.157 | ≈ 0.009 | 3 real replicate Pearson values | **SEM** = `std(ddof=1)/√3` (computed) |
| DANGO | 0.367 | ≈ 0.0004 | 3 real replicate Pearson values | **SEM** = `std(ddof=1)/√3` (computed) |
| TorchCell | 0.454 | 0.006 | reported `0.454 ± 0.006`; **no raw replicates** | **reported SE, hardcoded** |

Key points:

- **None of the error bars are dummy placeholders** — every value is grounded in data. The Yeast9 zero is real (a deterministic FBA model has nothing to average over).
- **DCell and DANGO** are genuine: three replicate Pearson values each, with a properly computed SEM.
- **TorchCell** has no raw replicate array in the source — only the reported `0.454 ± 0.006`. The script *reconstructs* a synthetic 3-element array (`mean+SE, mean, mean−SE`) purely so the bar height renders at 0.454, then **hardcodes** the error to the reported `0.006`. The reconstructed array must not be treated as raw data.
- **Methodological inconsistency to resolve before external use:** DCell/DANGO whiskers are **SEM**, while TorchCell's is a **reported SE** taken at face value. If that reported `± 0.006` is itself a SEM over the TorchCell replicates, the comparison is apples-to-apples; if it is an SD, TorchCell's whisker is ~√3× too large relative to the other two. Confirm what the TorchCell `± 0.006` represents (trace it back to the training run) before publishing.
- The same statistics appear in the SIMB conference abstract ([[conference.simb-2026.abstract]]) — keep the two in sync.
