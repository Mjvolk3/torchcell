---
id: c5l2jfgr5l8uopyde7bnmcg
title: Soft_label_underflow_audit
desc: ''
updated: 1790876495856
created: 1790876495856
---

## 2026.10.01 - Output on 001-small-build

Audit for the soft-label underflow fix of #529 (PR #584); see [[torchcell.transforms.regression_to_classification]].

```text
minmax=False fitness: labeled 1340841, all-zero before 0 (0.00%), duplicate edges 0, min width 0.0066
minmax=False gene_interaction: labeled 1023196, all-zero before 9422 (0.92%), duplicate edges 0, min width 0.002106
  bins [0, 31] counts [7034, 2388]; widths of bins 0 and 31: 0.9959, 0.6362; new row sums in [0.9999999, 1.0000001], smallest peak 0.5143
minmax=True fitness: labeled 1340841, all-zero before 0 (0.00%), duplicate edges 0, min width 0.004209
minmax=True gene_interaction: labeled 1023196, all-zero before 9422 (0.92%), duplicate edges 0, min width 0.001184
  bins [0, 31] counts [7034, 2388]; widths of bins 0 and 31: 0.56, 0.3577; new row sums in [0.9999999, 1.0000000], smallest peak 0.5143
CE: zero row counted valid: True; its CE -0.0
CE: task-1 mean 1.1815991 = row-0 CE / 2 = 1.1815991 (the zero row dilutes the mean)
CE: gradient on the zero row's logits [0.0, 0.0, 0.0, 0.0]
entropy reg: a zero row enters the KL distances as [0.25, 0.25, 0.25, 0.25] (uniform)
entropy reg: total finite True; the zero row adds no tightness term (read from the class_prob > 0 guard, not measured)
```
