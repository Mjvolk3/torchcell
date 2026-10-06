---
id: bavci3fd7z31h6ntp8fdo9b
title: V22_tables
desc: ''
updated: 1791291803737
created: 1791291803737
---

## 2026.10.06 - Tables of the overnight section of the Figure 3 gate

Typesets `results/v22_readout.json` and `results/prediction_shrinkage_probe.json` into `notes-tex/figure-3-gate/tables/` (`v22_arms.tex`, `v22_runs.tex`, `shrinkage.tex`), input by `sections/6-overnight.tex`. Collapse rule: the prediction spread ratio never reached 0.05 in 300 epochs (never launched), or reached it and then stayed below 0.01 for 50 consecutive epochs. A first rule (any epoch below 0.01 after epoch 200) flagged healthy runs on single-epoch dips and was replaced. Best value per score column in bold; an arrow marks an arm whose complete runs are all above or all below F_ref on shared split seeds. Run `v22_readout.py` first.
