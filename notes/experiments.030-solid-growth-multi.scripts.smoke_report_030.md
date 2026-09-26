---
id: eomd8t2h2z5zdclzxfrgfk8
title: Smoke_report_030
desc: ''
updated: 1790417029711
created: 1790417029711
---

## 2026.09.26 - The three criteria

Reads the token run's and the control run's `SmokeResult` and prints PASS/FAIL per criterion (plan decision 10), with delta the normalized offset (0.3): (1a) mean of pred(synthetic token) minus pred(own token) within 10% of delta, (1b) its sd over rows below delta/5; (2) reading real rows under the synthetic token raises their MSE by 0.5 to 1.5 delta squared; (3) the control's MSE over all rows exceeds the token run's by at least delta squared over 8 (half the delta squared over 4 floor a token-blind predictor pays on the cloned rows). Exit code 0 on PASS, 1 on FAIL; the smoke slurm job exits with it.
