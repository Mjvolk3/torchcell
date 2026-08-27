---
id: naqxv1mywlrnvsmnza29mms
title: Pull_round_leaderboards
desc: ''
updated: 1787795229282
created: 1787795229282
---

## 2026.08.26 - pull_round_leaderboards

One leaderboard across every phenotype strand. Pulls per-run validation history from the eight W&B projects, scores each with roll_max (max of a centered five-point rolling mean, an upward-biased order statistic whose bias grows with epochs run), and writes results/round_leaderboards.csv plus a summary JSON. ONE HTTP REQUEST PER METRIC KEY, deliberately: run.history(keys=[a,b]) inner-joins, so asking for a primary metric together with an auxiliary head's metric returns an EMPTY frame for every control run, which silently drops one side of every paired comparison.

Run from repo root:

```bash
PYTHONPATH=. python experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py
```

Context: [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
