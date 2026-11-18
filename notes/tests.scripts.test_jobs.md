---
id: it6vbxt9b3mtnx5ok8dm8kd
title: Test_jobs
desc: ''
updated: 1794961624026
created: 1794961624026
---

## 2026.11.17 - Tests for the jobs panel

Covers [[scripts.jobs]] and [[scripts.cron_run]]:

- `scripts/crontab.txt` against `CRON_JOBS` in both directions (the single-source check).
- `cron_period` on every supported shape, and a refusal for unsupported ones.
- The real wrapper into `tmp_path`: exit codes passed through, the history cap, the
  log markers, and the panel's readers on the records it wrote.
- Verdicts at the 1.5-period edge, and the recurring, slurm, docker, Actions, PR and
  issue sections rendered from fixture JSON at a fixed time with the zone pinned to
  UTC, asserted line for line.
- A `gh` timeout rendering as one "unreachable" line.

Run: `~/miniconda3/envs/torchcell/bin/python -m pytest tests/scripts/test_jobs.py -q`.
