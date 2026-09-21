---
id: 30lz913cce4a5voq51n1rx3
title: Mmli_campaign_plan
desc: ''
updated: 1790008342856
created: 1790008342856
---

## 2026.09.21 - Campaign table generator

Writes `notes-tex/025-mmli-campaign/tables/t1-campaign.tex` and `t2-queue.tex`, plus `results/mmli_campaign_plan.{csv,json}` and `results/mmli_queue_snapshot.json`. Context in [[experiments.025-solid-growth.mmli-campaign]].

W&B read hazard found while writing it: on the 104-epoch S3 seed 1 (run yb4gjh51, 12,083 steps) `scan_history(keys=["epoch", key])` returns 104 rows of `{"_step": 0, "epoch": None}` with no metric, and the unkeyed scan returns 188 rows covering 3 epochs. `run.history(keys=..., samples=10_000)` returns the real rows. The reader uses `history` and raises when the curve does not reach the summary epoch. `s3_closure_readout.py` had the same defect and reported that run as 3 epochs with no score; it is fixed the same way. `disjoint_embedding_readout.py` still uses the unkeyed scan, which has been complete on its 30-epoch runs.
