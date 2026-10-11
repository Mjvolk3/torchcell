---
id: dg0kbs12eem0wcrfj7t0mpx
title: Wandb_campaign_census
desc: ''
updated: 1791703240920
created: 1791703240920
---

## 2026.10.11 - Census of the 019 campaign on W&B

2,518 runs in 33 projects, 1,766 before SIMB (July) and 734 after. One row per project, phase and scored head with the window rule of the round; July runs almost never reached their budget (EarlyStopping on the validation loss, patience 20, was the default), so their scores are end-window and marked with a dagger. Cached pulls under the session scratch dir; `--from-cache` replays without history requests. Table in `notes-tex/figure-3-gate/tables/wandb_campaign_census.tex`.
