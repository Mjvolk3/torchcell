---
id: 9ik29bej91s52j942u6mx61
title: Wandb_view_043
desc: ''
updated: 1792754241462
created: 1792754241463
---

## 2026.10.23 - Saved views for the overnight round, loss first

`experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py --round overnight_2026_10_22 [--window-means]`
writes config `round` and `arm_label` on the round's runs, tags the two zombie smoke runs
(`7y481s36`, `epf3gh0d`, no epoch logged) `aborted`, and overwrites the round view plus one
view per arm. Views filter on `Config('round')` (so the zombies never show), group by
`arm_label`, show up to 1,000 runs, smooth with a five-epoch running average, and keep only
keys their runs log. View ids: `results/wandb_view_ids.json`. Rerun after every sync and
after every new run; a rerun overwrites in place (verified: same six ids).

| section | panels |
|---|---|
| 1 loss | train and validation loss by epoch; the same by optimizer step (the matched-steps axis); Kemmeren train and validation loss by step (joint against alone); per-type train loss |
| 2 headline validation | expression per-gene median; doubles per-double centered with the additive measured line; protein per-gene median; promoter Pearson |
| 3 per-dataset validation | Kemmeren, Messner, Zelezniak, turnover, Caudal, Muenzner, Boocock, N'Guessan per-gene median |
| 4 train side | Kemmeren traineval against val; traineval expression, protein, promoter; gradient norm and clip fraction; learning rate; operator gates |
| 5 bookkeeping | epoch seconds, epochs fixed by the budget, registered window, steps per epoch, doubles scored |

The step axis is there because the alone run has 2,102 epochs and the joint runs 50: on
an epoch axis the joint curves occupy the first 2% of the panel, and the transfer
comparison is defined at matched optimizer steps. The overnight runs log no `val/loss` (it
was added in commit e6ad22d0c), so section 1 holds train loss alone until a new run logs it.

`--window-means` writes `results/<round>_window_means.json`: per run the mean over the
evaluated epochs of its registered window (read from the run's `setup/window_*` keys) from
the unsampled history, and the Kemmeren transfer (joint minus alone at matched steps).
