---
id: 581teasjygct0jwbqyhi1a0
title: Gh_eval_ckpt_predictions
desc: ''
updated: 1789698598951
created: 1789698598951
---

## 2026.09.17 - Scoring saved checkpoints on GilaHyper and dumping per-gene predictions

`experiments/019-simb-multimodal/scripts/gh_eval_ckpt_predictions.slurm` rebuilds a run's arm through `gh_expr_008_arm.sh` with `trainer.eval_ckpt_path=<ckpt>` (train_cgt_multitask.py): the task is built from the run's own config and partition, the checkpoint's weights are loaded strictly, `trainer.validate` logs the run's `val/...` metrics (which must reproduce the W&B value at the checkpoint's epoch), and per-gene point predictions in raw units are written for the validation and test splits to `$DATA_ROOT/{val,test}-predictions/<source run group>.json`, named by the checkpoint's own group so they join back to the W&B run.

Checkpoints came from IGB (`$DATA_ROOT/models/checkpoints/<group>/<job>-best-metric-epoch=<e>.ckpt`, 80 MB each, rsync over the login node) into the same path on GilaHyper, and each is symlinked to `best_metric.ckpt` because Hydra's override grammar rejects an `=` inside a value.

The first submission (GilaHyper job 2339) sat behind a stuck `COMPLETING` job from another session, so the six evaluations ran directly on GPU 0 with the same arm script; the slurm file is the reproducible form.

Consumer: [[experiments.019-simb-multimodal.scripts.variance_stratified_pearson]].
