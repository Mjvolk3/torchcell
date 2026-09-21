---
id: 2i0afb161nceoc9daef9mae
title: Wandb_regroup_by_arm
desc: ''
updated: 1790010438463
created: 1790010438463
---

## 2026.09.21 - Group pages per arm

The 019 trainer writes a W&B group unique to each run (`<host>-<jobid>_<hash>`), so no group page spans an arm. This script rewrites the group on the server to the arm family and leaves the run name alone, since the name is the checkpoint directory. [[experiments.019-simb-multimodal.scripts.eval_ckpt_manifest]] now reads the directory from the name; on v13 and v14 the name matched the old group for 24 of 24 and 16 of 16 runs.

Re-run it after every `wandb sync` of a live run, because a sync replays the original group. Output: `experiments/019-simb-multimodal/results/wandb_groups_<round>.json`.

v16 (IGB `gpu` partition, job 2409562, six cards of three runs), 18 runs regrouped:

- <https://wandb.ai/zhao-group/torchcell_019_prot_v16/groups/J_joint>
- <https://wandb.ai/zhao-group/torchcell_019_prot_v16/groups/J_ref>
- <https://wandb.ai/zhao-group/torchcell_019_prot_v16/groups/J_expr>
