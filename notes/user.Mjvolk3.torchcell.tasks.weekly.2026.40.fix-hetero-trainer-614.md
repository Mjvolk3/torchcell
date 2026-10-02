---
id: tod64kg2rm6gec2xs9mbmua
title: fix-hetero-trainer-614
desc: ''
updated: 1790979430056
created: 1790979430056
---

## 2026.10.02

- [x] PR (branch `fix/hetero-trainer-614`): FIX(trainers) issue #614 and the rest of #596 in the hetero-cell trainer: the plateau scheduler is stepped on the val MSE at validation end, scheduler types and accumulation schedules are validated at construction (five 006 `{0:16}` / `{0:8}` configs corrected), batches are sized by `num_graphs`, the effective batch size counts `trainer.world_size`, the diffusion task no longer scores its training placeholders, both tasks share one `_shared_step`, unit mismatches and metric errors are refused. COO layout (item 7) left open. Notes: [[torchcell.trainers.int_hetero_cell]]; tests in [[tests.torchcell.trainers.test_int_hetero_cell]], [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]].
