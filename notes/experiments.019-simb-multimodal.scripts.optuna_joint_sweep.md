---
id: t360bc94y1qjc6tauxz32q5
title: Optuna_joint_sweep
desc: 'Delta controlled expr<->morph auxiliary-task experiment — does expression help morphology prediction on Ohya (and vice versa)'
updated: 1784771704373
created: 1784771704373
---

## 2026.07.22 - Delta controlled expr<->morph auxiliary-task sweep

**Question:** does expression data improve MORPHOLOGY prediction on Ohya — and does
morphology rescue expression's val floor? This is the abstract's cross-phenotype-class claim
made falsifiable: *the same latent cell embedding generalizes across qualitatively different
phenotype classes relevant to industrial strain design.*

Related: [[experiments.019-simb-multimodal.scripts.optuna_morph_sweep]] (morphology-ALONE on
all 4718 Ohya, IGB) · [[experiments.019-simb-multimodal.experimental-plans]] ·
[[experiments.019-simb-multimodal.fig3-expression-experiments]].

### The control (why this is clean, not just "a joint model")

To attribute a gain to *"expression helps"* and NOT to *"more data,"* the instance set must
be held fixed. So all conditions run on the **1,440 genotypes that carry BOTH modalities**
(`Q3_expression_and_morphology: 1440` in `results/fig3_overlap_census.json`; of these 1,326
also have fitness). The primitive is a new harness filter **`cell_dataset.require_modalities:
[expression_log2_ratio, calmorph]`** — the INTERSECTION over phenotype-type presence
(mirrors `restrict_dataset_names`, which is a union over dataset names), using the dataset's
`phenotype_label_index`. Only the active heads then vary:

| CONDITION | active_heads | objective |
|---|---|---|
| `expr` | `[per_gene]` | `val/per_gene/pearson_per_gene` |
| `morph` | `[global]` | `val/global/pearson_per_gene` |
| `joint` | `[per_gene, global]` | mean(expr, morph) honest r (both logged) |

**Read-out:** `joint − morph` = does expression help Ohya morphology; `joint − expr` = does
morphology rescue expression. Same 1,440 split across all three → the delta IS the
auxiliary-task effect. ("No backprop on instances without both" is realized by the fixed
instance set, so every gradient step sees both modalities in the joint condition.)

### Sweep

Per-modality target-norm is swept (expression: raw|zscore; morphology: raw|yeo_johnson|zscore
— `zscore` = the anti-mean-collapse lever), plus `hidden{16,32,64}`, `layers{2,3}`,
`graph_reg{0,.001}`, `hp_profile{baseline,aggressive}`. Single-GPU per trial, 4 Optuna
workers pinned per GPU over one shared SQLite study **per condition** (3 studies).

### Delta specifics

- 48 h walltime cap; compute nodes **have internet → W&B ONLINE** (no offline/sync dance).
- `gpuA40x4`, account `bbub-delta-gpu`, apptainer `rockylinux_9.sif`, project on
  `/scratch/bbub/mjvolk3/torchcell`, conda `/projects/bbub/miniconda3/envs/torchcell`.
- Single-GPU per trial also sidesteps the DDP `find_unused_parameters` failure the masked
  multitask heads trip (job 1012).

### Files

- `scripts/optuna_joint_sweep.py` — condition-parameterized driver (this note).
- `conf/delta_joint_expr_morph_000.yaml` — base config (`require_modalities`, single-GPU).
- `scripts/delta_joint_expr_morph.slurm` — 4-worker launcher (`--export=ALL,CONDITION=…`).
- Harness: `run_training` gained `cell_dataset.require_modalities` (intersection filter).

### Launch

1. **Data → Delta:** rsync the 13 GB `fig3_core` tree to
   `/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/019-simb-multimodal/fig3_core`
   (a Delta variant of `sync_igb_fig3_core.sh`).
2. **Code → Delta:** `git checkout feat/igb-mmli-optuna-morph` on Delta.
3. **Launch all three conditions** (each its own study):

   ```bash
   for C in morph expr joint; do
     sbatch --export=ALL,CONDITION=$C experiments/019-simb-multimodal/scripts/delta_joint_expr_morph.slurm
   done
   ```

### Open

- Head-weight balance (`head_weights`) fixed 1:1 for v1 — add as a joint-only sweep axis if
  the joint condition underperforms both baselines (a sign one head dominates the loss).
- Needs a 1-GPU smoke (compose → `run_training` → `require_modalities` prints
  `[require_modalities] … -> ~1440 rows`) before the full launch.
- After the rename refactor: objectives become `val/{expression,morphology}/pearson_across_instances`.
