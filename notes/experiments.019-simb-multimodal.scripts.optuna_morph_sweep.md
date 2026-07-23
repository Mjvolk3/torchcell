---
id: bihf46mb4j04aw83h4hk958
title: Optuna_morph_sweep
desc: 'IGB mmli offline Optuna MORPHOLOGY sniff sweep — setup + launch runbook (SIMB 2026 Fig-6 strand)'
updated: 1784768936036
created: 1784768936036
---

## 2026.07.22 - IGB mmli Optuna morphology sniff sweep

The morphology counterpart to the GilaHyper EXPRESSION grid — a single-GPU, small-model
sniff sweep on the CalMorph `global` head (278 features, **n_train = 3757**), run with
**Optuna** on the IGB BioCluster `mmli` partition. Closes the abstract's cross-phenotype-class
claim: the same latent cell embedding must generalize expression (Fig-3) AND morphology
(Fig-6). Do morphology **alone** first (this sweep), then expression+morphology **jointly**.

Related: [[experiments.019-simb-multimodal.experimental-plans]] ("IGB is OFFLINE → Optuna
path") · harness [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]] ·
expression side [[experiments.019-simb-multimodal.fig3-expression-experiments]].

### Objective + search space

Maximize `val/global/pearson_per_gene` — the **honest per-feature Pearson (across
instances)**; the per-strain Pearson stays high under mean-collapse, so it is NOT the
objective (same across-instances vs across-features logic as expression).

Search space (fixed split **seed=0** so trials are comparable; seed-replicate the winners
after), `3·2·3·2·2 = 72` discrete combos:

| lever | values |
|---|---|
| `hidden_channels` | 16, 32, 64 |
| `num_transformer_layers` | 2, 3 |
| `target_norm` | raw, yeo_johnson, **zscore** (anti-mean-collapse) |
| `graph_reg_lambda` | 0.0, 0.001 |
| `hp_profile` | baseline (lr 3e-4/do 0.1/wd 1e-8), aggressive (lr 1e-3/do 0.0/wd 1e-4) |

The `zscore` branch is the key morphology test: if raw/yeo_johnson collapse to the
per-feature mean on val (as 5M-param expression did), z-scoring each CalMorph feature forces
prediction of DEVIATIONS and should be the lever that lifts the honest metric.

### Architecture — why NOT Hydra submitit

`mmli` is a **single node** (compute-5-7, 4 GPUs). Submitit submits each trial as a separate
SLURM job, which just queues them behind each other on one node. Instead we use Optuna's
canonical multi-worker pattern: **4 workers, one pinned per GPU** (`CUDA_VISIBLE_DEVICES`),
all pulling trials from **one shared SQLite study** on scratch. One `sbatch`, one singularity
container, offline W&B.

**Bonus: single-GPU sidesteps a real DDP bug.** The prior 4-GPU morph run
(`gh_cgt_multitask_morph_000`, job 1012) **failed** with DDP `find_unused_parameters`:
with only `active_heads=[global]`, the other heads' params are unused in the loss and DDP
rejects that. The forward/model/data pipeline all worked (it logged `v_num` at Epoch 0
step 1/32 before the all-reduce error). Single-GPU (`strategy=auto, devices=1`) has no DDP
wrapper, so unused head params never trip this — the sweep avoids the bug for free.

### Files

- `scripts/optuna_morph_sweep.py` — Optuna driver (this note); one process = one worker.
- `conf/igb_mmli_optuna_morph_000.yaml` — base config (global head, single-GPU, EarlyStopping).
- `scripts/igb_mmli_optuna_morph.slurm` — mmli launcher (singularity, 4 GPU-pinned workers).
- `scripts/sync_igb_fig3_core.sh` — rsync the 13 GB `fig3_core` tree GilaHyper→IGB.
- Harness: `run_training` now RETURNS `trainer.callback_metrics` (additive; `main` ignores it).

### Launch runbook

1. **Data** (from GilaHyper) — mirror the built dataset (offline nodes can't reach Neo4j;
   `Neo4jCellDataset` skips the DB when `processed/` exists):

   ```bash
   bash experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh
   ```

   Lands at `/home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/019-simb-multimodal/fig3_core`
   (matches `run_training`'s `dataset_root` = `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/<dataset_tag>`).

2. **Code** (on IGB) — sync the repo and check out THIS branch (not `main`, so the GilaHyper
   expression grid on `main` is untouched):

   ```bash
   cd /home/a-m/mjvolk3/projects/torchcell
   git fetch origin && git checkout feat/igb-mmli-optuna-morph && git pull
   ```

3. **Launch**:

   ```bash
   sbatch experiments/019-simb-multimodal/scripts/igb_mmli_optuna_morph.slurm
   ```

4. **Monitor**: `squeue -p mmli -u mjvolk3`; tail
   `scratch/torchcell/experiments/019-simb-multimodal/slurm/output/019-morph-optuna_<jobid>.out`.
   Inspect the study:

   ```bash
   python -c "import optuna; s=optuna.load_study(study_name='morph_000', \
     storage='sqlite:////home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/optuna/optuna_019_morph.db'); \
     print(len(s.trials), s.best_value, s.best_params)"
   ```

5. **W&B sync** — offline runs are pinned to shared scratch via `WANDB_DIR`, at
   `/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/wandb/offline-run-*`.
   IGB compute nodes are offline but **login nodes have internet**, and scratch is shared, so
   sync from a `biologin` login node after the run:

   ```bash
   cd /home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal
   wandb sync wandb/offline-run-*
   ```

### Open / next

- After the sweep: pull the leaderboard, compare `raw/yeo_johnson/zscore` on the honest
  metric, seed-replicate the top-k, then wire the **joint** expression+morphology run to test
  shared-embedding transfer (the abstract's cross-phenotype-class claim).
- The `per_gene`→`across_instances` metric rename (blocked on the expression grid draining)
  will rename this objective to `val/morphology/pearson_across_instances`; update the driver's
  `OBJECTIVE_METRIC` when that lands.
