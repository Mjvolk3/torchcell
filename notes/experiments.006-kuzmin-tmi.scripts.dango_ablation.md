---
id: 86t8k0xm7aerf4ycfrk1udl
title: Dango_ablation
desc: ''
updated: 1792993903048
created: 1792993903048
---

## 2026.10.26 - Ablation of the reconstruction objective and the ReZero residual

### Question

Two claims about DANGO on the 006 trigenic build (STRING v9.1) that these runs can refute:
"pretraining doesn't really work, we can just train for the objective", and "ReZero lets
the model take the easy way out". One factor changes per arm; everything else is the served
STRING v9.1 configuration (`conf/dango_kuzmin2018_tmi_string9_1.yaml`).

"Pretraining" in this implementation is not a separate stage. The served
`LinearUntilUniform(transition_epoch=10)` weights the loss as
`alpha * reconstruction + (1 - alpha) * interaction`, with `alpha = 1` at epoch 0 (reconstruction
only), falling linearly to `0.5` at epoch 10 and staying at `0.5` for the rest of training. The
no-pretrain arm removes the reconstruction term entirely (`PreThenPost(transition_epoch=0)`,
`alpha = 0` from the first step); the reconstruction is still computed and logged unweighted,
but sends no gradient.

### Why the LMDB is bypassed

`scripts/dango.py` cannot read the build any more: `Neo4jCellDataset` fails on `ds[0]` with 538
pydantic validation errors (`perturbation_type: "deletion"` where the schema now wants
`sga_kanmx_deletion`; extra keys such as `deletion_description`; missing `released_kind` and
`environment.media.is_synthetic`; first breaking commit `933ee5ef3a`), and its
`processed/*.json.lock` files belong to uid 7474. DANGO needs only the three perturbed genes and
the interaction score per record, so `scripts/dango_ablation.py` reads them from the build's side
files under `$DATA_ROOT/data/torchcell/experiments/006-kuzmin-tmi/001-small-build/`, read-only
and without file locks:

- `processed/label_df.parquet`: 332,313 rows of record id and `gene_interaction`;
- `processed/is_any_perturbed_gene_index.json`: 4,036 genes, exactly three per record;
- `data_module_cache/index_seed_42.json`: the split the served runs used, train 265,851, val
  33,231, test 33,231 (asserted at start).

The gene index is the position in `sorted(genome.gene_set)` after `drop_empty_go()` (6,607
genes), with the six STRING v9.1 channels added by `build_gene_multigraph` and `to_cell_data`.
`load_records` and `build_cell_graph` are the helpers of
[[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_inference]], which verified this
convention against the served checkpoint 014mprap (best val MSE, epoch 81): recomputed val
Pearson 0.367222 against 0.367252 logged (delta -2.98e-05), val MSE relative delta 3.0e-05
(`experiments/010-kuzmin-tmi/results/dango_full_dataset_inference_check.csv`).

### Arms

| arm | config | loss schedule | residual | GilaHyper job |
|---|---|---|---|---|
| A baseline | `dango_ablation_baseline.yaml` | LinearUntilUniform, transition 10 | ReZero, gates init 0.01 | 6442 |
| B no_pretrain | `dango_ablation_no_pretrain.yaml` | PreThenPost, transition 0 | ReZero, gates init 0.01 | 6443 |
| C no_rezero | `dango_ablation_no_rezero.yaml` | LinearUntilUniform, transition 10 | plain `x + O(attn(x))` | 6444 |
| D no_pretrain_no_rezero | `dango_ablation_no_pretrain_no_rezero.yaml` | PreThenPost, transition 0 | plain | 6445 (held) |

B, C and D extend the baseline file; composing the four configs with Hydra shows they differ from
A only in `arm`, `wandb.tags` and their factor keys. The plain residual is
`Dango(residual="plain")` (`torchcell/models/dango.py`), which registers no `beta` parameter;
the default `"rezero"` keeps the served state_dict keys, parameter order and seeded init
(checked against `HEAD`), and `tests/torchcell/models/test_dango.py` covers both (34 passed).

Shared by every arm: STRING v9.1 graphs, lambda table from `determine_lambda_values()`
(neighborhood, coexpression, experimental 0.1; fusion, cooccurence, database 1.0), seed 42 for
the init and the train shuffle generator, AdamW lr 1e-5, weight decay 1e-6, batch 64, hidden 64,
4 heads, gradient clip 10, 150 epochs, one GPU, no DDP. The ReduceLROnPlateau scheduler is
registered but never stepped by `int_dango.RegressionTask`, so the learning rate stays at 1e-5,
as in the served runs. Validation and test use batch 512 (scores do not depend on batch
composition). Checkpoints: best val MSE (`save_top_k=1`) and last, under
`$DATA_ROOT/models/checkpoints/<group>/`. After training, the best-val-MSE checkpoint is scored on
test once and written to `test_metrics_best_val_mse.json` in that directory and to the W&B
summary (`best_mse_ckpt/*`). Logged per epoch: `val/gene_interaction/{Pearson,MSE}`,
`train/alpha`, `train/reconstruction_loss`, `train/interaction_loss`, `perf/epoch_wall_s`,
`perf/val_wall_s`, `perf/gpu_max_allocated_gib`, and `model/beta1`, `model/beta2` for the ReZero
arms. W&B project `torchcell_006-kuzmin2018-tmi_dango_ablation`, never the served
`torchcell_006-kuzmin2018-tmi_dango` (which `dango_full_dataset_si.py` reads in full).

Difference from the served runs: they used DDP at batch 64 per rank, so 014mprap (IGB, 2 ranks)
took optimizer steps of 128 records (2,077 per epoch) and the Delta runs steps of 256
(`experiments/010-kuzmin-tmi/results/dango_full_dataset_runs.csv`). These arms take steps of 64
(4,154 per epoch), so their epoch axis is not comparable to the served curves; arm-to-arm
comparison is. Hypothesis (untested): the val Pearson peak arrives at an earlier epoch than the
served 98 to 140.

### Smoke test (CPU, 2026-10-26)

No card was idle (all four RTX 6000 Ada at 63 to 91% utilization with 043 jobs), so the smoke
ran on CPU with `limit_train_batches=3`, `limit_val_batches=2` and `limit_test_batches=2`
(1,024 val and test records), W&B offline; its artifacts were moved out of the checkpoint and
W&B trees afterwards.

- All arms: split sizes 265,851 / 33,231 / 33,231; cell graph 6,607 genes with the same edge
  counts as the inference script (neighborhood 52,208, fusion 7,965, cooccurence 9,266,
  coexpression 320,295, experimental 226,057, database 40,093, with self loops); finite losses;
  val Pearson and MSE logged; best and last checkpoints written; test scored on the best
  checkpoint; every best checkpoint loads into `Dango` with `strict=True`.
- A: epoch-0 train losses 0.0041086, 0.0041050, 0.0041014 with `alpha = 1.0`; `alpha = 0.95` at
  epoch 1. Parameters 3,138,270.
- B: `train/alpha = 0.0` at epoch 0 and `train/loss` equal to `train/interaction_loss`
  (0.0015416).
- C: no `beta` parameter in the model or the checkpoint state_dict; parameters 3,138,268. Its
  first three losses equal A's to every printed digit, as expected for a reconstruction-only
  epoch 0 from the same seed, which confirms the shared init and shuffle.
- D: `alpha = 0.0` and no `beta` parameter.
- Peak RSS 7.2 GB (`/usr/bin/time -v`, arm A), hence `--mem=24g` rather than 64g.
- Not measured: GPU step time, epoch time and GPU memory. The first job to start logs them.

### Submission (2026-10-26 00:55)

`scripts/gh_dango_ablation_<arm>.slurm`: partition `main`, one GPU, 8 CPUs, 24g. `--time` is a
2-day placeholder because the GPU epoch time could not be measured; if 150 epochs do not fit,
resume from `last.ckpt` with `trainer.resume_ckpt_path=<path>`. All three were PENDING
(Priority) behind 043 jobs at submission, with `squeue --start` placing 6442 at 2026-10-26 09:04;
6445 (arm D) is held (`scontrol release 6445` to queue it). Logs go to the primary checkout's
`experiments/006-kuzmin-tmi/slurm/output/dango-abl-<arm>_<jobid>.out`. The jobs read code from
the `fig/dango-si-completion` worktree when they start, so that worktree must not be removed
while any of them is pending.

`scripts/igb_dango_ablation_<arm>.slurm` are cabbi variants (5 days, one GPU, W&B offline,
singularity `rockylinux_9.sif` because `dango.py` imports `torch_scatter`, which needs GLIBC 2.32
and IGB runs 2.28), written and not submitted. They need the 006 side files and `data/sgd`,
`data/go`, `data/string`, `data/tflink` copied to the IGB data root first.

### How the result will be read (plan, not a result)

Per arm: best val Pearson over epochs (a raw maximum, an upward-biased order statistic),
val Pearson at the best-val-MSE epoch, and test Pearson of the best-val-MSE checkpoint. One seed
per arm: the only spread available is the five served v9.1 runs (best val Pearson 0.3652 to
0.3676), which differ in DDP layout, so a difference of that size between arms is not
distinguishable from seed noise. For the ReZero claim, the gate trajectories
`model/beta1`, `model/beta2` of A and B are read beside C. Context, measured on the served
checkpoint 014mprap at epoch 81: gates 0.073 and 0.065, and zeroing both drops val Pearson from
0.367 to 0.079 (`experiments/010-kuzmin-tmi/results/dango_rezero_ablation.csv`), so in that model
the gated attention branch carries most of the predictive signal rather than being closed.
