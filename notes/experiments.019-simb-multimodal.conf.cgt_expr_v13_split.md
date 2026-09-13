---
id: kiwzuyjoby4la38v3nrgtl5
title: Cgt_expr_v13_split
desc: ''
updated: 1789288504970
created: 1789288504970
---

## 2026.09.13 - The split round: four partitions, two readouts, one 90/10 arm

**Why.** Every expression number since `cgt_expr_011` is one validation draw: the partition is pinned to `split_seed 0`, and the one measurement made before the pin put the between-split sd at 0.0444 against a between-arm sd of 0.0058, with split 0's 155-strain validation draw +0.0893 above the other two (the config comment in [[experiments.019-simb-multimodal.conf.cgt_expr_011]] and `train_cgt_multitask.py`). No round has varied the split since, and no held-out test number exists. This round re-draws the partition and pairs the two readouts that matter on each draw.

**Arms** (`gh_expr_008_arm.sh`, the split rides in the arm name; `seed` is initialization only):

| arm | override | what it is |
|---|---|---|
| `V_ref_s<k>` | `data_module.split_seed=<k>` | H_ref: E_full input, pinball, shared MLP readout |
| `V_concat_s<k>` | `multitask.concat_context=true` + split | H_concat, +0.0135 over H_ref on split 0 at 1,399 epochs, 4 of 4 seeds |
| `V_ref_s0_90`, `V_concat_s0_90` | split 0, `data_module.fold_test_into_train=true`, `trainer.run_test=false` | the 517 test records moved into train (155 labelled expression strains), same val set: what 10% more training data buys |

**Layout.** 24 runs, four per A40 card, six tasks on the `gpu` partition, all A40 so card type never enters a contrast; each card holds one split's two arms for two seeds, so every ref/concat contrast is paired within card and seed.

| task | runs |
|---|---|
| 0 | split 0, seeds 0-1 |
| 1 | split 0, seeds 2-3 |
| 2 | split 0 at 90/10, seeds 0-1 |
| 3 | split 1, seeds 0-1 |
| 4 | split 2, seeds 0-1 |
| 5 | split 3, seeds 0-1 |

**Partitions** come from `make_split_indices.py`, hashed in `results/split_indices_manifest.json`; seeds 0-2 already existed on both machines with identical hashes, seed 3 was built on GilaHyper and copied to IGB. Split 1 has 151/150 labelled expression val/test strains, the others 155/155 ([[experiments.019-simb-multimodal.scripts.make_split_indices]]).

**Budget and memory.** 6,000 epochs (band on split 0: 0.188 at 4,000, 0.192 at 6,000, 0.197 at 9,900, sd 0.017 to 0.022 throughout; five of eight long arms peaked by 4,109). Every packed five-day IGB task so far died of host RSS (61 GB against 60 GB at 3.5 to 4.6 days, cause unmeasured), so: 120 GB per task (half a 250 GB node), `--cpus-per-task=12` giving two loader workers per run, and `data_module.persistent_workers: false` (new passthrough) so a worker's growth is released each epoch. No smoke run, by decision: a failure is the measurement. Wall 10 days.

**Test.** `trainer.run_test: true` scores the best-by-metric checkpoint (`trainer.checkpoint.monitor: val/expression/pearson_per_feature`, the first ModelCheckpoint is what `ckpt_path="best"` resolves to) on the held-out test strains; the first test number on this task. The 90/10 arms have none.

**Submitted 2026-09-13.** First as job `2397304` (source `3bc98bbc`): all six tasks started within a minute, but 3 of 24 runs died at dataset construction on a temp-file rename race (`gene_set.json.tmp` shared by co-resident runs; `FileNotFoundError` in `write_json_with_lock`; the advisory lock did not exclude them on IGB scratch). Fixed in `torchcell/utils/file_lock.py` (per-process temp names) plus a 30 s launch stagger in the launcher, the array was cancelled at ~10 min and resubmitted as job **`2397311`** (source `8d53e492`; the recorded diff hash is non-empty only because the runs rewrite the tracked `results/calmorph_train_target_norm_per_gene.json` on IGB), tasks 0-5 on compute-0-0, -0-1, -0-2, all RUNNING; W&B project `torchcell_019_expr_v13` (offline, synced from the login node with `igb_login_wandb_sync.sh`):

```bash
H=$(git rev-parse HEAD); D=$(git diff HEAD | sha256sum | cut -d" " -f1)
sbatch -p gpu --gres=gpu:1 --cpus-per-task=12 --mem=120g --time=10-00:00:00 \
    --array=0-5 --export=ALL,W5_STAGE=split,TORCHCELL_SOURCE_GIT_HASH=$H,\
TORCHCELL_SOURCE_DIFF_SHA256=$D experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
```

**Readout plan.** `roll_max` at the matched epoch across all 24; (1) the absolute score on four draws with within-draw replicate spread, (2) H_concat minus H_ref paired on 12 card-and-seed pairs, (3) 90/10 minus 80/10/10 on split 0 paired on seeds 0-1, (4) test at the best-val checkpoint on the 20 runs that have one. The linear baselines on the same index files run on GilaHyper CPU ([[experiments.019-simb-multimodal.scripts.expression_baselines_split]]).
