---
id: f35auvn0quiy1hfpdcet80z
title: Igb_mmli_cgt
desc: ''
updated: 1788924442365
created: 1788924442365
---

## 2026.09.08 - IGB mmli Launcher for the 025 Full Build

4-GPU DDP launcher for one 025 arm on the IGB `mmli` node (compute-5-7, 4x A100 40 GB,
500 GB, no time limit), the IGB counterpart of `gh_cgt.slurm` and `delta_cgt.slurm`.
Takes the Hydra config name plus free overrides. Written for the joint-fitness arms
([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]]) but runs
any 025 config.

### What is on IGB

`sync_igb_025_build.sh` mirrored the build to
`/home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build/`.
Checked 2026.09.08: `processed/lmdb/data.mdb` is 554,075,586,560 bytes on both machines
with the same mtime, `raw/lmdb` is the empty stub PyG's `_download` needs, and
`data_module_cache/` holds the split indices. `data/sgd/genome`, `data/go/go.obo`,
`data/string` and `data/tflink` are present. The 010 build is not on IGB; only the 025
full build is, so every IGB arm reads the default `dataset.root_rel`.

### Why the job checks out its own worktree on the compute node

IGB's main clone (`~/projects/torchcell`) sits on the 019 branch and must not be flipped
under a job. The login node is access and transfer only, and a `git worktree add` there
checks out 12k files, which drew a warning in July 2026. Compute nodes have no network.
So the split is: `git fetch origin <branch>` once on the login node (network, light), and
the job itself runs `git worktree add --detach <path> origin/<branch>` on the compute node
at first use, or `checkout --detach origin/<branch>` after, printing the SHA it runs. A
later push changes what a later job runs, never a running one. The worktree gets a copy of
the main clone's `.env`; the exports in the script win over it since `load_dotenv` never
overrides a set variable, and `EXPERIMENT_ROOT` points at the worktree so the index
artifacts under `results/` are the branch's own.

## 2026.09.10 - The compute nodes have no git; the worktree is prepared on the login node

The design above never ran. The first chain submitted through it (2390540 to 2390543,
the Q-split embedding arms) died in 12 seconds at `git: command not found` on
compute-5-7, and `rockylinux_9.sif` carries no git either, which the 019 wave-5 launcher
had already recorded from its own canary (2324270). With the chain on `afterany`, all
four failed in sequence.

The launcher now expects the worktree to exist and only verifies it: it reads
`gitdir:` from the worktree's `.git` file and prints the SHA from that directory's
`HEAD` (a detached HEAD holds the SHA directly). The checkout moves to the login node,
which is what the 019 launchers do:

```bash
git -C ~/projects/torchcell fetch origin feat/025-fitness-joint-head
git -C ~/projects/torchcell worktree add --detach \
    ~/projects/torchcell.worktrees/025-fitness-joint-head origin/feat/025-fitness-joint-head   # first time
git -C ~/projects/torchcell.worktrees/025-fitness-joint-head checkout --detach origin/feat/025-fitness-joint-head  # afterwards
```

A job reads whatever the worktree holds when it starts, so the checkout is advanced
only between jobs, never under a running one. The preflight also checks the four
sequence-embedding builds (`fudt`, `calm`, `protT5`, `random`) on IGB scratch.

### Budget

The weekly note measured the KL arm at ~58 min per epoch on this node against 19.4 on
GilaHyper. The replication (job 1598) peaked at epoch 14 and plateaued through epoch 35,
so the 4 day wall clock (~99 epochs) reads the plateau. Three arms need the whole node
each (4 GPUs, 250 GB), so they run one after another: about 12 days for the three.

### Launch, 2026.09.08

Queued three runs at seed 42, differing in the fitness weight only:

```bash
sbatch -J 025-kl-fit1   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_008
sbatch -J 025-kl-ctrl   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_000
sbatch -J 025-kl-fit0.1 experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_009
```

Runs log offline; sync from the login node with
`experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh` and read progress from
W&B on GilaHyper, never from the IGB logs.

## 2026.09.15 - Fitness seed 2 on cabbi

`cgt_s0_r_kl_fit_014 +seed=2` (joint interaction + fitness on the random split, weight
1.0) submitted 00:05 as cabbi 2400109 from a fourth login-node worktree
`~/projects/torchcell.worktrees/025-fitness-joint-head-d` at 0f476a6c, the first IGB
run carrying the layer-restricted attention path, the gradient probe and the peak-memory
log. compute-3-3 (8 x RTX 6000) had five GPUs free after the flanks run 2395008 finished
its 100 epochs (COMPLETED, 1 d 12 h 28 m). Delta holds the same arm as 22034667 (chain
head, waiting on Priority); whichever finishes first is seed 2, the other a fourth
replicate. The only complete run of this arm so far is Delta 22030924 (seed 1, W&B
b3n4ax4a): validation Pearson 0.4523 at epoch 18, a max over 30 epochs, against 0.447
to 0.462 for 010's interaction-only checkpoints.

### Replaced by the closure composite, 00:20

2400109 cancelled at 25 min (Delta already holds `fit_014` seed 2 as 22034667) and
cabbi 2400200 submitted in its place: `cgt_s3_r_kl_fit_030 +seed=1`, the joint objective
on S3, the closure composite of the 376,732 triples: every single (5,691), every double
whose gene pair lies inside some triple (739,222 doubles, dmf + dmi, 694 of them carrying
a SynthLethDB record) and the triples (tmf + tmi); 1,121,645 records, of which 1,046,299
train and the pinned 37,673 + 37,673 triples are validation and test. No essentiality:
the build holds none. First run of the `unpinned_to_train` path on a GPU. Expected about
36 h on this node by scaling from the S0 rate, unmeasured.

### 01:47: restarted for the per-order metrics, and lost the GPUs to a 019 array

2400200 was not hung: its first batch took about an hour of cold reads (the closure's
1,046,299 records are spread over the 554 GB LMDB, `num_workers=0`, so every record is
a serial random read until the page cache holds them), then ran at about 25 s per batch,
which is a 7 h first epoch; later epochs should be cache-warm and far faster. Cancelled
at 1 h 10 m so the run carries the per-order metrics from the start, and resubmitted as
2401111 from worktree `-d` advanced to 6bee6bde. The four freed GPUs were taken within a
minute by a 019 wave-5 array (2400350, one GPU each) queued at 00:39 from another
session, so 2401111 is pending on Resources behind vkp5's three tasks and the four 019
tasks; it starts when four GPUs free. Check `squeue -p cabbi` before cancelling a job
that holds GPUs another queue wants.

## 2026.09.16 - Second mmli chain: the joint arm and the matched control to three seeds

Worktree `-e` at b2602cca (val batch 32, gradient probe, per-order metrics all included). Chain, `afterany`, 4 GPUs each: 2408604 `cgt_s0_q_kl_embfit_027 +seed=1`, 2408605 `+seed=2`, 2408606 `cgt_s0_q_kl_rand_018 +seed=1`, 2408607 `+seed=2`. The node's four A100s are held by another user's array with no time limit, so the chain starts when four free at once. The seed sets `seed_everything` (weights, dropout, loader workers) and the data module's shuffle order; the split is pinned by the Q artifact and does not move with it.

## 2026.09.16 - S3 closure arm moves from cabbi to the mmli chain

cabbi 2401111 was cancelled (its start estimate had slipped to 2026-09-25 behind another user's jobs and the 019 array) and resubmitted on mmli as 2408785 (`cgt_s3_r_kl_fit_030 +seed=1`, worktree `-e`), inserted after the second composite-plus-fitness seed and ahead of the random-vector seeds: 2408604 -> 2408605 -> 2408785 -> 2408606 -> 2408607. S3 is 1,121,645 records; the 010 random split pins the triples, every other record trains, and `per_order_metrics` writes train metrics per order so the triple fit reads apart from the doubles. Expected about 19 hours at 4 A100s.

## 2026.09.16 - S3 at 130 epochs, first in line

The 30-epoch S3 cell (2408785) was replaced by `cgt_s3_r_kl_fit_031` (2408791, seed 1, worktree `-f` at 5efb7f34): the same closure arm with the epoch ceiling at 130 under the constant 2.5e-4 rate, about 3.3 days at the measured S0 rate scaled by 2.98x, inside the 4-day clock. It starts as soon as the running composite-plus-fitness seed 1 (2408604) ends; the second composite-plus-fitness seed and the two random-vector seeds now queue behind it. The trainer keeps best-Pearson, best-MSE and last checkpoints per run under `$IGB_DATA_ROOT/models/checkpoints/<group>/`; there is no early-stopping callback, so a run that is clearly finished is stopped by hand and its best checkpoint stands.

## 2026.09.17 - The closure cell died on three conflicting singles; masked and resubmitted

Job 2408791 (S3, 130 epochs) failed 1 h 48 min in, at epoch 0: one rank raised `fitness: a batch row carries more than one value` and the other three died on the 30-minute NCCL timeout. Three single-deletion genotypes in S3 carry two fitness values, a measured Costanzo 2016 single-mutant fitness near 1.0 beside a SynthLethDB single-gene record at 0.0 (YPL212C at record 10303289, YHL047C at 13253926, YKL117W at 13254909). No S0 arm ever met one because the triples carry exactly one value per label. The decode (`RegressionTask._coo_label`, commit 3090299f) now masks such a row out of the loss and every metric and counts it in `coo_conflict_rows`; the two values are not averaged, since 0.5 is a fitness nobody measured. Resubmitted from worktree `-g` as the job printed by the launcher, chained behind the running random-vector seed 1 (2408606) with random-vector seed 2 behind it. Composite-plus-fitness seed 2 (2408605) completed at 11:12 and awaits sync.

## 2026.09.17 - Closure replicates queued; the first closure epoch is cold

Seeds 2 and 3 of `cgt_s3_r_kl_fit_031` are queued (2408993, 2408994) behind random-vector seed 2, all from worktree `-g`, so the chain is S3 seed 1, random-vector seed 2, S3 seed 2, S3 seed 3. Seed 1 (2408888) passed the masked conflict row at epoch 0 and is training, but its first epoch runs at about 16 s per step against 2.4 s on the triples-only cells, 294 of 1,022 steps in 78 minutes. Hypothesis (untested until epoch 1 reports): the 1.1M-record pool is being read cold from the LMDB and the second epoch will run near the projected 36 minutes once the page cache holds it. If the rate does not recover, the 4-day clock ends the cell near epoch 21, with the best-Pearson and last checkpoints kept.

## 2026.09.18 - Cold epoch confirmed; single-gene fitness was never logged per order

The cold-cache hypothesis held: seed 1 (2408888) took 4 h 21 min for epoch 0, 36 min 24 s for epoch 1, and epochs 2 and 3 ran at the same 0.47 to 0.50 steps per second. On the first read after the sync (rank-0 run `kj03xx8y`, epochs 0 to 3), val interaction Pearson on the pinned triples ran 0.405, 0.415, 0.427, 0.430 and val fitness Pearson 0.909, 0.922, 0.930, 0.933; four epochs of a 130-epoch constant-rate run are not a result, only a sign the cell trains. Train counts per order were 739,217 doubles and 301,387 triples and zero singles. The zero exposed a logging bug, not a data one: the per-order record counter advanced only for the interaction collection, and the epoch-end log gated the fitness collection on that same count, so the 5,694 singles, which carry fitness and no interaction label, had their fitness metrics computed and discarded every epoch. Commit 2e96f24f gives each collection its own count and names the label in the count key (`<stage>/n_records/<label>/order<k>`), with a test that feeds a single through both collections. Seed 1 keeps running on the old code (its order-1 fitness can be read later from its checkpoints); the queued replicates were cancelled and resubmitted from worktree `-h` at 2e96f24f as 2409031 (seed 2, after random-vector seed 2) and 2409032 (seed 3, after seed 2). The four seed-1 runs are grouped on W&B as `s3_fit_031`.
