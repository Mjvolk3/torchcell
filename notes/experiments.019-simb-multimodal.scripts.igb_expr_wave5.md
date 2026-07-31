---
id: fri3l10w8pqt6wtl6cswxso
title: Igb_expr_wave5
desc: ''
updated: 1785531973390
created: 1785531973390
---

## 2026.07.31 - Powering the Post-Perturbation Mixing Test on Borrowed IGB GPUs Without Letting the Partition Become the Variable

H2 asks whether mixing gene tokens AFTER the perturbation lets one gene tell another how to express -- the encoder runs on the wildtype graph at batch 1, so `H_genes` is strain-independent and nothing routes between gene tokens once the perturbation is applied. The evidence against that channel was n=1 on seed 0 (`E0_perceiver_on` -0.0302, `GEARS_crossgene` -0.0243) and carries no significance, so this launcher buys 8 paired replicates per arm on mmli A100s and adds the cabbi probe and the 10,000-epoch `long` stage. Everything else in the file is what it costs to make an IGB run comparable to a GilaHyper one and attributable to a source commit.

| stage | partition | arms x seeds | runs/GPU | epoch cap | wall | reachable |
| --- | --- | --- | --- | --- | --- | --- |
| `canary` | cabbi | `X_mix` x 1, `fast_dev_run` | 1 | -- | 00:30 | minutes |
| `h2` | mmli | 3 x 8 = 24 | 2 | 300 | 20 h | 3 rounds, fits |
| `probe` | cabbi | 4 x 2 = 8 | 1 | 400 | 20 h | 4 rounds |
| `long` | mmli | `W_ref` x 2 | 2 | 10000 | 20 h | cap NOT reachable, by design |
| `wave6` | mmli | 12 x 1 | 3 | 10000 | 48 h | ~7,350 at 23.51 s/epoch |

- **Resources come from the sbatch COMMAND LINE, and packing is set per partition by VRAM.** One run measured 20,026 MiB: an A100 80 GB therefore takes 3 (~20 GB headroom) and a cabbi RTX 6000 (24 GB) takes exactly ONE. Packing is uniform WITHIN a stage -- a paired reference is only a valid denominator for arms that ran beside it -- and deliberately NOT uniform across stages, so an mmli arm is never compared to a cabbi arm and each stage co-launches its own `W_ref` (`_007` pooled A100 with RTX 6000 Ada; means 0.0242 vs 0.0182).

- **mmli packing then had to come DOWN to 2, and the cause was starvation, not VRAM.** The first submission (2324287) ran 3 runs/GPU at `--cpus-per-task=12`, giving each run (12-2)/3 = 3 dataloader workers against GilaHyper's 7. After 14 minutes all 12 runs sat at epoch 1-2 while GH had reached ~23 and cabbi (1 run/GPU, 6 workers) 5 -- with the A100s reading 0% utilization at 25 GB, i.e. idle waiting on data, worsened by LMDB reads over GPFS. At that rate 400 epochs x 3 rounds needed ~50 h against a 20 h wall. Fix: submit at `--cpus-per-task=16` so 2 runs/GPU yields 7 workers, matching GH. `X_mix64` (a capacity variant) was dropped to pay for the replicates: 3 arms x 8 seeds = 24 runs at 2/GPU = 3 rounds.

- **No requeue chain, deliberately.** The `_007` launchers derived `--time` from a hardcoded DEADLINE minus 120 min of slack and also passed `--deadline`; when a job waited in queue longer than the slack, its TimeLimit no longer fit and Slurm killed it instantly in DEADLINE state at 0 s elapsed, silently -- that is what happened to 2323142. Both partitions are `MaxTime=UNLIMITED`, so a plain fixed walltime removes the whole failure mode.

- **W&B runs offline because compute nodes have no internet.** `WANDB_MODE=offline`; runs land under `$DATA_ROOT/wandb-experiments/<group>/wandb/offline-run-*` and are synced afterwards from the LOGIN node with `scripts/wandb_sync_offline.sh`.

- **Source provenance is resolved at SUBMIT time, on the login node, because nothing downstream has git.** Two canaries established where git actually lives: 2324270 died with `FileNotFoundError('git')` when the train script shelled out to `git rev-parse` inside `rockylinux_9.sif`, and 2324272 died the same way after the call moved into the launcher, because **IGB compute nodes have no git either** (`which git` returns nothing on compute-3-3, while `/usr/bin/sha256sum` is present). So the submitting shell exports the values through `--export`:

```bash
H=$(git -C "$WT" rev-parse HEAD)
D=$(git -C "$WT" diff HEAD | sha256sum | cut -d" " -f1)
sbatch ... --export=ALL,W5_STAGE=h2,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D ...
```

  A `command -v git` branch keeps the same script self-resolving on a host that does have git (GilaHyper). If neither path yields a hash the run records "unavailable" and says so loudly -- an unattributable run must not be pooled with attributable ones.

- **`set -u` is relaxed around the cluster bootstrap and inside the container.** IGB's `/etc/bashrc` line 12 references `BASHRCSOURCED` before defining it, so under `-u` sourcing it aborts the job with "BASHRCSOURCED: unbound variable" -- exactly how the first canary (2324269) died, before reaching python. Conda's activate scripts are the same class of failure inside the container. `-e` and `pipefail` stay on in both places.

- **One `singularity exec` per run, with the loop in the OUTER shell**, so arm, seed, paths and worker count are expanded on the host and the container receives a fully-formed command. A loop nested inside `bash -c "..."` has to escape a bash array through several quoting layers, which is how a launcher silently runs the wrong arm.

- **Job ordering is seed-major**, so each task's co-resident runs are DIFFERENT arms rather than copies of one. The metric is unaffected either way (the budget is a fixed epoch count, so a slower neighbour costs wall time, not score), but arm-blocked packing would align "what you share a GPU with" perfectly with the arm axis, and there is no reason to leave that correlation in.

- **The `long` stage takes the saturation question past its cap.** The 1500-epoch GH runs had eval-mode TRAIN pearson still rising at 0.64-0.72 with `max == last` in 4 of 4 runs, and val pearson dipping around epoch 200-300 before climbing past its early peak to a project-best 0.1980 at epoch 1367 -- neither had plateaued, so the cap was the binding constraint, not the model. 10,000 epochs is deliberately far past where anything is expected to move; ModelCheckpoint keeps best-by-loss AND best-by-metric AND last, so cancelling the moment the curves flatten costs nothing. Read the train side against the replicate ceiling of 0.7746 (`results/expression_ceiling_replicate.json`, `primary_ceiling_mean_sqrt_r`, n=82 paired strains, from `expression_ceiling_replicate.py`): traineval at 0.64-0.72 is near the point where additional train fit is fitting measurement noise by construction.

- **`train_eval_every=10` in `long`, not 5**: the eval-mode pass sweeps 4,074 train records against validation's 534 and costs ~65 s, so at every=5 it would add ~13 s to every epoch and ~36 h across 10,000 of them. Every=10 halves that and still yields 1,000 curve points.

- **`wave6` spends its budget on arms rather than replicates**: 12 pair-rank/regularization arms (`V_ref`, `V_sink`, `V_basis16/32/64`, `V_film`, `V_hadamard`, `V_hadamard_add`, `V_drop2/3`, `V_wd1e4/1e2`) at one seed fill 3 runs/GPU x 4 A100 exactly, on `cgt_expr_012`, with the noise floor carried from wave 5's measured across-init sd of ~0.006 (n=8, fixed split).
