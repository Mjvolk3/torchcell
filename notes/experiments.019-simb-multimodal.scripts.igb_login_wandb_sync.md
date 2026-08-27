---
id: 3ycc1j2xjqglkelnyp1bwbz
title: Igb_login_wandb_sync
desc: ''
updated: 1787839619640
created: 1787839619640
---

## 2026.08.27 - Sync IGB offline runs without tripping the login-node limits

Script: `experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh`

### The problem

IGB compute nodes have no internet, so every mmli/cabbi run writes `offline-run-*` and is
synced later; the login node is the only place that can reach wandb.ai. That makes
`wandb sync` one of the few things legitimately run there. But the documented recipe was

```bash
wandb sync wandb/offline-run-*
```

which hands EVERY run to ONE python process. That is what generates the Biocluster warning:

```
Process: python   CPU%: 1.6   Mem%: 6.9   Limits: %cpu: 15.0  %mem: 5.0
Hostname: biologin-2.igb.illinois.edu
```

Peak memory scales with how many runs go to a single invocation. The activity is allowed;
the shape of the command is not.

### The fix

```bash
bash experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh \
  /home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal
```

- **One `wandb sync` process per run directory**, so peak memory is one run's worth.
- **`nice -n 19`**, so it yields to anything interactive on a shared login node.
- **Skips runs carrying a `.wandb.synced` marker**, so it is resumable and a re-run after an
  interruption is cheap. This is what makes one-at-a-time affordable.
- **Pauses between runs** (`SYNC_PAUSE_S`, default 2), which is what keeps AVERAGE CPU under
  the 15% cap; a tight loop of short processes can average as high as one long one.
- **`SYNC_LIMIT=N`** to cap a pass, **`DRY_RUN=1`** to list what would be synced.
- A single corrupt run is reported and left for the next pass rather than aborting the
  backlog.

### The rule this sits under

Login node = rsync / wandb sync / git / sbatch ONLY. Anything else goes through `sbatch`, or
`srun --pty /bin/bash` for an interactive shell.
<https://help.igb.illinois.edu/Biocluster>

Context: [[experiments.019-simb-multimodal.scripts.optuna_morph_sweep]] ·
[[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
