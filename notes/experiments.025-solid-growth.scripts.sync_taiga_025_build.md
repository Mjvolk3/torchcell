---
id: ezjr9l10hifxxg8tncvtpnd
title: Sync_taiga_025_build
desc: ''
updated: 1788974992972
created: 1788974992972
---

## 2026.09.09 - GilaHyper to Delta Without Duo, by Way of Taiga

Delta needs Duo for every ssh, so a 517 GB rsync from GilaHyper cannot be driven by an
agent. The Zhao-group Taiga share removes the hop: the Radiant VM `torchcell-database`
(`rocky@141.142.216.218`, key auth from GilaHyper) mounts it at `/mnt/zhao5` (NCSA
SUP-29573, configured 2026.09.09), and Delta mounts the same share at
`/taiga/illinois/eng/chbe/zhao5`. GilaHyper writes through Radiant; Delta reads the bytes.

| stage | path |
|---|---|
| source (GilaHyper) | `$DATA_ROOT/data/torchcell/experiments/025-solid-growth/001-full-build/{processed,data_module_cache}` |
| Radiant | `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build/` |
| Delta, same bytes | `/taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build/` |
| Delta, loader path | `/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build` -> symlink to the row above |

The symlink keeps `DATA_ROOT` on `/scratch`, where genome, GO, STRING and TFLink already
are, so no config changes; `delta_cgt.slurm` preflights the 025 path for non-010b configs.

### Three things that cost a retry

- The export is the lowercase path `/taiga/illinois/eng/chbe/zhao5`; the ticket's
  capitalized path gets "No such file or directory" over NFSv4 and a plain `mount` then
  falls to v3 and hangs uninterruptibly. Probe exports by mounting the server's
  pseudo-root read-only: `mount -t nfs -o vers=4.2,ro,retry=0 taiga-nfs...:/ /mnt/x`.
- The NFS server resolves rocky's groups itself, so the local `zhao5_nfs` (gid 555647)
  membership is invisible to it and a plain session is refused at `/mnt/zhao5`. Only a
  process whose primary group is 555647 gets in: `sg zhao5_nfs -c ...`. rsync's remote
  side therefore runs through `/usr/local/bin/rsync-zhao5`
  (`exec sg zhao5_nfs -c "rsync $*"`), installed on the VM.
- Files must be group-readable, because Delta reads them as `mjvolk3` (uid 67392,
  shown there as group `icc_zhao_lab`) while the VM writes them as uid 1000 (shown on
  Delta as `ansible`). `--chmod=ug+rwX --no-g`; the first pass without it landed the
  cache JSONs at mode 600.

### Measured

First full pass 2026.09.09 from GilaHyper: about 111 MB/s over the campus link for the
554,075,586,560 byte LMDB, 1 h 20 m estimated. Delta saw the directory tree appear within
seconds of the transfer starting. Whether LMDB random reads over the Taiga NFS mount from
a Delta compute node are fast enough to train on is not measured; the first Delta job on
this path is the measurement.

Landed 13:47 CDT: 554,075,586,560 B, second pass no transfers, 1 h 19 m end to end.

### Delta launch, 2026.09.09

From the branch worktree `/scratch/bbub/mjvolk3/torchcell.worktrees/025-fitness-joint-head`
(added from the `/projects` clone with `git worktree add`, so the sweep's own checkout at
`/scratch/bbub/mjvolk3/torchcell-sweep` is never touched under its running jobs), after
`delta_preflight_025.sh` reported OK:

```bash
for s in 1 2; do
  sbatch --account=bfjt-delta-gpu --time=48:00:00 -J 025-kl-fit1-s$s \
    experiments/025-solid-growth/scripts/delta_cgt.slurm cgt_s0_r_kl_fit_008 +seed=$s
  sbatch --account=bfjt-delta-gpu --time=48:00:00 -J 025-kl-ctrl-s$s \
    experiments/025-solid-growth/scripts/delta_cgt.slurm cgt_s0_r_kl_000 +seed=$s
done
```

| job | arm | seed |
|---|---|---|
| 21919310 | `cgt_s0_r_kl_fit_008`, fitness weight 1.0 | 1 |
| 21919311 | `cgt_s0_r_kl_000`, control | 1 |
| 21919312 | `cgt_s0_r_kl_fit_008`, fitness weight 1.0 | 2 |
| 21919313 | `cgt_s0_r_kl_000`, control | 2 |

`+seed=` because neither config declares a seed key (the 010b sweep configs do). Together
with IGB's seed-42 set ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]]) this gives
three seeds each of the weight-1.0 arm and the control, and one of weight 0.1. Delta
compute has internet, so these runs log online to the 025 W&B project with tags `seed_1`
/ `seed_2` and `build_025-full`.
