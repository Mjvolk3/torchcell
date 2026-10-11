---
id: e3mi3iczc542yk67u5rcj1y
title: R3_sweep
desc: ''
updated: 1791711454249
created: 1791711454249
---

## 2026.10.11 - Round 3 as two waves: the screen, then the cells that need no reading

`r3_sweep.py` writes round 3 of the mixture trainer
([[experiments.040-inhibitor-synergy-wetlab.scripts.train_mixture]]) as sweep files, one
Delta submission script and one plan table per wave. One config is one fold of one arm;
one job is one GPU, one process and the folds of one arm that fit under the target of
10.5 A40 hours, and every job asks the 12 hour partition limit rather than its estimate.
The estimates come from the round-1 and round-2 runtimes scaled by epochs, sources and
seeds, and they are of a regime that had not converged, which is why the limit is asked.

**Wave 1 (`r3_*`, 32 jobs, 25 arms, one seed, folds 0 to 2)** is the screen written up
in the experiment note: the five source sets at 150 epochs, 300 on the heaviest, the
round-1 length as a bridge, three source encodings, two auxiliary weightings, three dose
arms, three gene-cold arms, three capacity arms, and the two headline arms on fold seeds 1
and 2. Its recipe is 038 round 10's with `fit_on: train` and `select: val`, so every arm
selects on a held-out signal.

**Wave 2 (`r3w2_*`, 26 jobs, 60 configs, `--wave 2`)** is submitted while wave 1 is
still queued, and holds only cells worth running whatever wave 1 says:

| block | arms | jobs | what it settles |
|---|---|---|---|
| ensemble | `van_e150`, `vanhop_e150`, `vanhophet_e150`: seeds 1 and 2 on folds 0 to 2 (`_s12`), seeds 0 to 2 on folds 3 and 4 | 11 | the three source-set arms at five folds and three seeds, which the data-combining comparison needs whichever way it falls |
| partitions | `vanhophet_e150_fs1`, `_fs2` | 2 | the three-source arm on the other two fold seeds (wave 1 repeats only the one- and two-source arms there) |
| host head | `van_e150_anchors`, `_ft`, `_ftenv`, `_lh0p1`, `vanhophet_e150_ftenv` | 5 | round 1's host-head arms, whose wet-lab readouts are the panel, re-scored on a model selected on a held-out signal |
| dose | `van_e150_film`, `vanhophet_e150_film` | 2 | the single-source dose ablation under the new protocol, and the compound hold-out twin of `vanhophet_e150_film_dose` |
| learning curve | `{van, vanhop, vanhophet}_e150_n{5, 10}` | 6 | the compound-count curve under the held-out protocol, with and without the public sources; the `e150` arms are the full-train point (18 or 19 compounds) |

The seed-only configs carry `_s12` between the arm and the fold (`mix_van_e150_s12_f0`)
so their results directory and W&B run never collide with wave 1's, while
`wandb_view.arm_of` strips the suffix and groups them under the same arm. The learning
curve needed one trainer change, `n_fit_compounds` under `fit_on: train`, recorded in the
trainer's note; the second wave's submission script says so in its header, because the
Delta tree has to be re-rsynced before it runs.

Both waves were submitted on `bfjt-delta-gpu` on 2026-10-11 at 04:56 after the tree
rsync: wave 1 as Delta jobs 22806796 to 22806827, wave 2 as 22806828 to 22806853.

What waits for a wave-1 reading is round 4: the ensemble on whichever source encoding,
auxiliary weighting, capacity and gene-cold arms move, and the fold-seed replication of
those.

Outputs: `conf/mixture/r3_*.yaml`, `conf/mixture/r3w2_*.yaml`,
`conf/mixture/r3_submit_delta.sh`, `conf/mixture/r3w2_submit_delta.sh`,
`results/r3_plan.csv`, `results/r3w2_plan.csv`. The 111 `r3_*_p*.yaml` files of an
earlier packing (five folds per arm) that the plan never referenced were removed the same
day; the plan table is the list of what was submitted.
