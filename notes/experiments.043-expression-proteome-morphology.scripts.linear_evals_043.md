---
id: 1f6pzsobboq2oi4cq87v4qx
title: Linear_evals_043
desc: ''
updated: 1792634290131
created: 1792634290131
---

## 2026.10.21 - The linear ladder, first run

`experiments/043-expression-proteome-morphology/scripts/linear_evals_043.py`, run stage by stage
through `scripts/gh_linear_evals_043.slurm` (`STAGE`, `DATASET`; CPU only, 16 CPUs, 48 GB).
Reads `conf/splits/global_split_v1.json` and refuses a store whose raw-index hash differs.

- `extract` (job 6051): every needed record by raw index through `Raw043Dataset` in a 16-process
  pool, cached as one pickle per dataset under
  `$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/linear_evals_cache/`,
  plus the Costanzo 2016 single-mutant fitness (`smf_costanzo2016` dev store). Boocock de novo
  records are skipped by a byte check on the medium before parsing.
- `deletion --dataset {kemmeren, messner, zelezniak, hu, nadal}`: N0, multi-output kernel ridge
  (penalty `trace(K)/n * 10^g`, g in -3 to 3 by halves, chosen on fold `(k+1) mod 10` by the
  median per-gene Pearson, refit on every non-test fold), kNN k 5, 10, 20 by cosine, each on the
  composite and on each of the four LayerNormed blocks. Nadal-Ribelles is scored per
  environment; targets keep columns finite in at least half the units.
- `doubles`, `isolates`, `segregants`, `vaishnav`, `morphology`: the rungs of the primer, as
  described in the module docstring.
- `report`: `results/linear_evals_043.json`, `results/linear_evals_043_split_a_summary.csv`,
  `results/linear_evals_043_isolates_segregants_summary.csv`, the per-part JSON and CSVs under
  `results/linear_evals_043/`, and four figures.

Conventions worth knowing when reading the numbers: a constant prediction scores 0 (N0 per gene
and after centering), so a paired interval against N0 is the model's own interval; the bootstrap
runs on fold 0 for ridge and kNN 10 (kNN 5 and 20 carry point values only); missing training
targets are set to the training mean before the closed-form solve.

Results and job ids: [[2026.10.21 - Global split and linear evaluations|dendron://torchcell/experiments.043-expression-proteome-morphology#20261021---global-split-and-linear-evaluations]].
