---
id: 79r789egs8tlogtnhqnw400
title: Expression Strand Retrospective
desc: ''
updated: 1788716940040
created: 1788716940040
---

## 2026.09.06 - Expression strand, split off from the six-strand recap

Split off from [[experiments.019-simb-multimodal.phenotype-strand-retrospective]], which
covers all six phenotype strands and stays as it was. This note is strictly the expression
strand (Kemmeren + Sameith, masked-label objective, project `torchcell_019_expr_v9` onward)
and is the running record of every experiment launched after the 2026-08-27 recap. Older
expression history: [[experiments.019-simb-multimodal.expression-round-retrospective]]
(the 2026-07-30 structural diagnosis) and [[experiments.019-simb-multimodal.wave6-design]].

### Where the strand stands

Leaderboard (`experiments/019-simb-multimodal/results/round_leaderboards.csv`, project
`torchcell_019_expr_v9`, `roll_max` = max of a centered 5-epoch mean of
`val/expression/pearson_per_feature`):

| run | dist | seed | epochs | roll_max | note |
|---|---|--:|--:|--:|---|
| `d94cy5az` | quantile | 0 | 18,990 | 0.2430 | resume of `0fymu4py` |
| `0fymu4py` | quantile | 0 | 12,229 | 0.2393 | resume of `hx8pxdic` |
| `hx8pxdic` | quantile | 0 | 9,900 | 0.2382 | fresh run |
| `vqek7ali` | quantile | 1 | 5,328 | 0.2183 | `R_pergene` seed 1 of the mechanism round, not a plain replicate (corrected 2026-09-08, see below) |
| `ia6312dv` | laplace_crps | 1 | 2,039 | 0.2154 | |

The three leaders are one lineage, not three runs, and their spread of 0.005 is inside the
replicate sd. The honest central estimate for the incumbent config (quantile head, lr 3e-4,
dropout 0.1, L=6, hidden 90, mask prior, s1_pool, seed 0, 9,900 epochs) is **0.1965 +/-
0.0222 over 8 identical-config runs**; 0.238 is the max of those 8 draws (memory
`019-expr-long-budget-replicate-spread`). Baselines from `expression_baselines.py`: per-gene
mean 0.0000, bilinear on ProtT5 0.1040, embedding-neighbor 0.0908. The incumbent beats the
bilinear by 0.093, about 4.2 sd. Replicate ceiling for the target is 0.775.
`nmse_at_primary_peak` is above 1.0 for every live run (best 1.0051): no run beats the
per-gene mean on squared error at its Pearson peak.

### Running now

| where | job | what | state at 2026-09-06 12:10 CT |
|---|---|---|---|
| IGB `gpu` | `2371531_2`, `2371531_3` (`019-wave5-igb`, launcher `igb_expr_wave5.slurm`) | quantile seed 1 replicates (`vqek7ali`, `3qy1rh0o`), one run per A40 on compute-0-2 | 3d 16h elapsed, 1d 08h left, epoch ~5,330 at last sync; at ~1,450 epochs/day they hit the 5-day wall near 7,300, short of 9,900 |
| Delta `bflt-delta-gpu` | `21830323_0-7` (`019-expr-v10`) | v10 grid, 16 cells x 2 seeds = 32 runs, 4 per node, 1,400 epochs | all 32 alive, no tracebacks, epochs 150 to 400, val Pearson 0.003 to 0.12 (partial) |

Everything else on IGB has ended: mmli `2369693_0`/`_3` and cabbi `2369697_1` finished
2026-09-05 and their final epochs are synced. Later the same day the metric-aligned round
took 5 cabbi cards (`2378262`, `2378267`, `2378268`, next section).

### Objective round (launched 2026-08-31, IGB jobs 2368333 / 2368337 / 2368339)

Question: is the quantile head load-bearing, or would crps, laplace_crps, or a point head do
as well at a matched budget? Read from the leaderboard on 2026-09-06, long runs only
(>= 1,000 epochs), collapse = final Pearson numerically zero. The `n` counts include the
2026-09-05 resumes of the same runs, so fresh runs per arm are 3, not 6.

| dist | n | collapsed | live mean | live sd | live max | min nmse at peak |
|---|--:|--:|--:|--:|--:|--:|
| point | 6 | 6 | none live | | | |
| crps | 6 | 4 | 0.1697 | 0.0094 | 0.1764 | 1.0754 |
| laplace_crps | 6 | 0 | 0.1947 | 0.0167 | 0.2154 | 1.0247 |
| quantile | 20 | 0 | 0.1952 | 0.0258 | 0.2430 | 1.0051 |

Finding: the point head collapses to the mean every time, crps collapses 4 of 6 times,
laplace_crps never collapses and its live mean sits on the quantile replicate baseline.
Quantile vs laplace_crps is a null at this budget: the gap of 0.0005 is against a detectable
gap of about 0.042 (3 vs 8 replicates at sd 0.0222). The head choice was never load-bearing
between those two; what it buys is not collapsing.

### Replicate spread as a function of budget (`short_budget_spread.py`)

The same 8 identical-config quantile runs, `roll_max` over the prefix at each budget:

| epochs | mean | sd |
|--:|--:|--:|
| 350 | 0.1198 | 0.0096 |
| 500 | 0.1223 | 0.0058 |
| 700 | 0.1411 | 0.0166 |
| 1,000 | 0.1609 | 0.0099 |
| 1,500 | 0.1670 | 0.0117 |
| 2,000 | 0.1745 | 0.0151 |
| 2,800 | 0.1821 | 0.0175 |
| 4,000 | 0.1883 | 0.0171 |
| 6,000 | 0.1917 | 0.0177 |
| 9,900 | 0.1965 | 0.0222 |

The spread is not monotone in budget: 700 is the noisiest point below 2,000. That is why the
v10 grid runs 1,400 epochs with 2 replicates rather than 700 with 4 (main-effect resolution
0.0117 vs 0.0116, at 2.3x the budget). Two traps hit while measuring this: resumes pass an
epoch filter and inherit the config, so they masquerade as replicates (identify a resume by
where its curve starts), and the leaderboard's `is_collapsed` tests the whole-curve max so a
run that collapses late reads as healthy (test the last value).

### Does a short screen rank like a long run? (`budget_rank_preservation.py`)

Spearman between the 700-epoch and 3,000-epoch `roll_max` of the same run: **-0.139
(p = 0.53, n = 23)**, top-3 overlap 1 of 3, top-5 overlap 1 of 5. After resume exclusion the
set is essentially one config, so this measures within-config persistence only: a run's
early rank among its own replicates says nothing about its late rank. Whether a short screen
orders DIFFERENT configs correctly is not answerable on this corpus. It is what the v10 grid
will answer, with 16 configs at 1,400 epochs.

### Loss minimum vs Pearson peak (`loss_min_vs_pearson_peak.py`, 2026-09-06)

Full per-epoch history of 23 live v9 runs. Median epoch of the `val/loss` minimum is 481
(range 203 to 3,097); zero runs bottom out by epoch 100. Median final `val/loss` is 11.4%
above its minimum, and every run's Pearson peak (median epoch 2,488) comes after its loss
minimum, at a loss already 7.1% above the floor. The long budget buys a median +0.054 of
Pearson on a rising validation objective. Detail and figure:
[[experiments.019-simb-multimodal.scripts.loss_min_vs_pearson_peak]].

### v10 grid on Delta (`delta_expr_v10_grid.slurm`, config `cgt_expr_v10_grid.yaml`)

Four two-level factors on the incumbent, everything else pinned: embedding
(random_1024 / prot_T5_all), trunk (L=6 h=90 / L=2 h=45), readout (MLP / linear), weight
decay (1e-8 / 1e-4). 16 cells x 2 seeds, 1,400 epochs, 4 runs per node in parallel, 8 array
tasks. Blocking by the interaction bits `g = (b0 xor b1) + 2 (b2 xor b3)` so every node
carries 2 of 4 on every factor, at the cost of confounding embedding x trunk and
readout x weight-decay with node. Launch history: `21796128` (all 64 failed, shared checkout
had switched branch), `21813317` (log directory inside the vanished checkout), `21814726`
(`KeyError: EXPERIMENT_ROOT`, `.env` is gitignored), `21830323` running. Fixes in commits
`2877aafa`, `a4b77001`, `3b33830c`: isolated clone at `/work/hdd/bbub/mjvolk3/torchcell-v10`,
logs at `/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v10`, environment exported by the job.

### Metric-aligned objective round, SUBMITTED 2026-09-06 (cabbi, 8 runs)

Canary `2375312` (fast-dev-run of both arms in the container) completed clean in 14 min 48 s
with train losses matching the GilaHyper fast-dev-run to five decimals. Submitted at source
`c99c1a9a` (seeds 0-1) and `e16b55ae` (seed 2 and batch 64), all on compute-3-3:

| job | stage | runs | packing |
|---|---|---|---|
| `2378262_0`, `_1` | `pearson` seeds 0, 1 | `Q_pearson`, `Q_pearson_mse` x 2 seeds = 4 | 2 per card |
| `2378267_2` | `pearson` seed 2 | `Q_pearson`, `Q_pearson_mse` = 2 | 2 per card |
| `2378268_0`, `_1` | `pearson_b64` seeds 0, 1 | `Q_pearson_b64` = 2 | 1 per card |

So 3 replicates per arm for the two main arms (detects about 0.042 against the n=8 baseline,
the same power as the objective round) plus a 2-seed batch-64 variant of the pure loss.
The three extra runs were added after the first two tasks had started, when the other
user's cabbi array cleared and three cards freed; `gpu` and mmli were fully allocated at
that moment and were not touched. Canary observation: at initialization the `pearson_mse`
gradient norm is 17.1 (clipped) against 1.2 for `pearson`, so the MSE term dominates the
early gradient at weight 1.0.

Design as prepared:

Question: what happens when the objective stops fighting the metric? The loss-versus-Pearson
result above says every proper-scoring head bottoms out on `val/loss` at a few hundred
epochs and is overfitting by its own measure while Pearson keeps rising. These arms train
DIRECTLY on the metric, so the two cannot disagree on when to stop.

Implementation (`torchcell/losses/distributional.py`, tests in
`tests/torchcell/losses/test_distributional.py`, 86 passing, mypy strict clean):

- `dist: pearson`: loss = 1 minus the mean per-feature Pearson over the genes still hidden
  at the current unmasking step, with each gene's correlation taken over the strains in the
  batch where it is hidden. Columns with fewer than 3 scored rows or a constant prediction
  or target are dropped, as the metric drops them. Point-shaped head, `point()` identity,
  no PIT, so no `calib/*` keys. Scale-free: `mse`, `nmse`, `pred_sd_ratio` of this arm are
  NOT comparable to other arms.
- `dist: pearson_mse`: the same plus the masked MSE at weight 1.0 as a scale anchor.

Design: arms `Q_pearson` and `Q_pearson_mse` in `gh_expr_008_arm.sh`, stage `pearson` in
`igb_expr_wave5.slurm`. 2 arms x seeds {0, 1} = 4 runs, 2 per GPU on the 2 cabbi cards free
at 12:30 CT, `max_epochs` 9,900, 5-day wall, `train_eval_every` 10. Identical budget,
packing and hardware class to the objective round, so the runs are read against the same
n=8 quantile baseline with `roll_max` over epochs <= 9,000. Not touched: mmli (user
decision), the `gpu` partition (fallback only if cabbi fills).

Known confounds, accepted: lr stays 3e-4 (tuned for pinball, not for 1 - r); batch 32 means
each gene's training correlation is estimated over at most 32 strains, fewer at k > 0.
One run per arm per seed cannot resolve a gap under about 0.042.

What the two arms answer. Hypothesis (untested): `pearson` will reach its validation Pearson
peak earlier than quantile and its `val/loss` minimum and Pearson peak will coincide by
construction on train, so a persisting gap between them on validation is pure
generalization gap. If `pearson_mse` matches `pearson` on Pearson while keeping nmse near
quantile's, the scale anchor is free. If `pearson` collapses (all columns constant, loss
stuck at the connected zero), that is a measured failure mode and the reason `pearson_mse`
exists.

### Open decisions

- Which metric is the target. Stopping on `val/loss` says 300 to 700 epochs and the
  long-budget campaign is over; stopping on Pearson says the curves have not turned at
  19,000. The paper reports Pearson; the objective is the pinball loss.
- Whether the two IGB seed-1 runs are worth their remaining day; they will not reach 9,900.
- What, if anything, goes on the 16 idle IGB GPUs before the v10 grid reads out (about
  2026-09-08 at 33.6 epochs/h).

## 2026.09.08 - Three readouts: the v10 grid, the mechanism round, the Pearson round at day two

Delta's queue is empty and IGB holds only the five Pearson-round tasks. Everything below
was synced from the IGB login node (`igb_login_wandb_sync.sh`, 12 runs, 2026-09-08 16:40 CT)
or read from the online Delta project.

### Corrections to the 2026-09-06 record

- `vqek7ali` and `3qy1rh0o` were described above as "quantile seed 1 replicates". They are
  `R_pergene` seed 1 and `R_pergene_basis64` seed 1 of the mechanism round; the leaderboard's
  `dist` column reads `quantile` for every mechanism arm because all of them use the
  quantile head. The IGB `gpu` jobs `2371531_2/_3` were the seed-1 half of that round, not
  extra replicates of the incumbent.
- Every `R_ref` run of the mechanism round is tagged `stage-wave4b`, not `stage-mech`: the
  arm script's `case` had an older `R_ref` branch first and took it. Fixed in
  `gh_expr_008_arm.sh`; the readout selects by arm plus config tag.

### Every packed five-day task on IGB has died of host memory

`sacct`: `2369697_0/_1` (mechanism seed 0, cabbi) and `2371531_2/_3` (seed 1, A40) all
ended `OUT_OF_MEMORY` with `MaxRSS` 61.4 and 61.1 GB against `ReqMem` 60 GB, after 3.5, 4.2,
4.5 and 4.6 days. In each task the cgroup handler killed one of the two packed runs and the
other finished its 8,500 epochs. The Pearson-round packed tasks `2378262_0/_1` and
`2378267_2` read 21.3 GB `MaxRSS` at 1.9 days (`sstat`), the solo batch-64 tasks 17.2 GB.
Hypothesis (untested): host RSS grows roughly linearly through a run, so the packed
Pearson tasks reach the 60 GB line near day five, on the edge of their wall. Memory of a
running job cannot be raised; nothing to do but watch. Whatever grows has not been
identified; the `file_system` sharing-strategy change of `f1bfa95b` is the obvious suspect
and is unverified.

### The v10 grid on Delta: embedding content is the only large factor

`21830323` finished: five tasks completed 1,400 epochs, three timed out at the two-day wall
between epochs 990 and 1,198. Readout in
[[experiments.019-simb-multimodal.scripts.v10_grid_factorial]] at the matched budget of
epochs <= 990, `roll_max`, 32 runs, pooled within-cell replicate sd 0.0246 (16 df):

| factor | level 0 -> level 1 | effect | t |
|---|---|--:|--:|
| embedding | random_1024 -> prot_T5_all | +0.068 | +7.8 |
| trunk | L=6 h=90 -> L=2 h=45 | -0.012 | -1.4 |
| readout | MLP -> linear | +0.007 | +0.8 |
| weight decay | 1e-8 -> 1e-4 | +0.005 | +0.5 |

Embedding x trunk +0.017 (t +2.0); every other interaction |t| < 1.3. One run
(`te8272kk`, cell 1 seed 0) never left the chance band (0.019) while its seed-1 twin is the
grid's best run (0.169 full-run); without it the pooled sd is 0.0122 and the trunk effect
firms to -0.018 (t -4.0). Readout width and weight decay are measured nulls at about 0.02.
The best cell by mean is 13 (prot_T5, L=6, linear, wd 1e-4) at 0.143; the incumbent's eight
replicates score 0.161 +/- 0.010 at 1,000 epochs with `calm` embeddings, which is not a
level of the grid (the factor was built as a content contrast at matched width 1024). So
the grid does not rank calm against ProtT5; the one healthy draw of the incumbent-with-
ProtT5 cell at 0.141 is not a resolved gap against 0.161.

### The mechanism round: per-gene readout +0.03, below the round's resolution

Readout in [[experiments.019-simb-multimodal.scripts.mech_round_readout]] at the matched
budget of epochs <= 4,079 (set by the killed `R_pergene_basis64` seed 0):

| arm | seed 0 | seed 1 | paired diff vs `R_ref` (s0 / s1) | mean |
|---|--:|--:|--:|--:|
| `R_ref` | 0.1881 | 0.1693 | | |
| `R_basis64` | 0.1746 | 0.1996 | -0.014 / +0.030 | +0.008 |
| `R_pergene` | 0.1888 | 0.2237 | +0.001 / +0.054 | +0.028 |
| `R_pergene_basis64` | 0.2136 | 0.1985 | +0.026 / +0.029 | +0.027 |

Both `R_ref` draws sit inside the incumbent band at 4,000 epochs (0.1883 +/- 0.0171). The
pair term alone is a null with sign-disagreeing pairs. The per-gene readout arms average
+0.027, the combined arm positive at both seeds, against a design resolution of about 0.06:
a direction, not a result. Visible in every curve: the per-gene arms peak early (epochs
1,200 to 2,600) and give part of it back by 8,500 (`R_pergene` ends at 0.137 and 0.170
against peaks of 0.189 and 0.224), while the reference and basis arms are flat or still
rising.

### Scripts

- [[experiments.019-simb-multimodal.scripts.v10_grid_factorial]]
- [[experiments.019-simb-multimodal.scripts.mech_round_readout]]
- [[experiments.019-simb-multimodal.scripts.pearson_round_readout]]
- [[experiments.019-simb-multimodal.scripts.pull_round_leaderboards]]
- [[experiments.019-simb-multimodal.scripts.short_budget_spread]]
- [[experiments.019-simb-multimodal.scripts.budget_rank_preservation]]
- [[experiments.019-simb-multimodal.scripts.loss_min_vs_pearson_peak]]
- [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]]
- [[experiments.019-simb-multimodal.scripts.igb_login_wandb_sync]]
