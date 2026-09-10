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

## 2026.09.08 - Typeset counterpart: notes-tex/019-simb-multimodal-expression

This note now has its own notes-tex document, `notes-tex/019-simb-multimodal-expression/`
(build with `make`, gate with `make check`, figures via `make plots`). It carries the
consolidated state of the strand claim by claim (section 1), the readouts of 2026-09-08
(section 2, moved out of the six-strand retrospective's former section 11 so that document
stays as reviewed), and the open decisions (section 3). The design and launch sections
(9 and 10) stay in `notes-tex/019-simb-multimodal/`, since sections 1 to 8 there reference
them thirteen times and both review rounds covered them. Figures used by the document are
written under stable names: `v10_grid_factorial`, `mech_round_readout`,
`pearson_round_readout`, `loss_min_vs_pearson_peak` (the last renamed from its timestamped
form today; its numbers are unchanged, and the Pearson-objective arms are now excluded from
that audit by construction).

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

### The Pearson round at day two: collapse is the rule at batch 32

Readout in [[experiments.019-simb-multimodal.scripts.pearson_round_readout]], runs at 1.9 of
5 days, every number partial. Of six packed batch-32 runs, five are on the floor: all three
`Q_pearson` (peaks of 0.13 to 0.145 at epochs 150 to 240, then predicted spread falls to
1e-7 and the metric to 0 by epochs 584 to 680, seed 1 flickering until 2,203) and two of
three `Q_pearson_mse` (seed 2 by epoch 14, seed 0 by 360). The loss does not see it: it
drops constant columns as invalid and keeps reading about 0.56 on whatever still varies.
`Q_pearson_mse` seed 1 is alive at 0.194 and rising. The two solo batch-64 `Q_pearson` runs
are alive at 6,050 epochs, `roll_max` 0.194 and 0.186 (peaks near 1,900 to 2,400, drifting
down since), on the incumbent band of 0.1917 +/- 0.0177 at 6,000 and not above it. Batch
size and solo packing are confounded in the b64 arm, so why it survives is not measured.
Nothing in this round beats the quantile head at any budget read so far.

### Open decisions, 2026-09-08

- The Pearson round: let it run to the wall (the b64 runs and `pearson_mse` seed 1 are the
  only informative survivors) or free the three cabbi cards holding collapsed runs now. The
  packed tasks cannot be partially cancelled without killing their live sibling.
- Whether the packed-run memory growth is worth chasing before any further five-day packed
  submission; the alternative is solo runs or a 48 GB-per-run budget.
- What the v10 result changes: ProtT5 content is worth 0.07 over random at 1,000 epochs, and
  calm against ProtT5 is unmeasured at any budget. A two-arm calm-vs-ProtT5 replicate set at
  the incumbent config is the cheapest next contrast.

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

## 2026.09.08 - Correction: the eight "identical-config replicates" are the eight v9 mask-schedule arms

Found while indexing every run for the W&B table. The eight long-budget runs that
`short_budget_spread.py` treats as replicates (`8r5ewoaq`, `da5g4o9v`, `ebkzn1ao`,
`f2wf23oy`, `hx8pxdic`, `rb3bhryq`, `tow1z48n`, `u1vuznme`) share every leaderboard config
column (head, lr, dropout, L, hidden, prior, decoder, seed 0) and differ in what the
leaderboard does not carry. Their W&B configs, read 2026-09-08:

| run | arm | mask_schedule | mixing | gate | roll_max |
|---|---|---|---|---|--:|
| `hx8pxdic` | `M_fine` | [0,10,30,100,300,1000] | on | on | 0.2382 |
| `ebkzn1ao` | `M_coarse` | [0,100,1000] | on | on | 0.2091 |
| `rb3bhryq` | `M_nomix` | [0,10,100,1000] | off | on | 0.2057 |
| `tow1z48n` | `M_off` | none | on | on | 0.2008 |
| `da5g4o9v` | `M_hi` | [0,1000,3000] | on | on | 0.1887 |
| `8r5ewoaq` | `M_sched` | [0,10,100,1000] | on | on | 0.1824 |
| `f2wf23oy` | `M_lo` | [0,5,10,30] | on | on | 0.1804 |
| `u1vuznme` | `M_gate_rezero` | [0,10,100,1000] | on | rezero | 0.1663 |

Consequences, stated plainly:

- 0.1965 +/- 0.0222 is the mean and spread ACROSS the eight mask-schedule arms, not a
  replicate estimate of the incumbent. The incumbent schedule proper (`M_sched`) is one
  draw at 0.1824; the 0.2382 headline is the `M_fine` arm, one draw of one schedule.
- The spread-by-budget table, the power arithmetic (detects ~0.042 at 3 vs 8) and every
  "incumbent band" in the readout figures are arm spread plus nondeterminism. They bound
  the replicate spread from above, so the resolution claims are conservative, not wrong.
- Replicate spread measured on runs that DO share a config: v10 pooled within-cell sd
  0.0246 at epochs <= 990 (0.0122 without the stuck run); mechanism-round `R_ref` seeds
  differ by 0.019 at epochs <= 4,079; objective-round `crps` live sd 0.0094.
- Within-round contrasts (v10 main effects, mechanism paired differences) are unaffected.
- The rank-preservation Spearman (n = 23, "essentially one config") also spans these arms.
- Interesting in its own right, and unmeasured until now: `M_off` (no masked objective)
  scores 0.2008, inside the spread of the schedules, so at k = 0 scoring the masked-label
  objective has not been shown to help. One draw per arm; a hypothesis until replicated.

Fixed today: figure labels and docstrings of the three readout scripts, the spread
script, both notes-tex documents (the SIMB launch section carries a dated correction
paragraph; the expression document is rewritten where it said replicates), and the memory
record. The `short_budget_spread.py` config assertion now names the columns it can see
and says what it cannot.

## 2026.09.08 - Pearson round pruned; ListMLE ranking objective prepared

### Pruned at day 2.1

From the 19:01 CT sync, five of eight Pearson-round runs were constant-output (validation
Pearson 0, predicted spread 1e-7). `2378262_0` (`Q_pearson` s0 + `Q_pearson_mse` s0) and
`2378267_2` (both seed-2 runs) held only dead runs and were cancelled at 19:30 CT; two cabbi
cards freed. `2378262_1` holds the live `Q_pearson_mse` seed 1 beside the dead `Q_pearson`
seed 1; the dead process (PID 2893629, identified by its `Q_pearson,seed1` tag) was sent
SIGTERM through an overlapping step and the sibling kept training on the whole card. The
two solo `Q_pearson_b64` tasks are untouched. Left running: `2378262_1`, `2378268_0`,
`2378268_1`.

### ListMLE: rank the strains per gene

Question: given a set of knockouts, which strain expresses gene g highest? That is a
ranking across strains per gene, the quantity per-feature Spearman scores, and it is what
the Pearson loss also targets but through a scale-free correlation that drops constant
columns. ListMLE (Plackett-Luce likelihood of the true ordering of the strains in the
batch, per hidden gene, Xia et al. 2008) keeps a gradient on a constant column and is
shift- but not scale-invariant (the loss keeps falling as score gaps grow), so the pure arm
is expected to drift in output scale and the anchored arm pins it. The list is the batch,
so batch size is part of the objective, as for Pearson.

Implementation (commit pending): `listmle` and `listmle_mse` in
`torchcell/losses/distributional.py` (`masked_per_feature_listmle`, `listmle_loss`,
`DEFAULT_LISTMLE_MSE_WEIGHT = 1.0`, `LISTMLE_MIN_ROWS = 2`; masked rows sorted last with a
large-negative score so they leave every suffix logsumexp), point-shaped, no PIT. Tests:
98 passing, mypy strict clean. Arms `Q_listmle`, `Q_listmle_mse` in `gh_expr_008_arm.sh`;
stages `listmle` (2 arms x seeds 0-1, 2 per card, `max_epochs` 6,000) and `canary_listmle`
in `igb_expr_wave5.slurm`. 6,000 rather than 9,900 because every packed five-day task so
far died of host memory between days 3.5 and 4.6, and the Pearson peaks came by 3,200.

Cards at 19:35 CT: cabbi 2 free of 8 (3 mine, 3 another user), `gpu` 6 A40 idle on three
nodes, mmli 4 A100 idle (not to be touched). Submission waits for approval.

### ListMLE round, SUBMITTED 2026-09-08 20:05 CT

Approved with the batch-64 arm added, two seeds. Container canary `2385788` completed clean
in 9 min 23 s with first-epoch numbers matching the GilaHyper fast-dev-run to five
decimals. Source `39703f5c`, clean diff.

| job | partition | stage | runs | packing | max_epochs |
|---|---|---|---|---|--:|
| `2385807_0`, `_1` | cabbi (compute-3-3) | `listmle` seeds 0, 1 | `Q_listmle`, `Q_listmle_mse` x 2 = 4 | 2 per card | 6,000 |
| `2385808_0`, `_1` | gpu (compute-0-0, A40) | `listmle_b64` seeds 0, 1 | `Q_listmle_b64` x 2 | 1 per card | 6,000 |

Six runs. Scored by `roll_max` and by per-feature Spearman (the quantity the objective
targets) at matched budgets against the Pearson round and the v9 long-budget arms. All
four tasks were RUNNING 20 s after submission.

### ListMLE round at 7.5 hours (2026-09-09 03:35 CT)

| run | epoch | val Pearson | val Spearman | pred/true spread | val loss |
|---|--:|--:|--:|--:|--:|
| `Q_listmle_b64` seed 0 | 1,000 | 0.095 | 0.088 | 0.42 | 3.002 |
| `Q_listmle_b64` seed 1 | 1,000 | 0.082 | 0.085 | 0.47 | 3.015 |
| `Q_listmle` seed 0 | 636 | 0.091 | 0.099 | 0.43 | 2.444 |
| `Q_listmle` seed 1 | 480 | 0.077 | 0.077 | 0.30 | 2.437 |
| `Q_listmle_mse` seed 0 | 636 | 0.000 | 0.000 | 0.00000 | 3.529 |
| `Q_listmle_mse` seed 1 | 480 | -0.033 | -0.032 | 0.00002 | 3.520 |

The anchored arm sat at the mean predictor from epoch 160 on at both seeds (zero spread,
loss flat at 3.52; the MSE gradient is 50x the ranking gradient at initialization and
finds its own optimum). Both `Q_listmle_mse` processes were killed by PID at 03:40 CT, the
same way as the dead `Q_pearson` seed 1; their pure siblings continue on whole cards. The
pure arms learn, slowly: the quantile head read 0.12 at epoch 350 and 0.16 at 600, so
ListMLE is behind by about 0.07 at matched epochs, with predicted spread growing from 0.05
to 0.47 as the loss rewards larger gaps. Hypothesis (untested): the ranking gradient is
small in absolute terms (norm 0.3 at initialization) and the pinned lr 3e-4 is low for it.

Surviving Pearson runs at the same check: `Q_pearson_mse` seed 1 at epoch 5,273 reads
0.176 and is still rising (0.159 at 4,077); `Q_pearson_b64` at 7,700 read 0.157 and 0.149,
past their peaks of 0.194 and 0.186.

## 2026.09.09 - The expression document made self-contained; ListMLE at day one; W&B "finished" is not "ended"

### Correction

An earlier read today called all eight Pearson-round runs finished because W&B showed
state `finished`. They are offline runs synced from the IGB login node, and every sync
stamps the snapshot `finished`; `squeue` at 18:10 CT showed `2378262_1`, `2378268_0`,
`2378268_1` and all four ListMLE tasks RUNNING. The batch-64 pair was at 9,114 and 9,118
of 9,900 (about six hours from the end), the anchored seed-1 run at 6,576 (about 1.3
days), ListMLE at 1,792 to 2,438 of 6,000. All eight cabbi cards are allocated (three to
another user), four A40s on `gpu` idle. The document and this note now say "running, at
epoch N" wherever they said "finished".

### The document, retitled and self-contained

`notes-tex/019-simb-multimodal-expression/` is now "Knockout Expression: the
genotype-to-expression strand". A new section 0 carries the expression material of the
SIMB retrospective over so nothing there is needed: the task and ceiling (0.7746 from 82
shared deletions), the model and the one place strain identity enters it, the pair-term
degeneracy and rank ladder, the distributional-head table the incumbent rested on, the
decoder-family arms of v8 waves 1 to 3 (new generated table, `decoder_arms_table.py`),
the four baselines, the GEARS/State/CPA operators and the Ahlmann-Eltze benchmark, the
imputation oracle, and the campaign arithmetic. Every number keeps its result-file
source. The SIMB document is untouched.

Answer to "did we do the heads comparison, GEARS etc.": the distributional-head
comparison is the objective round (done, heads indistinguishable). The decoder-family
comparison (GEARS-style cross-gene readout `GEARS_crossgene`, `D1_bilinear32`,
`E0_perceiver32`, `H0_factor`, `C0_concat`, propagation, null sink) was run in v8 waves
1 to 3, 35 runs, every one stopped between epochs 50 and 276, one seed for most, so it
was never a comparison. Of those families only the response basis and the per-gene
readout have since been trained at a resolving budget (mechanism round). GEARS-style
cross-gene, bilinear, Perceiver alone and FiLM have not.

### CLS is wild-type for every strain

Confirmed in the model file: the encoder runs once on the unperturbed graph per batch;
`h_CLS` has measured across-strain sd 0.0 against 0.973 for `z_S` (code comment at the
PerGeneHead construction), which is why the FiLM conditioner was rewired to `z_S` alone.
The perturbation never enters the graph-masked attention. Three untested designs are in
the document's next section: perturb CLS with the existing operator (cheap, no pair
term), edit the gene token before the encoder (one encoder pass per strain), or typed
perturbation tokens entering the encoder (gene-edit token attends to its gene, an
environment token, e.g. an antifungal, attends to CLS or all genes), which is the only
one of the three that covers environmental perturbations.

### ListMLE round at day one (18:15 CT)

| run | epoch | roll_max @ epoch | Spearman roll_max | last | spread ratio |
|---|--:|--:|--:|--:|--:|
| `Q_listmle` seed 0 | 1,959 | 0.168 @ 1,920 | 0.173 | 0.167 | 0.80 |
| `Q_listmle` seed 1 | 1,792 | 0.150 @ 1,488 | 0.154 | 0.140 | 0.75 |
| `Q_listmle_b64` seed 0 | 2,438 | 0.144 @ 2,433 | 0.134 | 0.137 | 0.67 |
| `Q_listmle_b64` seed 1 | 2,437 | 0.150 @ 2,057 | 0.138 | 0.144 | 0.67 |

Long-budget arms: 0.1745 +/- 0.0151 at 2,000, 0.1821 +/- 0.0175 at 2,800. So one seed
inside the band, three below, every curve still rising, none peaked. No collapse in any
pure run; spread ratio climbing toward 1 as a shift-invariant objective should make it;
leaderboard `nmse` 74.7 is the free per-gene location, not a fit failure. On Spearman the
batch-32 seed 0 reads 0.173 at 1,920 where the Pearson-objective survivors read 0.171 to
0.178 at 6,500 to 9,100. Hypothesis (untested): lr 3e-4 inherited from the pinball loss
is low for the ranking gradient (norm 0.3 at init).

### What fits the hardware (proposal, for discussion)

Delta `bbub` has 3,738 GPU-hours at a 48-hour wall; the v10 grid reached 990 to 1,400
epochs per 48-hour task, so 1,000-epoch designs fit and 9,900-epoch ones do not. (a)
`calm` vs `prot_T5_all` at the incumbent config, 3 seeds each, 1,000 epochs: 6 runs,
about 300 GPU-hours, resolves about 0.03 and gives the first true replicate spread of the
incumbent. (b) `R_ref` vs `R_pergene` at 1,000 epochs, 3 pairs: 6 runs, resolves 0.03 at
the 1,000-epoch sd of 0.0099. (c) one batch-32 solo pure-Pearson seed, 9,900 epochs, on
cabbi when the batch-64 tasks free their cards (about 6 h). (d) ListMLE learning-rate arm
after the round reads out. The encoder-side perturbation designs need implementation
first.

Scripts: `pearson_round_readout.py --round listmle` (new mode; Spearman columns;
`wandb_state`), `decoder_arms_table.py` (new), `wandb_run_index.py` (ranking round; 80
runs), `loss_min_vs_pearson_peak.py` (58 mm, labels no longer clipped). Leaderboard v9
refreshed with `--full-history --refresh`. Document builds to 17 pages, `make check`
clean.

### Per-gene readout replicates, SUBMITTED 2026-09-09 22:55 CT

Approved without a canary. Job `2389901_0`, `_1` on cabbi (compute-3-3), stage
`pergene_rep`: `RR_ref` vs `RR_pergene` (same configs as `R_ref` / `R_pergene`, tags
`stage-pergene-rep`, `round-pergene-rep`), seeds 2 and 3, 1,000 epochs, one pair per card
(2 runs per card, `train_eval_every=10`), 30-hour wall. Source `4db2fbc8`, clean diff
(`e3b0c442`), IGB checkout fast-forwarded to the pushed branch before submission. Task 2
(seed 4) waits for the third cabbi card, freed when `2378262_1` reaches 9,900 (about a
day). Readout: three new pairs plus the two mechanism-round pairs at epochs <= 1,000,
resolving about 0.03 at the 1,000-epoch spread of 0.0099. Both tasks RUNNING 20 s after
submission.

Batch-64 Pearson tasks `2378268_0`, `_1` COMPLETED at 3 d 01 h 45 m; final readout after
the sync. Embedding arm (calm vs ProtT5, or a stack) held until a CPU probe of stacked
embeddings (ProtT5 + calm / codon frequency / chrom pathways, plus a reporter-side promoter
probe) says which contrast is worth cards; the neighbor probe puts NT and species-LM
promoter/terminator embeddings at the random floor on the deletion side (0.005 to 0.036
against a floor of 0.011 to 0.033).

## 2026.09.10 - Input-richness round SUBMITTED on Delta (v11)

Job `21948711_[0-2]` on Delta `gpuA40x4`, account `bbub-delta-gpu`, submitted 13:39 CT,
PENDING (Resources) behind the bfjt 025 jobs. Project `torchcell_019_expr_v11`, config
`cgt_expr_v11_emb.yaml` (v9 incumbent through v9_mask, `max_epochs` 1,400,
`train_eval_every` 10), launcher `delta_expr_v11_emb.slurm`, job-owned shallow clone
`/work/hdd/bbub/mjvolk3/torchcell-v11` at `bc0b6497`, clean diff, hash and diff sha
exported into every run. Four arms x seeds 0-2, one run per A40, one seed per node:

| arm | `cell_dataset.node_embeddings` | input width | preprocessor params |
|---|---|---|--:|
| E_calm | [calm] | 768 | 369,639 |
| E_ptt5 | [prot_T5_all] | 1,024 | |
| E_calm_ptt5 | [calm, prot_T5_all] | 1,792 | |
| E_full | [fudt_upstream, calm, prot_T5_all, fudt_downstream] | 3,328 | 5,846,759 |

Species-aware fungal LM (SpeciesLM, registry `fudt_*`) for the cis windows rather than
the nucleotide transformer: yeast-trained, 768-d per region against NT's 2,560, neighbor
probe 0.036 / 0.020 against NT's 0.005 to 0.012. Both files cover 6,607 genes (the 3'
rebuild of 2026-07-27 restored the 28 Q0 mitochondrial genes; verified by loading the
.pt) and Delta's copies match GilaHyper's sha256 (`ae196f34`, `5ee64676`). E_full's
block order is locus order and is legibility only; the blocks feed one linear layer.
Width confound named in the config: the projection's hidden layer is (input + 90) / 2,
so E_full carries 16x the incumbent's preprocessor parameters (total 6.56M vs 1.19M); a
width-matched random filler arm is the control if E_full leads. No Delta canary: the
E_full list was proven by a CPU `fast_dev_run` on GilaHyper (exit 0, one train batch plus
validation) in the submitted order. `sbatch --test-only` accepted the array. About 576
GPU-hours of bbub's 3,738. Logs: `/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v11/`.

Readout plan: `roll_max` at a matched budget (smallest final epoch across the 12) with
paired within-node contrasts, E_ptt5 - E_calm (the contrast v10 could not make),
E_calm_ptt5 - E_ptt5, E_full - E_calm_ptt5; three seeds resolve about 0.03.
