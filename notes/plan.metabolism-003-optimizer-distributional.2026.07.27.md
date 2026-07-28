---
id: 2yecgmklqjrpiuyog11ns71
title: 'Metabolism _003 -- optimizer x distributional sweep'
desc: 'Design + measured justification for the three-arm metabolism sweep on GilaHyper, ported from expression round _007'
updated: 1785209036038
created: 1785209036038
---

## 2026.07.27 - Round _003 design

The metabolism port of expression round `_007` ([[experiments.019-simb-multimodal.fig3-expression-experiments]]).
Three arms, one GPU each on GilaHyper, running concurrently:

| arm | experiment | target | F | ranked on | ceiling |
|---|---|---|---:|---|---:|
| betaxanthin | `020-cachera-betaxanthin` | Cachera 2023 corrected fluorescence | 1 | Pearson | 0.914 |
| beta_carotene | `021-ozaydin-beta-carotene` | Ozaydin 2013 colour score (ordinal) | 1 | **Spearman** | ~0.54 |
| mulleder19 | `022-mulleder-metabolome` | Mulleder 2016 19 amino acids | 19 | Pearson | unmeasured |

Every number below regenerates from
[[experiments.020-cachera-betaxanthin.scripts.analyze_002_noise_and_lambda]]
(`experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda.py` ->
`results/analysis_002_noise_and_lambda.json`).

### The measurement that shaped the round

`_002` used a TPE sampler, which re-proposes points -- so several configurations ran more than
once with **identical hyperparameters and an identical seed** (42 throughout; that sweep never
overrode `cfg.seed`). Those accidental repeats are the only direct measurement of the noise
floor every config comparison is made against:

| arm | repeated trials | values | pooled sigma | dof |
|---|---|---|---:|---:|
| betaxanthin | 3 / 10 / 11 | 0.4216 / 0.4095 / 0.3584 | **0.0302** | 3 |
| betaxanthin | 19 / 31 | 0.4211 / 0.3896 | | |
| beta_carotene | 18 / 22 | 0.1993 / 0.1494 | **0.0249** | 3 |
| mulleder19 | 7 / 10 | 0.1798 / 0.1780 | 0.0012 | 1 (unusable alone) |

The mechanism is structural, not incidental. `pearson_per_feature` is a **mean over features**,
so its noise falls as `1/sqrt(F_eff)`. Expression has F = 6,127 and is denoised by its own
metric; the two scalar arms have F = 1, where the objective is a single correlation over ~473
validation strains. Same metric name, same code path, two different statistical regimes --
which is why `_007`'s single-stage design is right for expression and insufficient here.

Consequences for betaxanthin:

- top five `_002` configs span **0.021 = 0.68 sigma** -> statistically **indistinguishable**
- selection inflation for a best-of-36 is `sigma * E[max_36]` = **0.064**, so the reported
  **0.4301 sits on a bias-corrected floor near 0.366**
- resolving a 0.02 gap needs `sigma_eff <= 0.010`, i.e. **R = 10 replicates**

mulleder19's dof-1 sigma estimate is not usable on its own (a chi-square interval on 1 dof
spans roughly 0.4x-30x the point estimate); plan on ~0.007 from the `1/sqrt(19)` argument.

### What that buys: a two-stage sweep

| stage | what runs | answers |
|---|---|---|
| `screen` | ~60 QMC trials, one seed each | which axes matter |
| `confirm` | top-K screen configs x 5 seeds, ranked on the **mean** | which config is actually best |

`cfg.seed` drives **both** model init and the `CellDataModule` split (`random_seed=seed`), so
the confirm stage is repeated-random-subsampling validation rather than an init re-roll -- the
stronger estimator for "best in expectation over splits", which is the quantity worth selecting
on. R = 5 takes betaxanthin's sigma to 0.013.

The launcher resolves `screen | confirm | done` from the study's own completed-trial counts at
each job start, so the deadline-guarded requeue chain carries an arm through both stages
unattended and stops taking a GPU once confirm is full.

### Inherited from _007 unchanged

- `lr` / `weight_decay` continuous log-uniform, `dropout` categorical -- the `hp_profile`
  bundle **decomposed**. The bundle was as unattributable here as in expression, and TPE
  starved it worse: `baseline` got n = 5 / 4 / **2** across the three arms while posting the
  best mean on two of them.
- Halton QMC (low discrepancy on the float axes; optuna delegates categoricals to a seeded
  random sampler, which is what balances the level counts `_002` lacked).
- Ranking on the peak `_max` of the phenotype-namespaced metric via `DistHead.point()`.
- Calibration recorded `_at_peak`, never ranked.

### What had to change, and the measurement forcing each

- **`energy` / `energy_rank` dropped on the scalar arms.** At F = 1, `Sigma = diag(sigma^2) +
  V V^T` has `V` of shape `[1, k]`, so `V V^T` is a scalar and the energy score degenerates
  into a Monte-Carlo CRPS -- noisier than the closed form already swept. A phantom dimension.
- **`energy_rank` for mulleder19 is a grid {0, 8}, not `_007`'s 32.** A 19x19 covariance has
  rank <= 19, so 32 cannot be realized, and `DistHead` has no guard -- it would allocate
  `V [19, 32]` and waste the excess. `_007`'s 32 was measured (participation-ratio rank 32.8
  of the 6,127-gene expression residual); the equivalent has **not** been measured for 19
  amino acids. Follow-up: re-run `residual_covariance_diagnostic.py` on this target.
- **Per-arm lambda grids.** The same nominal lambda buys very different prior strengths:

  | arm | measured ratio at 6.5e-5 | parity lambda | `_003` grid |
  |---|---:|---:|---|
  | betaxanthin | 0.41 | 1.6e-4 | {0, 1.6e-6, 1.6e-5, 1.6e-4, 1.6e-3} |
  | beta_carotene | 0.17 | 3.4e-4 | {0, 3.4e-6, 3.4e-5, 3.4e-4, 3.4e-3} |
  | mulleder19 | 0.29 | 2.2e-4 | {0, 2.2e-6, 2.2e-5, 2.2e-4, 2.2e-3} |

  A shared lambda is **not** a shared prior. `_002` ran all three at 6.5e-5.

  A 5-epoch smoke run checked the extrapolation directly. mulleder19 landed at
  `ratio_to_data = 0.992` at lambda 2.2e-4 -- parity, exactly as predicted. **beta_carotene
  landed at 8.6**, nine times the prediction, so its parity may sit nearer 4e-5 than 3.4e-4.
  The two disagreeing estimates come from different reductions (`_002` = ratio at last epoch
  of a full run; smoke = `_at_peak` of a 5-epoch run), and the four-decade grid brackets both,
  so the sweep settles it. The arm's *declared* lambda was lowered to 3.4e-5 accordingly --
  that value governs only a manual non-sweep run, where a prior at 8.6x the data loss would
  be training the prior rather than the data.
- **`num_transformer_layers` stays swept {2, 4}.** `_007` could freeze it because `_006` swept
  {2,4,6,8} on expression and found it inert; `_002` measured a large, monotone, **arm-dependent**
  effect (betaxanthin 2 -> 0.288, 4 -> 0.197, 6 -> 0.124, while beta_carotene preferred 4).
- **`graph_reg_depth` not swept** -- with L swept down to 2 (layers 0..1), `_007`'s middle `[2]`
  is out of range half the time and the axis would mean different things in different trials.
- **`node_embeddings` cut to {prot_T5_all, random_1000}**, keeping the width-matched control
  that makes "content, not identity" a measurement.

### Three defects carried over from _002, fixed here

1. **Sub-module dropouts were never bound.** `base.yaml` sets a literal `0.1` on
   `perturbation_head.dropout` and `learnable_embedding.preprocessor.dropout`, and the model
   only falls back to `model.dropout` when the key is **absent**. The `_002` driver overrode
   `model.dropout` alone -- so a trial sampling `dropout = 0.0` still ran those sub-modules at
   0.1. **`_002`'s dropout attribution is partly void.** Same defect class `_007` fixed for
   expression.
2. **`perturbation_head.num_heads` was a literal 6** while expression used 9. Both divide
   hidden = 90, so nothing crashed -- expression and metabolism were quietly running different
   perturbation operators. Now bound to `${model.num_attention_heads}`. Note this **changes
   the model relative to `_002`**.
3. **`min_delta: 0.0`** let any improvement reset early-stopping patience, including a noise
   blip, so a noisy metric kept runs alive on nothing. Now 1e-4, with `max_epochs` 250 -> 200.

### The run-length confound

Ranking on `_max` takes a maximum over however many validation epochs a run survived, and early
stopping makes that length depend on the hyperparameters. Correlation between trial duration and
the ranked objective:

| arm | r(duration, objective) | within hidden=90 |
|---|---:|---:|
| mulleder19 | **+0.754** | **+0.799** (n=23) |
| betaxanthin | +0.248 | +0.121 (n=23) |
| beta_carotene | +0.208 | +0.239 (n=32) |

On mulleder19 that is r^2 = 0.57 of the between-trial variance. Two readings cannot be separated
from this data -- better runs keep improving so early stopping lets them run longer (benign), or
longer runs take more draws at the max (mechanical) -- and either way any axis that changes run
length, **`lr` above all**, gets credit partly through draw count. Three additions make it
auditable rather than silent, all recorded and none ranked:

- `{metric}_smooth3_max` -- max of a 3-epoch rolling mean. A config whose `_max` far exceeds its
  `_smooth3_max` peaked on one lucky epoch rather than on a plateau.
- `val/n_val_epochs`, `val/peak_epoch` -- the covariates. A peak near the end means the run was
  still improving; peaks scattered early with a long tail is the noise-max signature.
- `{phenotype}/pred_sd_ratio` -- `sd(pred)/sd(target)`, the mean-collapse diagnostic **for a
  scalar head**. `pearson_per_instance` correlates across features within a strain and is
  undefined at F = 1, so it came back `None` on two of three arms in `_002`, leaving them with
  no collapse signal at all.

### Launch

```bash
cd <worktree-root>
ARM=betaxanthin   sbatch --job-name=020-bx-003  experiments/020-cachera-betaxanthin/scripts/gh_optuna_metabolism_003.slurm
ARM=beta_carotene sbatch --job-name=021-bc-003  experiments/021-ozaydin-beta-carotene/scripts/gh_optuna_metabolism_003.slurm
ARM=mulleder19    sbatch --job-name=022-m19-003 experiments/022-mulleder-metabolome/scripts/gh_optuna_metabolism_003.slurm
```

`DEADLINE` defaults to `2026-07-30T22:00:00` (a **fixed** date -- a relative one would be
recomputed by every requeue and the chain would never end). Override at submit time if needed.
Trial economics from `_002`: median 35 / 25 / 46 min, so ~60 screen + 8x5 confirm fits the
window on each arm.

### Open, and deliberately not in this round

- **The Merzbacher nested split is built but NOT wired into training.** No config references
  `results/merzbacher_nested_split.json`; `CellDataModule` does an 80/10/10 random split keyed on
  `random_seed`, so roughly 511 of their 639 test genes currently sit in our **train** set. The
  comparison 020 exists for cannot be made from these runs. See
  [[experiments.020-cachera-betaxanthin.merzbacher-comparison]].
- The Cachera build is stale w.r.t. the shared name resolver (issue #195), costing 10 of the 639.
- The flux layer stays deferred -- published k_cat covers 79 of Yeast9's 1,161 genes (6.8%).
  [[plan.cgt-metabolism-flux-layer.2026.07.26]]
