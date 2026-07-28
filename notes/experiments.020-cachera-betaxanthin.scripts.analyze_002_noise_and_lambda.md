---
id: 12p2hso23uabn3x0niq3t7j
title: Analyze_002_noise_and_lambda
desc: 'Regenerates the replicate-noise, selection-inflation, lambda-parity and run-length numbers the _003 metabolism sweep design rests on'
updated: 1785209028782
created: 1785209028782
---

## 2026.07.27 - What this script is for

Every design decision in [[plan.metabolism-003-optimizer-distributional.2026.07.27]] is
justified by a measurement rather than a preference, and this is where those measurements come
from. It reads the three `_002` Optuna studies and writes
`experiments/020-cachera-betaxanthin/results/analysis_002_noise_and_lambda.json`.

```bash
python experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda.py
```

It lives in `020` but reads all three arms' study files, because the F = 1 vs F = 19 contrast
is the argument -- neither arm is interpretable alone.

## 2026.07.27 - The four readouts

**1. Replicate noise.** TPE re-proposes points, so `_002` accidentally ran several
configurations twice with identical hyperparameters *and* identical seed 42. The pooled
within-group SD of those repeats is the noise floor every config comparison is made against.
Reported with dof, because the groups are small: betaxanthin and beta-carotene have dof 3,
mulleder19 only dof 1 -- and a chi-square interval on 1 dof spans roughly 0.4x to 30x the point
estimate, so that one is flagged `usable: false` rather than quoted.

**2. Selection inflation.** A sweep's reported best is a *maximum over trials*, biased upward
even when every configuration is equally good. Blom's approximation for `E[max of n standard
normals]` converts the replicate sigma into that bias and gives a bias-corrected floor.

**3. Graph-prior calibration.** `graph_reg_ratio` (graph term / data term) was recorded per
trial, so the map from `graph_reg_lambda` to actual prior strength is measurable *per arm* --
and it is not shared. The relationship is slightly sub-linear (the data term moves too), so the
slope is the median of per-point slopes rather than a fit through the origin.

**4. Run-length confound.** Correlation between trial duration (a proxy for validation-epoch
count) and the ranked objective, computed both overall and within the `hidden = 90` cell so
"bigger model is slower" cannot explain it.

## 2026.07.27 - Results

```text
_002 REPLICATE NOISE  (identical config AND identical seed 42)

betaxanthin  (F = 1)
    trials [3, 10, 11]: [0.4216, 0.4095, 0.3584]  range=0.0632
    trials [19, 31]:    [0.4211, 0.3896]          range=0.0315
    pooled sigma = 0.0302 (dof 3)
    top5 [0.4301, 0.4234, 0.4216, 0.4211, 0.4095] spread=0.0206 = 0.68 sigma -> INDISTINGUISHABLE
    best 0.4301 - selection inflation 0.064 -> floor 0.3662
    replicates for sigma_eff <= 0.010: R = 10

beta_carotene  (F = 1)
    trials [18, 22]: [0.1993, 0.1494]   range=0.0499
    trials [30, 31]: [0.1521, 0.1184]   range=0.0337
    trials [35, 37]: [0.1498, 0.1398]   range=0.01
    pooled sigma = 0.0249 (dof 3)
    top5 [0.2231, 0.1993, 0.1521, 0.1498, 0.1494] spread=0.0736 = 2.96 sigma -> separable
    best 0.2231 - selection inflation 0.0537 -> floor 0.1694
    replicates for sigma_eff <= 0.010: R = 7

mulleder19  (F = 19)
    trials [7, 10]: [0.1798, 0.178]  range=0.0017
    pooled sigma = 0.0012 (dof 1)   <-- dof too low to use on its own

GRAPH-PRIOR CALIBRATION  (ratio = graph term / data term)

betaxanthin:   2.6e-05->0.2414(n6), 6.5e-05->0.4119(n19), 1.3e-04->0.5417(n6)
    slope ~ 6337 * lambda  =>  PARITY at lambda = 1.6e-4
beta_carotene: 2.6e-05->0.0921(n5), 6.5e-05->0.1694(n12), 1.3e-04->0.3794(n6)
    slope ~ 2919 * lambda  =>  PARITY at lambda = 3.4e-4
mulleder19:    2.6e-05->0.2342(n16), 6.5e-05->0.2910(n5), 1.3e-04->0.4832(n7)
    slope ~ 4477 * lambda  =>  PARITY at lambda = 2.2e-4

RUN-LENGTH CONFOUND  (r between trial duration and the ranked objective)

  betaxanthin    r = +0.248   within hidden=90: 0.121 (n=23)
  beta_carotene  r = +0.208   within hidden=90: 0.239 (n=32)
  mulleder19     r = +0.754   within hidden=90: 0.799 (n=23)
```

## 2026.07.27 - Reading the marginal-effects block

The JSON also carries per-axis marginal means, under the key
`marginal_effects_TPE_CONFOUNDED` -- named that way on purpose. `_002` used TPE, which
allocates trials toward the region it currently believes is good, so the level counts are
**unbalanced** and the means are confounded with sampling order. `hp_profile=baseline` posted
the best mean on betaxanthin (0.293) and mulleder19 (0.129) on n = 5 and n = **2**; a level
with n = 2 was not given a fair hearing. Each level therefore carries its `n`, and `_003`
switches to QMC precisely so the next round's marginals do not need this caveat.
