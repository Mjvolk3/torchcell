---
id: vjpljm8e7dr07logz1rzysx
title: Fba_screen_medium
desc: ''
updated: 1789066928698
created: 1789066928698
---

Rerun of the Yeast9 FBA baseline (Fig. 2d "Yeast9 FBA", [[experiments.007-kuzmin-tm.scripts.fba_baseline_si]]) on the medium the trigenic screens were scored on. Script: `experiments/007-kuzmin-tm/scripts/fba_screen_medium.py`, launcher `gh_fba_screen_medium.slurm` (gilahyper, CPU only). Scoring and panel g: [[experiments.007-kuzmin-tm.scripts.fba_screen_medium_si]]. Medium recipes: [[torchcell.metabolism.media]].

## 2026.09.10 - Design: three arms on the frozen run's deletion sets

The frozen baseline (`results/cobra-fba-growth_backup_20250923_134447/`, 2025-09-15) solved every deletion set on yeast-GEM 9.0.2's default medium: ammonium nitrogen, glucose 1 mmol/gDW/h, no amino acid. The screens' final selection plates were SD/MSG -His/Arg/Lys/Ura + canavanine/thialysine/G418/clonNAT (Kuzmin 2018 SI; the ontology's `SGA_TM_SELECTION`). The rerun keeps everything else fixed: the same `unique_perturbations.json` (sha256 recorded in each arm's `fba_metadata.json`), the same GLPK solver and 60 s limit, the same `gene.knock_out` deletion, fitness proxy and interaction formulas (imported from `targeted_fba_growth_fast.py`), and adds what the frozen run never recorded: the cobrapy and optlang versions, the torchcell commit, the full bound vector (`medium_bounds.json`, a serialized `MediaBounds` with every component's resolution) and the wild-type exchange fluxes.

| arm | medium | glucose | nitrogen | amino acids | purpose |
|---|---|---|---|---|---|
| `screen_medium` | `SGA_TM_SELECTION_FBA` | 3.3 | MSG at 0.165 | 16 of 20 (no His/Arg/Lys), adenine, no uracil | the rerun |
| `sm_glucose_3.3` | `SM_FBA` | 3.3 | ammonium, open | none | separates the nitrogen source and supplements from the carbon rate |
| `yeast9_default` | `model.medium` as shipped | 1 | ammonium, open | none | reproduces the frozen run with today's cobrapy |

Each arm solves 4,036 singles, 651,181 doubles and 332,313 triples in a process pool and writes `{singles,doubles,triples}_deletions.parquet`, `{digenic,trigenic}_interactions.parquet`, `wt_growth.csv`, `medium_bounds.json` and `fba_metadata.json` to `results/fba_screen_medium/<arm>/`. Smoke test (`--limit 40`, 8 processes): the default arm reproduces the frozen wild-type growth 0.08584397 exactly; the screen medium opens 42 exchanges, resolves every nutrient, and excludes agar and the four selection agents by role.

## 2026.09.10 - Result: the medium moves growth, not the interaction correlation (gilahyper job 1671)

All three arms ran in 37 min on 96 processes (job 1671, commit c7c86263 plus the branch's media module; cobra 0.30.0, optlang 1.8.3, GLPK), every one of the 3 x 987,530 solves `optimal`. Scored by [[experiments.007-kuzmin-tm.scripts.fba_screen_medium_si]] (`results/fba_screen_medium/stats.json`, `summary.csv`).

| arm | open exchanges | WT growth | WT-like singles / doubles / triples | no-growth triples | nonzero tau | r(tau) | r(fitness), covered | inconsistent doubles / triples above min double |
|---|---|---|---|---|---|---|---|---|
| frozen baseline (2025-09-15) | 16 | 0.0858 | 97.7 / 97.0 / 93.1% | 4.9% | 231 | 0.0019 | 0.151, 0.254 | 253 / 173 |
| yeast9_default (rerun) | 16 | 0.0858 | 97.7 / 97.0 / 93.1% | 4.9% | 182 | 0.0034 | 0.151, 0.254 | 157 / 127 |
| sm_glucose_3.3 | 25 | 0.3138 | 98.1 / 97.3 / 93.3% | 4.4% | 27 | -0.0005 | 0.160, 0.269 | 0 / 0 |
| screen_medium | 42 | 0.4966 | 98.5 / 98.0 / 94.3% | 3.5% | 30 | -0.0007 | 0.173, 0.289 | 0 / 0 |

- The default-medium rerun reproduces the frozen wild-type growth to 1.3e-9 and the growth bands to 13 triples out of 332,313, but not the solver failures: 157 inconsistent doubles against 253 (146 with YGR144W in both), 127 triples above their slowest double against 173, and 182 nonzero tau against 231. The +1, +2 and -1 predictions of the frozen run are therefore GLPK failures at glucose 1 that are not reproducible solve to solve, not modeled interactions.
- At glucose 3.3 (both the ammonium control and the screen medium) the internal checks pass with zero violations, and every nonzero tau lies in a fully covered triple: 27 and 30 of them, with values between -0.39 and 0.04 on the screen medium (value counts in `stats.json`).
- The interaction correlation stays at zero on every medium (r between -0.0007 and 0.0034; Spearman 0.026 on the screen medium, P = 0.69 for Pearson). The fitness correlation rises from 0.151 to 0.173 (0.254 to 0.289 within triples containing a model gene), most of it from the carbon rate (SM at 3.3 gives 0.160) and the rest from the supplements.
- Triples at wild-type growth move from 93.1% to 93.3% (carbon rate) to 94.3% (nitrogen source and supplements); no-growth triples from 4.9% to 4.4% to 3.5%.

Panel g of `FigS-yeast9-fba` is `fba_screen_medium_tau.svg` (composed by [[experiments.007-kuzmin-tm.scripts.fba_baseline_compose_figure]], which no longer draws the placeholder); the SI note gained a "Rerun on the screen's medium" paragraph and `tab-fba-screen-medium.tex`.
