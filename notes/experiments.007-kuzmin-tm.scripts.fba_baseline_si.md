---
id: ukpehrnxuzwmwq1qa27hfiu
title: Fba_baseline_si
desc: ''
updated: 1788665928760
created: 1788665928760
---
Source data and provenance for Supplementary Note `note:fba` (`paper/nature-biotech/sections/si-note-fba.tex`) and the data panels of `FigS-yeast9-fba`, produced by `experiments/007-kuzmin-tm/scripts/fba_baseline_si.py`.

## 2026.09.05 - The Fig. 2d Yeast9 FBA baseline, recomputed from its frozen run

**Input (the only committed record of the baseline):** `experiments/007-kuzmin-tm/results/cobra-fba-growth_backup_20250923_134447/` (run 2025-09-15T17:50, `targeted_fba_growth_fast.py`, 128 processes, 608.6 s; matched by `match_fba_to_experiments.py`). Plus the yeast-GEM 9.0.2 SBML under `$DATA_ROOT/data/torchcell/yeast-GEM/`, loaded through `YeastGEM()` to record the medium and check the wild-type growth (recomputed 0.08584402 vs frozen 0.08584397; the script asserts agreement within 1e-6).

**Outputs:** `experiments/007-kuzmin-tm/results/fba_baseline_si/` (`stats.json` holds every number the note and the figure state; `yeast9_default_medium.csv`; `triples_matched.parquet`, one row per triple with measured and predicted tau and fitness and the count of its genes in the model; `growth_bands.csv`; `coverage.csv`), `paper/nature-biotech/sections/tab-fba-medium.tex`, and the four panels below.

**Measured (all from `stats.json`):**

- Medium: the model default, 16 open exchanges, glucose 1 mmol/gDW/h, ammonium nitrogen, oxygen and ions unbounded; objective `r_2111`; NGAM `r_4046` = 0.7. Solver GLPK (optlang), 60 s limit; all 987,530 solves returned `optimal`, none timed out.
- Coverage: 736 of 4,036 screened genes in Yeast9; triples with 0/1/2/3 genes in the model: 220,392 / 65,214 / 39,288 / 7,419.
- Growth bands (wild-type within 1e-3 / reduced / fitness < 0.01): singles 97.7 / 0.7 / 1.6%; doubles 97.0 / 0.9 / 2.2%; triples 93.1 / 2.0 / 4.9%. Predicted triple fitness takes 38 distinct values at 3 decimals; 1 (308,363), 0 (16,276), 0.994, 0.983, 0.999, 0.15 (888), 0.367 (563), 0.998 cover 99.6%.
- Correlations: tau r = 0.0019 (n = 332,313, P = 0.27, Spearman -0.0007); fitness r = -0.0046 (n = 299,146). 97.6% of predicted |tau| < 1e-6, 99.93% < 1e-3; 231 nonzero (+1: 122, +2: 62, -1: 19). The manuscript's 0.0006 does not reproduce from the frozen files; its subset or run is not recorded.
- Consistency: 253 / 632,616 checkable doubles and 52 / 285,606 checkable triples contradict the single-deletion result (246 doubles contain YGR144W); 173 triples exceed the minimum of their doubles; 8 doubles exceed their singles. Cause not determined.

**Panels (true-size SVGs, `notes/assets/images/007-kuzmin-tm/`):**

![fba_baseline_tau](assets/images/007-kuzmin-tm/fba_baseline_tau.svg)

![fba_baseline_fitness](assets/images/007-kuzmin-tm/fba_baseline_fitness.svg)

![fba_baseline_growth_bands](assets/images/007-kuzmin-tm/fba_baseline_growth_bands.svg)

![fba_baseline_coverage](assets/images/007-kuzmin-tm/fba_baseline_coverage.svg)

Composed into the figure by [[experiments.007-kuzmin-tm.scripts.fba_baseline_compose_figure]]. Background and the media work that followed the run: [[experiments.007-kuzmin-tm.FBA-interaction-experiments]].
