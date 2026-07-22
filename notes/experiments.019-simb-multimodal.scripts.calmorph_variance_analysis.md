---
id: w8w596xaw1alpafarptw3x5
title: Calmorph_variance_analysis
desc: ''
updated: 1784704552157
created: 1784704552157
---

## 2026.07.22 - WS10b CalMorph target normalization (source + variance + transform)

Script: `experiments/019-simb-multimodal/scripts/calmorph_variance_analysis.py`
([[experiments.019-simb-multimodal.scripts.calmorph_variance_analysis]]).
Companion training change: `train_cgt_multitask.py`
([[experiments.019-simb-multimodal.scripts.train_cgt_multitask]]).

### Problem

The morphology training target (served `calmorph` vector) was RAW/unnormalized, so the
un-normalized MSE loss was ~4.7 M while Pearson still "rose". WS10b sources a principled
normalization from the Ohya 2005 paper + SI, analyzes per-feature variance, and implements
a per-feature z-score in the training path.

### CalMorph structure (281 vs 501) - resolved

The nominal CalMorph vocabulary is **501 parameters** = **281 BASE** (`CALMORPH_LABELS`,
the served `calmorph` vector) + **220 coefficient-of-variation** (`CALMORPH_STATISTICS`,
the separate `calmorph_coefficient_of_variation` set, prefixed CCV/ACV/DCV/TCV). Verified
against the sha256-pinned raw mirror `mt4718data.tsv` (4718 mutants x 501 cols): 281 in
base, 220 in CV, 0 unmatched. The loader stores RAW values (no pre-transform), so the raw
file's 281 base columns are exactly the served target.

### SI finding on Ohya's OWN normalization (source-verified)

Ohya 2005 SI **Supporting Text** (mirror
`ohyaHighdimensionalLargescalePhenotyping2005/si/si12.md`) prescribes a per-parameter
pipeline on WILD-TYPE data:

- **Box-Cox power transform**, verbatim: *"The wild-type data are divided by the mean and
  transformed by the function defined below"* -> `F_{p,a}(x)`, params chosen to minimize
  the Anderson-Darling statistic.
- **Standardize**, verbatim: *"the transformed data are standardized by
  y = (F_{p,a}(x) - Mean)/SD"* (per-parameter z-score of the transformed WT distribution).
- **Shapiro-Wilk feature selection**, verbatim: *"Thresholding P value with 0.3, 0.5, and
  0.7, the number of parameters whose P value is greater than or equal to the threshold is
  349, 254, and 161, respectively... we chose thresholding the Shapiro-Wilk test P value
  with 0.5, resulting in statistically robust parameters of 254."* -> **kept 254 of 501,
  discarded 247** as non-normal / unreliable (paper.md line 25, 38).

So the paper's own normalization is per-feature (divide by WT mean -> Box-Cox ->
standardize), and it **drops** the degenerate/non-normal parameters. That 254/247 verdict
is a NORMALITY test for their abnormality statistic, spanning all 501 base+CV params - NOT
a variance floor on our 281 base target.

### Variance analysis (per-feature, 4718 mutants)

Outputs: `results/calmorph_feature_variance.csv` + `results/calmorph_variance_summary.json`.

- **Scale span is the smoking gun:** across the 281 base features, `|mean|` in
  [6.7e-5, 1.49e4] and `std` in [1.2e-5, 5.46e3] - ~8 orders of magnitude. Big-scale
  features (Whole_cell_size ~1.49e4, Actin_region_size ~1.16e4) dominate a raw MSE, which
  is why the loss was O(1e6).
- **0** truly zero-std (constant) features.
- **3** degenerate features (robust CV = IQR/|median| < 0.01, zero IQR) on the full 4718:
  `A113_A1B`, `A113_C` (Actin_n_ratio = "no actin patch found ratio", ~0.001-0.002 mean)
  and `C123_C` (Small_bud_ratio, ~0.0005). These are the "values barely change across
  samples" features the user flagged: bounded fractions that are ~0 for almost every strain
  (only a few outliers move). On the TRAIN split (n=3757) the same threshold flags 6
  (adds `A113_A`, `D203`, `D205`) - threshold-boundary sensitive.

### Keep/drop recommendation (the user's flagged decision)

**KEEP all 281 base features** with a per-feature z-score (train-split stats) + an
epsilon-floored std; **FLAG** the degenerate list for optional user-driven dropping. Do NOT
silently drop. Rationale: after z-scoring, a degenerate feature becomes unit-variance and
contributes noise EQUALLY (it does not blow up the O(1) loss), so keeping is safe; and the
SI's 254/247 is a normality verdict over all 501 (base+CV), not a variance floor on our 281
base target, so importing the 247-drop wholesale is unjustified. The degenerate features are
surfaced in `results/calmorph_train_target_norm_global.json` (`degenerate_features`).

### Transform implemented (train path)

`train_cgt_multitask.py`: `compute_per_feature_target_stats()` accumulates per-feature
mean/std over the TRAIN split ONLY (no val/test leakage), applied in
`_extract_targets_and_masks` (`(target - mean)/(std + eps)`) and invertible via
`task.denormalize()` for raw-scale inference. Config `multitask.normalize_vector_targets:
[global]` (+ `target_norm_eps`, `degenerate_robust_cv`). Stats registered as buffers
(device-moved + checkpointed).

### Smoke re-verify (before -> after, `train_cgt_multitask_smoke_morph`)

- **Loss:** ~4.5 M (O(1e6)) -> ~1.14 (O(1)). Train loss decreases 2.78 -> 0.75; with more
  capacity/epochs val loss falls to 0.912 (BELOW the z-scored variance floor of 1.0 -> model
  explains ~9% of variance = real signal). **Goal met.**
- **Pearson:** baseline flatten-Pearson "rose" 0.047 -> 0.114, but that was a **feature-scale
  artifact** - the flatten correlation over [B, 281] was dominated by the huge between-feature
  scale (big features big in both pred and target). After per-feature z-scoring, between-feature
  scale is removed and flatten-Pearson measures across-strain per-feature signal only; the
  smoke reports ~0. The honest "it learns" evidence is the val loss dropping below 1.0.
  **Follow-up for the real run:** report an epoch-level per-feature-averaged Pearson (mean over
  the 281 features of the across-strain correlation), which is the sensitive morphology metric;
  the current per-batch flatten-Pearson is retained from WS10a but is not sensitive post-norm.
