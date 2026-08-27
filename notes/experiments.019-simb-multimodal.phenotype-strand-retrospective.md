---
id: zy9ird64wx0zsgwyqxlbd8z
title: Phenotype Strand Retrospective
desc: ''
updated: 1787795214616
created: 1787795214616
---

## 2026.08.26 - Retrospective across all six phenotype strands

Typeset deliverable: `notes-tex/019-simb-multimodal/main.pdf`
(`make -C notes-tex/019-simb-multimodal` to rebuild, `make check` for the gate).
This note is the working index; the PDF is the document.

### What it covers

One section per strand, all scored the same way from one leaderboard:
expression (Kemmeren + Sameith), the masked-label expression objective, morphology
(CalMorph / Ohya), the joint expression+morphology arm, betaxanthin (Cachera),
betaxanthin with a metabolome head, amino-acid pools (Mulleder), and beta-carotene
(Ozaydin).

### New measurements made for this retrospective

- **Morphology ceiling, measured for the first time: 0.611** over the 278 modeled
  CalMorph features (122 WT replicates against 4,718 mutants), with 201 of 278
  individually above 0.5. Best achieved is 0.0824, so 13.5% of ceiling.
  Script: `experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py`
  -> `results/morphology_noise_ceiling.json` + `results/morphology_feature_ceiling.csv`.
- **Morphology was only ever trained on 1,161 strains** of the 4,718 that exist,
  because every morph config sets
  `require_modalities: [expression_log2_ratio, calmorph]`. Fitness shares 4,220
  strains with morphology against expression's 1,440.
- **Amino-acid profile predicts betaxanthin at out-of-fold r = 0.298; tyrosine, the
  precursor, predicts it at 0.064** (marginal r = -0.076). About three quarters of
  the profile's power survives regressing out single-mutant fitness. See
  [[experiments.023-metabolome-betaxanthin-joint.scripts.betaxanthin_amino_acid_predictivity]].
- **The metabolome auxiliary head currently COSTS betaxanthin performance**:
  -0.0265 +- 0.0159 over the five comparable 023 grid cells, 2 of 5 positive. The
  +0.0203 that motivated the replication reproduces (+0.0192) in the one cell it
  came from.
- **Expression best is now 0.2407** (v9 run `hx8pxdic`, peak epoch 9,188 of 9,997),
  up from the 0.2044 the round retrospective recorded, and still peaking in the last
  tenth of its run. No peak has been observed at any budget.

### Artifacts

| what | where |
|---|---|
| leaderboard, all strands | `experiments/019-simb-multimodal/results/round_leaderboards.csv` |
| leaderboard summary | `results/round_leaderboards_summary.json` |
| paired bx/metabolome arms | `results/bx_aa_paired_summary.json` |
| morphology ceiling | `results/morphology_noise_ceiling.json` |
| new panels | `results/retrospective_panels.json` |
| figure option boards | `notes/assets/drawio/Fig3-options.drawio`, `Fig6-options.drawio` |

### Scripts

- [[experiments.019-simb-multimodal.scripts.pull_round_leaderboards]]
- [[experiments.019-simb-multimodal.scripts.build_retrospective_tables]]
- [[experiments.019-simb-multimodal.scripts.plot_retrospective_panels]]
- [[experiments.019-simb-multimodal.scripts.build_figure_option_boards]]
- [[experiments.019-simb-multimodal.scripts.morphology_noise_ceiling]]

### Panels

![](assets/images/019-simb-multimodal/retrospective_achieved_vs_ceiling.svg)

![](assets/images/019-simb-multimodal/retrospective_peak_position.svg)

### Related

[[experiments.019-simb-multimodal.expression-round-retrospective]] ·
[[experiments.020-cachera-betaxanthin.merzbacher-comparison]] ·
[[plan.simb-2026-multimodal-cgt.2026.07.21]]

## 2026.08.27 - Revision 2, answering twelve review comments

Dispositions ledger: `notes-tex/019-simb-multimodal/review/round-1-dispositions.md`.
Comments pulled with `python notes-tex/common/zotero_comments.py 019-simb-multimodal`.

### Two claims in revision 1 were WRONG

1. **"Loss and metric point in opposite directions" was true only of the QUANTILE LOSS.**
   Measured on both long runs: val `mse` bottoms at epoch 9,175 (v9) and 3,922 (v8), i.e.
   ALONGSIDE the Pearson peaks at 9,674 and 3,921. Only `val/loss` turns early (463, 141).
   Consequence: best-by-`mse` checkpointing would have been nearly right, so checkpointing
   is a narrower problem than recorded and is not what holds the strand at 0.24.
2. **The best betaxanthin number is 0.4301, not 0.372.** 0.4301 is `betaxanthin_002` in
   `torchcell_020_betaxanthin`/`_v3` (n_train 4,235); 0.372 is `_v4` (n_train 3,698), whose
   split pins the 640 Merzbacher test genes OUT of training. The comparison uses the lower
   number on purpose.

### New this revision

- **`nmse` never returns below 1** after ~epoch 400: the model is no better than "predict
  each gene's mean" in squared error while reaching r = 0.236. Cause is calibration:
  `nmse = 1 + s^2 - 2rs`, minimized at s* = r; the run sits at s = 0.460 vs r = 0.236,
  **1.95x over-dispersed**. Post-hoc rescale by r/s: nmse 1.010 -> 0.944, no correlation
  changes. See [[experiments.019-simb-multimodal.scripts.expression_objective_diagnosis]].
- **Cost: the best expression run was 91.4 h = 3.81 DAYS** (9,999 epochs, 32.9 s/epoch).
  ~2,500 epochs/GPU-day. This sets the campaign arithmetic.
- **Masked unmasking cannot supply the pair term**: at k=0 the revealed set is empty, every
  encoded feature is exactly zero, and the forward pass is identical to the unconditioned
  model. Scoring happens at k=0. Different axis from the pair term.
- **010 uses BOTH Kuzmin papers**: 91,111 + 301,798 = **392,909** records, not 91,111.
- **Full project census**: `project_census.py` -> `results/project_census.json`.
  **28 projects, 2,187 runs.** Morphology is the only strand whose n_train never varies,
  and it never varies from 1,161 across all 397 morphology-bearing runs.
- **Coverage is now stated in the document**: per-run history covers 13 of 28 projects;
  `pull_round_leaderboards.py` is cached-per-project and resumable, rest queued.
- **New section: the expression + morphology campaign**, sized in GPU-days
  (E = 4,000 epochs justified by an arm observed to peak at 3,921; two seeds justified by a
  measured across-init sd of ~0.006).

### Still queued

`pull_round_leaderboards.py` on the 15 remaining projects. It is resumable; delete
`results/_leaderboard_cache/<project>.json` to refresh one.
