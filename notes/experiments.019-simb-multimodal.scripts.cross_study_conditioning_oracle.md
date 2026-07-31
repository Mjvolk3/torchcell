---
id: 0er1c5zth1dsk3lukcc9zh7
title: Cross_study_conditioning_oracle
desc: ''
updated: 1785531929858
created: 1785531929858
---

## 2026.07.31 - Deciding whether the conditioning oracle predicts biology or the array it was measured on

The within-study masked-conditioning oracle scored `pearson_per_feature` = 0.7932 at m = 1000 (`masked_conditioning_oracle.py` -> `results/masked_conditioning_oracle.json`, 155 held-out Kemmeren strains), which sits ABOVE the 0.7746 replicate ceiling (`expression_ceiling_replicate.py` -> `results/expression_ceiling_replicate.json`, mean_g sqrt(r_g) over the 82 cross-study replicate pairs, mean reliability r = 0.6111). A score above the ceiling for predicting a re-measurement is only possible if the predictor is reproducing something the re-measurement does not share -- i.e. the target's own array. This script settles which, by fitting the conditioner on Kemmeren 2014 and scoring it against Sameith 2015's independent measurement of the same 82 deletion strains, and the answer is that roughly 40% of the within-study number was same-array structure.

- **The design is a single-variable swap.** All four arms use the same observed-gene draw, the same Sigma (fitted on 1247 Kemmeren strains with the 82 shared ones excluded from both mu and Sigma), and the same tuned lam; only the study that measured the TARGET changes between `within_kem` and `cross_kem_to_sam`. That is what makes the drop attributable rather than a confounded comparison. `conditional_mean` and `per_feature_pearson` are copied verbatim from the within-study script so the two numbers are the same statistic.

- **Measured** (`results/cross_study_conditioning_oracle.json`, n_eval = 82 shared single deletions, 6169 complete-case reporter genes, 0 dropped for NaN, 5 draws, seed 0):

  | m | within_kem | within_sam | Kem -> Sam | Sam -> Kem | xstudy ceiling | cross/within |
  |---|---|---|---|---|---|---|
  | 10 | 0.4562 | 0.3649 | 0.2335 | 0.2122 | 0.6111 | 0.512 |
  | 100 | 0.6693 | 0.6417 | 0.3815 | 0.3922 | 0.6110 | 0.570 |
  | 1000 | 0.7832 | 0.7779 | 0.4838 | 0.4803 | 0.6107 | 0.618 |

  At m = 1000 the across-draw sd is <= 0.005 in every arm, and the permuted-strain null is -0.015 to -0.003 (sd ~ 0.027), so the ~0.30 within-vs-cross gap is orders of magnitude outside the noise.

- **The ceiling violation is resolved exactly, and it is not a Kemmeren quirk.** `within_kem` = 0.7832 on the 82 shared strains reproduces the 0.7932 measured on 155 different Kemmeren strains, so the drop is not the evaluation set changing. `within_sam` = 0.7779 puts the SECOND study at the same place -- both studies independently exceed the 0.7746 replicate ceiling when observed and held-out genes come off the same array, and both fall to ~0.48 when they do not.

- **0.48 is not zero, and it must be read against 0.6107, not 1.0.** `xstudy_agreement` -- the two studies' own per-feature Pearson on the identical held-out gene set at identical n -- is what a perfect biological predictor could score. The cross arms recover 0.4838 / 0.6107 = 0.79 of that. So there IS substantial study-independent cross-gene structure; what the within-study run overstated was how much.

- **Reframing: the oracle bounds IMPUTATION, not genotype -> expression.** The 0.7932 headline describes a model given real measured values from the same hybridization as its target. A masked-label objective trained inside Kemmeren would spend part of its capacity learning Kemmeren's measurement process, and the within-study oracle overstates the prize by the size of the gap. The honest bound for a predictor that must generalize off the array it was fit on is the ~0.48 cross number, capped by the ~0.61 cross-study agreement -- both far below the 0.7746 replicate ceiling that a pure genotype -> expression model is scored against.

- **The collapse is not a tuning artifact.** lam is selected on a 155-strain within-Kemmeren tune split and frozen across all four arms; the script also records an oracle-lam variant chosen on the evaluation strains themselves. At m = 1000 the oracle-lam column is IDENTICAL to the frozen-lam column to all reported digits (every draw picked lam = 0.01 anyway), so no choice of ridge recovers the cross arm.

- **Method guards worth keeping.** Genes with any NaN in either study are dropped rather than imputed (imputing would fabricate exactly the cross-gene structure under audit); per-gene means come from TRAIN Kemmeren only and the same mu is applied to Sameith, which is harmless because per-feature Pearson is invariant to per-column shifts (per-gene SCALE differences between studies remain a stated limitation); pairing uses `_deletion` verbatim from `expression_ceiling_replicate.py`, so the pairs are literally the ones that produced the ceiling being argued against.

- **`gh_cross_study_conditioning_oracle.slurm` is a deliberately CPU-only wrapper.** No `--gres=gpu`: this is a numpy covariance solve, and a GPU request would idle a card the training wave needs. It pins `OMP/OPENBLAS/MKL_NUM_THREADS` to `SLURM_CPUS_PER_TASK` (numpy otherwise spawns one thread per physical core regardless of the allocation) and sets `PYTHONPATH="$PROJECT_ROOT"` so a worktree run does not silently import the primary checkout's torchcell. The `--mem=32g` / `04:00:00` request is slack, not need -- the full [F, F] covariance is never formed, only the Sigma_MM and Sigma_UM blocks.

  ```bash
  sbatch --export=ALL,ORACLE_PROJECT_ROOT=$PWD \
      experiments/019-simb-multimodal/scripts/gh_cross_study_conditioning_oracle.slurm
  ```
