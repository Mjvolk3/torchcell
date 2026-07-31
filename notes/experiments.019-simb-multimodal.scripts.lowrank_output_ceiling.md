---
id: y8uv4goafwfd6zddod081hr
title: Lowrank_output_ceiling
desc: ''
updated: 1785531944372
created: 1785531944372
---

## 2026.07.31 - Prove headroom is not the constraint, so a null on the pair-rank ladder indicts the mechanism and not the capacity

Wave 6 spends eight GPU arms asking how much (perturbation, gene) pair structure the decoder can express, and a flat result is only interpretable if we already know the arms were not capacity-starved. This script answers that on CPU, with no training: it measures the best per-feature Pearson a rank-r output head could reach **if its r coefficients were predicted perfectly**, so every rank on the wave-6 ladder carries a stated ceiling before it is given a GPU slot. Run as job 1437 on GilaHyper (`019-lowrank-ceiling_1437.out`, 2026.07.30); numbers below are from `experiments/019-simb-multimodal/results/lowrank_output_ceiling.json`.

- **The oracle construction is what makes it a bound.** Y is loaded with the traversal copied byte-for-byte from `residual_covariance_diagnostic.py` (1482 single-deletion strains x 6169 reporter genes), split with seed 0 so val is the *same* 155 strains as `masked_conditioning_oracle.py`, gene means taken from fit strains only, then V_r = top-r right singular vectors of the fit residuals. Val is reconstructed as `R_val V_r V_r^T` -- the coefficients come from val itself, which no model can see. Coefficient-prediction error is therefore set to zero and only the error forced by the r-dimensional bottleneck remains; a real rank-r head scores at or below the row.

| rank | train basis | val basis (cheating envelope) | train-only basis | frac val var |
|---|---|---|---|---|
| 16 | 0.6662 | 0.7836 | 0.6660 | 0.529 |
| 32 | 0.7265 | 0.8665 | 0.7276 | 0.619 |
| 33 | 0.7314 | 0.8698 | 0.7295 | 0.624 |
| 64 | 0.7799 | 0.9285 | 0.7781 | 0.702 |
| 128 | 0.8126 | 0.9861 | 0.8105 | 0.756 |
| 1327 (full fit basis) | 0.8720 | -- | -- | 0.855 |

- **This is what sized `V_basis16/32/64` in `gh_expr_008_arm.sh`.** r=32 is the measured participation-ratio effective rank (32.78, `residual_covariance_diagnostic.json`); r=16 is deliberately *below* it, as the arm that should lose something if the low-rank story is right; r=64 is the smallest swept rank whose ceiling (0.7799) clears the replicate noise ceiling 0.7746 (`expression_ceiling_replicate.json::primary_ceiling_mean_sqrt_r.ceiling`), i.e. the first rank at which the bottleneck provably costs nothing measurable against labels this noisy. The wave-6 header states the consequence directly: 0.727 at r=32 and 0.780 at r=64 against a 0.198 best (`reference.observed_model_pearson_per_feature` recorded 0.109, the earlier value from `expression_ceiling_replicate.json::observed`) -- 3.7-3.9x headroom over 0.198, so a null on the ladder is a statement about the mechanism.
- **The effective rank overstates compressibility for this metric, and the curve says so.** The train-basis curve is still climbing past r=128 (90% of the full fit basis only at r=128, 95% at r=256), because per-feature Pearson weights a low-variance gene exactly as much as a high-variance one while the SVD spends its components on high-variance genes. Reading r=33 as "enough" would have been wrong: its ceiling is 0.7314, still under the noise ceiling.
- **Two honest qualifications are carried in the script, not buried.** The fit basis is Frobenius-optimal, not metric-optimal, so `ceiling_val_basis` (V_r from val's own residuals) is reported as the fully cheating envelope -- the gap 0.7265 vs 0.8665 at r=32 is how much of the ceiling is basis *choice* rather than rank. And `ResponseBasisHead` adds its rank-r term to a full-rank local readout, so this curve bounds a head whose entire output is rank-r, not the existing additive arm.
- **Sensitivity, not assertion.** Refitting mu and the basis on the 1172 train-only strains (excluding the oracle script's 155 tune strains) moves every ceiling by <= 0.002, so treating tune strains as fit data -- legitimate here since nothing is tuned -- is checked rather than assumed. Reference numbers (`replicate_noise_ceiling`, `masked_oracle_m1000_val_mean`, `effective_rank_residual_covariance`) are read from their own result JSONs at write time so this file cannot drift from its citations.
- **`gh_lowrank_output_ceiling.slurm` requests no GPU on purpose.** The work is three thin SVDs in numpy (the [1327, 6169] fit residual, the [1172, 6169] train-only residual, the [155, 6169] val residual); the 6169^2 covariance is never formed, and the 32g request is slack. It pins `OMP/OPENBLAS/MKL_NUM_THREADS` to `SLURM_CPUS_PER_TASK`, because this job is almost entirely LAPACK gesdd and numpy would otherwise thread to every physical core and take the node while the wave's GPUs are running.

```bash
sbatch --export=ALL,ORACLE_PROJECT_ROOT=$PWD \
  experiments/019-simb-multimodal/scripts/gh_lowrank_output_ceiling.slurm
```
