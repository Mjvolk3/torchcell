---
id: bd0n9yhjde2e6nq7aqbiqmq
title: Masked_conditioning_oracle
desc: ''
updated: 1785531980633
created: 1785531980633
---

## 2026.07.31 - Bounding What a Masked-Label Objective Could Buy Before Spending Any GPU on It

The v9 masked-label objective is a real build -- per-batch masking, a reveal schedule, cross-gene routing, and a loss restricted to the hidden entries -- and it can only pay off if observing some genes actually constrains the others. This script answers that with no training and no GPU: take residuals against each gene's TRAIN mean, model them as `R ~ N(0, Sigma)`, and score the closed-form conditional mean `E[R_U | R_M] = Sigma_UM (Sigma_MM + lam I)^{-1} R_M` on held-out strains with the project's own `pearson_per_feature`. That is an upper bound on any masked-conditioning architecture whose leverage is linear residual structure -- and it came back at 2-4x the entire trained model, which is what authorized building v9.

### The measurement

`experiments/019-simb-multimodal/scripts/masked_conditioning_oracle.py` -> `results/masked_conditioning_oracle.json` (1,482 single-deletion strains x 6,169 genes; 1,172 train / 155 tune / 155 val; 5 independent draws of the observed set per m; seed 0):

| m observed | val pearson_per_feature | sd over 5 draws | tuned lam |
| --- | --- | --- | --- |
| 0 | 0.0000 | -- | floor by construction |
| 10 | 0.4084 | 0.0310 | 1e-4 .. 1e-2 |
| 100 | 0.6756 | 0.0116 | 1e-3 .. 1e-2 |
| 1000 | 0.7932 | 0.0014 | 1e-2 |

- Against the trained model's 0.198 val `pearson_per_feature` (the reference `conditioning_gain_after_genotype.py` benchmarks against), revealing ten random genes is already 2.1x and a thousand is 4.0x. That gap, not an intuition, is what made the objective worth the build.
- The m = 0 row is exactly 0 *by construction*, not by measurement: the per-gene mean is constant across strains, so every per-feature correlation is identically zero. Recording it explicitly keeps the floor in the artifact instead of in someone's head.
- The reveal schedule was calibrated, not guessed. `residual_covariance_diagnostic.json` puts the residual structure at effective rank 32.78 (the rank-32 subspace carries 59.1% of residual variance), so m = 10 is under-determined, m = 100 over-determines by ~3x, and m = 1000 saturates. The measured curve has exactly that shape -- the across-draw sd collapses 0.031 -> 0.012 -> 0.0014 as m passes the rank.

### What keeps the bound from being self-flattering

- **`lam` is tuned on an inner TRAIN split, never on the evaluation strains.** `Sigma_MM` from 1,172 strains is badly conditioned at m = 1000, so a ridge is mandatory; choosing it on val would inflate the bound. 155 tune strains select `lam`, which is then applied unchanged.
- **Gene means come from TRAIN only.** Centering with a mean that saw the evaluation strains is exactly the leak that manufactures correlation.
- Zero-variance predictions score 0, not NaN -- dropping them would silently score the oracle only on the genes it happened to move.
- The full `[F, F]` covariance is never formed, only the `Sigma_MM` and `Sigma_UM` blocks, which keeps a 6,169-gene solve at tens of MB and seconds of wall time.

### What the number does NOT say, and the two follow-ups that pinned it down

The docstring flags the limit up front: the residual is taken against the per-gene MEAN, not the model's prediction, so this bounds the ABSOLUTE linear gene-gene structure and not the INCREMENT over the current model. Two follow-up scripts turned each caveat into a measurement, and both narrowed the prize:

- **Biology or the array?** m = 1000 at 0.7932 EXCEEDS the replicate-based expression ceiling of 0.775 (`expression_ceiling_replicate.py`) -- the signature of predicting something a re-measurement does not share, since observed and held-out genes come off the same two-colour array. `cross_study_conditioning_oracle.py` separates them on the 82 deletions measured in both Kemmeren 2014 and Sameith 2015: observing in one study and predicting the other retains only **0.51 / 0.57 / 0.62** of the within-study number (`within_kem` 0.4562 / 0.6693 / 0.7832 vs `cross_kem_to_sam` 0.2335 / 0.3815 / 0.4838 at m = 10 / 100 / 1000; `results/cross_study_conditioning_oracle.json`). Roughly half the headline was same-array technical structure.
- **Reachable from genotype?** Our val metric is computed at m = 0, so masked training can only move it if the gene-gene structure overlaps what genotype predicts. `conditioning_gain_after_genotype.py` removes a genotype-conditional kNN mean (prot_T5, k = 25, leave-one-out) before re-running the same oracle and finds the gain essentially untouched -- **0.975 / 0.992 / 1.006** of the col-mean baseline at m = 10 / 100 / 1000 (`results/conditioning_gain_after_genotype.json`). The conditioning signal is close to orthogonal to genotype, i.e. the masked objective reads as an imputation capability rather than a route to a better m = 0 score. That script states its own honest limit: kNN is weaker than the trained model, so this bounds the orthogonality question rather than settling it -- settling it needs the model's own residuals from a checkpoint pass.

### The launcher

`gh_masked_conditioning_oracle.slurm` is deliberately thin and deliberately CPU-only -- `-p main`, no `--gres=gpu`, because this is a covariance solve and the four GPUs were reserved for the training wave; requesting one would idle it for the duration. It pins the BLAS thread count to the allocation (numpy otherwise spawns one thread per physical core regardless of what SLURM granted), sets `PYTHONPATH` to the worktree root so the run imports this branch's torchcell rather than the primary checkout, and echoes the short HEAD into the log so the artifact traces to a commit.

```bash
sbatch experiments/019-simb-multimodal/scripts/gh_masked_conditioning_oracle.slurm
# ORACLE_PROJECT_ROOT=<other tree> to run the same script from a different worktree
```
