---
id: maj712cqot7cixpzaop3xxl
title: Conditioning_gain_after_genotype
desc: ''
updated: 1785531987871
created: 1785531987871
---

## 2026.07.31 - Is the masked-conditioning gain reachable from genotype? No -- 97.5-100.6% of it survives removing a genotype predictor

`masked_conditioning_oracle.py` established that observing m genes predicts the rest far better than our trained model predicts anything (0.4084 / 0.6756 / 0.7932 at m = 10 / 100 / 1000 versus ~0.198), but it took residuals against each gene's MEAN -- so it measured "how much gene-gene structure exists", not the question the v9 masked-label objective actually turns on. Our val metric is computed at m = 0, because at inference we have genotype and nothing else; a masked objective can only move that number if the gene-gene structure it exploits overlaps what genotype can reach. This script removes a genuine genotype-conditional predictor first and re-runs the oracle on the remainder. Essentially all of the gain survives, so the objective optimizes a channel that is switched off exactly where we score.

- **The comparison.** `R_mu = Y - mu_col` (per-gene train mean, what the first oracle used) versus `R_knn = Y - muhat_knn`, where `muhat_knn` is a leave-one-out cosine-similarity-weighted mean over the k = 25 nearest OTHER strains in `prot_T5_all` space -- i.e. a predictor using only the perturbed gene's sequence embedding, exactly the information the model has. The diagonal of the similarity matrix is set to `-inf` so a strain never predicts itself (which would make its residual identically zero). Split constants, `conditional_mean()` and `per_feature_pearson()` are IMPORTED from `masked_conditioning_oracle.py` rather than re-derived, which is what makes the two number sets comparable instead of merely similar.

- **Result** (`results/conditioning_gain_after_genotype.json`; val per-feature Pearson, mean +/- SD over 5 mask draws; n_strains 1482, n_genes 6169, n_train 1172, n_val 155, ridge tuned on a disjoint 155-strain tune split, seed 0):

  | m | residual = `Y - col_mean` | residual = `Y - knn(prot_T5, k=25)` | retained |
  |---:|---:|---:|---:|
  | 10 | 0.4169 +/- 0.0272 | 0.4066 +/- 0.0245 | 97.53% |
  | 100 | 0.6749 +/- 0.0088 | 0.6698 +/- 0.0079 | 99.24% |
  | 1000 | 0.7931 +/- 0.0020 | 0.7977 +/- 0.0019 | 100.58% |

- **The drop is not distinguishable from zero.** At m = 10 the loss is 0.0103, well inside the +/- 0.027 per-draw SD; at m = 1000 the kNN residual scores slightly HIGHER than the mean residual. Removing the genotype prediction takes away nothing the conditioning oracle was using.

- **How this determines the reading of the v9 arms.** A teacher-forced masked-label objective must be justified as an imputation capability with m > 0 reported explicitly -- not as a route to a better m = 0 score. If a v9 arm's m = 0 val Pearson moves, the cause is something other than the conditioning signal (regularization, optimization, the pair-rank decoder), and attributing it to masked conditioning would be wrong.

- **Consistent with, but not implied by, `residual_covariance_diagnostic.json`.** That diagnostic reported split-half reproducibility of the residual correlation PATTERN at 0.8706 (col_mean) versus 0.8687 (knn), a drop of 0.002. A correlation pattern can survive while its exploitable MAGNITUDE shrinks; this script shows the magnitude survives too.

- **Honest limit, recorded as the follow-up.** kNN in embedding space is weaker than the trained model, so this BOUNDS the orthogonality question rather than settling it. The exact statement needs the model's own residuals, which requires a checkpoint inference pass.

- **Observation, untested reading:** the m = 1000 oracle (0.7977) sits at/above the 0.7746 replicate ceiling for expression. These are not the same estimand -- conditioning on 1000 genes of the same array can exploit array-level structure that a genotype-only predictor cannot -- so the oracle numbers must not be read as a target for the m = 0 metric.

- **Wrapper** (`gh_conditioning_gain_after_genotype.slurm`) is deliberately CPU-only: no `--gres=gpu`, because this is a covariance solve (30 solves of a 1000x1000 system at worst) and the four GPUs are reserved for the training wave -- requesting one would idle it. It pins `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` to `SLURM_CPUS_PER_TASK` so numpy does not spawn a thread per physical core regardless of the allocation, and puts the experiment `scripts/` dir on `PYTHONPATH` so the sibling-module import above resolves. Root is overridable via `ORACLE_PROJECT_ROOT`. Produced by job 1438 on gilahyper (source `21e93e52`); job 1435 aborted first on a wrong import path for `NodeEmbeddingBuilder`.

- **Caveat:** the slurm file's header frontmatter still carries the `gh_masked_conditioning_oracle` path/link it was copied from, so its dendron backlink points at the wrong note.
