# Reviewer 6 of 10: training dynamics, objective and regularization (2026-10-04)

Read-only audit by an independent agent; every hypothesis is labeled "Hypothesis (untested)".

Path legend (abbreviations used below):

- `results/...` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/...
- `conf/...` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/...
- `10-launch.tex` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex
- `0-strand.tex` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex
- `02_trunk_collapse.md` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/02_trunk_collapse.md
- `torchcell/models/equivariant_cell_graph_transformer.py` = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py
- W30 files = /Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/conf/cgt_030_s3_r_tok_fit_000.yaml and /Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/torchcell/losses/point_dist_graph_reg.py

## 1. ESTABLISHED (measured)

- **Training has not converged, on either side.** In v19 K_expr run jnqmuhj8, validation loss is flat at the mean-predictor level (0.2509 at epoch 0, 0.2505 at epoch 40) and then rises to 0.2628 at epoch 1,100. `pred_sd_ratio` stays at or below 0.007 until epoch 60, and launch comes at about epoch 80. Train-set Pearson in eval mode (`traineval/expression/pearson_per_feature`) goes 0.580 to 0.621 between epochs 1,000 and 1,190 while `traineval nmse` falls 0.644 to 0.597, so it is still improving at the end. qcnbk8ue and olkikplr behave the same way (W&B histories, 1,200 epochs each). The 10,000-epoch run hx8pxdic reaches a train Pearson of 0.7769 at epoch 9,000 and 0.7799 at 9,975: train fit flattens below 0.8 and never memorizes (`results/expression_curve_hx8pxdic.csv`).
- **Loss minimum versus metric peak.** In v19, validation loss bottoms at mean epoch 72, 70 and 87 for K_expr, K_joint and K_prot (n=12 each, `results/joint_checkpoint_readout.json` `v19.runs`). That minimum sits on the end of the plateau where predictions are still near-constant: `pred_sd_ratio` is 0.048 at epoch 80 in jnqmuhj8. What each stopping rule gives you:

| Arm | Pearson at loss minimum | Fixed-window score | Roll-max (epoch) |
|---|---|---|---|
| K_expr, expression | 0.047 (n=12) | 0.094 (n=11) | 0.099 (mean epoch 1,058) |
| K_prot, proteome | 0.038 | 0.059 | 0.087 (mean epoch 567) |

  In v9 under the quantile objective (n=18, `results/loss_min_vs_pearson_peak.json` `by_dist.quantile`), loss bottoms at median epoch 477 with Pearson 0.143 there, against a roll-max of 0.195 at median epoch 3,270. Roll-max is an upward-biased order statistic.
- **Predictions are over-dispersed, not too narrow.** For centered predictions, nmse ≈ 1 − 2sr + s², where s is the prediction spread ratio and r the Pearson. jnqmuhj8 at epoch 1,100 (r 0.126, s 0.412) gives 1.066 against a logged 1.089. The MSE-optimal spread is s = r. A post-hoc rescale gives nmse 0.944 at s/r 1.95 (hx8pxdic) and 0.949 at s/r 2.21 (b50f93ju) (`results/expression_objective_diagnosis.json` `calibration`), and leaves Pearson unchanged. Arithmetic, not measured: a distribution-matching term pushes s toward 1, which gives nmse ≈ 1.76 at r = 0.12.
- **Regularization and optimization settings, with budgets.**
  - Weight decay: v15 window means are 0.1095 (reference), 0.1201 (1e-2) and 0.0997 (1e-1), n=4 per arm. Runs ended at median epoch 1,504 to 2,141 (`results/v15_wd_readout.json`). v10 tested 1e-4: +0.005, t 0.53, 16 v 16, at 990 epochs or fewer (`results/v10_grid_factorial.json`).
  - Learning rate was swept only by the v7 Optuna search: 289 runs at median 67 epochs. Spearman(lr, score) is −0.51, but Spearman(epochs, score) is +0.47, so the result is confounded with budget (`results/round_leaderboards.csv`).
  - Dropout: one run each. tc33mjlf (dropout 0) and 1vhu95lc (dropout 0.1) are equal at epoch 90 (0.145, 0.137) but end at 0.139 against 0.198 by epoch 1,600.
  - Everything else is constant in v19. No scheduler (`lr_scheduler.type null`, warmup 0), and batch size 32. Gradient clipping at 10 never binds, since the gradient norm is 0.02 to 0.04. Early stopping is off.
- **Capacity was never tested at a real budget.**
  - Width and depth were varied only in rounds with median budgets of 19 to 87 epochs (v2, v3, v5, v6, cgt_multitask). The median launch epoch is 67 to 113 (`02_trunk_collapse.md` §1c).
  - The only matched long-budget contrast is v10: L2/h45 (0.68M) against L6/h90 (1.45M) gives −0.012, t −1.38, 16 v 16, at 990 epochs.
  - None of the 1,007 runs under 1M parameters goes past 199 epochs, except v8 (0.888M) and v10 (0.683M).
  - In v8, 0.888M models reach 0.13 to 0.15 by epochs 70 to 110 (17 runs, roll-max epoch about 100).
- **Wall time does not depend on model size.**
  - Time per epoch: v10 runs at 188 s for L2/h45 against 194 s for L6/h90 (n=32 each). v6 runs at 41.8 s for 0.5 to 1M against 46.5 s for 4 to 10M.
  - In v19, three runs share one card: one Slurm job hosts all three arms, for example job 2421439 runs jnqmuhj8, ipwprg20 and olkikplr. Median GPU utilization is 97%.
  - A plain epoch takes 38.8 s. An epoch that includes the train-set eval pass takes 112.8 s, so that pass is 20% of epoch time. Another 12% of runtime falls outside epochs.
  - Packing benchmark: one run alone does 2.23 it/s; two runs packed on one card do 1.51 it/s each (`results/packing_benchmark/`).
  - The trunk runs once per step on the 6,608 wild-type tokens with batch dimension 1. Strain identity enters only after it (`torchcell/models/equivariant_cell_graph_transformer.py` forward, about line 2915), so the trunk's output does not depend on the strain.
- **What the working interaction models (W30) actually use.**
  - Loss: MSE + 0.1 × Sinkhorn W2 (geomloss, blur 0.05). The W2 term compares the batch's scalar predictions with the batch's targets, with no buffer, on a per-GPU batch of 256 to 512. A graph regularizer is added with λ 1 × 0.001 per head on layer-1 attention (W30 files in the legend).
  - Size and training: 4.78M parameters (3.13M in the trunk), h180, L8. Training sets are 301,386 records (run 0yw7moue) and 1,046,316 records (xy3xnpau), which is 270 to 950 times v19's 1,103 strains. 36 to 134 epochs; learning rate 1e-4 with cosine warm restarts, or 2.5e-4 constant; 2 to 4 GPUs.
  - What v19 lacks of these: the distribution term, graph regularization (`graph_reg_lambda 0`), the learnable embedding, the large batch and the data.
- **Embedding content outweighs trunk size.** In the v10 factorial, embedding content has an effect of +0.068 (t 7.8); the trunk-size effect is −0.012.

https://wandb.ai/zhao-group/torchcell_019_prot_v19/runs/jnqmuhj8

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/1vhu95lc

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

## 2. NOT ESTABLISHED OR CONTRADICTED

- **"145 runs from 0.12M to 4.7M, pearson(log10 params, score) = −0.129"** (`10-launch.tex` l.517, `0-strand.tex` l.285) has no generating script or result file. I could not reproduce it from `round_leaderboards.csv`. Pooled over the expression strand it is −0.001 (n=1,182). Within single rounds it ranges from −0.41 (v3) to +0.37 (v8). It is also confounded with dataset, metric and budget. "Scale is not the lever" is untested at a real budget.
- **"Overfitting rabbit hole" is contradicted.** The train set is not memorized (train Pearson at most 0.78 after 10,000 epochs), and validation Pearson rises while validation loss rises. The validation loss is the wrong ruler here; the curves do not show classic overfitting.
- **Short screens preserving rank is not established.** `budget_rank_preservation.json` measures persistence between seeds within one configuration (22 of 24 runs in one cell), not ranking between configurations. In the one dropout pair (v8), the effect appears only late.
- **Whether a distribution term helps Pearson has not been measured in 019.** The v7 energy, CRPS and NLL arms ran about 64 epochs.

## 3. ERRORS AND INCONSISTENCIES

- The v15 config states a 6,000-epoch budget (`conf/cgt_expr_v15_wd.yaml`), but the runs ended at median epoch 1,504 to 2,141, and one run (996rzljw) stopped at 1,109.
- v19 is reported as "6.67M", but the incumbent at the same h90/L6 is 1.19M with a 590k trunk (`launch_plan_evidence.json` `param_count`). The extra 5.5M is presumably in the input preprocessor for the four embeddings. That is an inference; I did not check a v19 checkpoint.
- `train/grad_norm_clip_frac` is the mean gradient norm divided by 10, not the fraction of steps clipped.
- v19 logs `loss_fn: mse`, but the expression head trains on pinball loss (`dist: quantile`).
- `save_loss_min: true` saves a near-mean predictor.
- `pinball_vs_mse` has a figure but no results JSON.
- Over-dispersion and the "under-shrinkage" name: the summary defines under-shrinkage as s > r (predictions spread more than the noise warrants). The "prediction spread is only 0.39 to 0.46" framing reads as the opposite, as if predictions needed more spread, and a distribution term would act on that misreading.

## 4. UNTRIED OR UNDER-BUDGETED LEVERS (ranked)

1. **Cache or remove the trunk.** Its output does not depend on the strain, and time per step does not shrink with model size. A trunk-free model (embedding MLP, then the perturbation head) has never been measured.
2. **Data learning curve**, training on 25, 50 and 100% of strains: never run.
3. **Early stopping on a fixed metric window**, plus the launch-restart rule (E1 in `02_trunk_collapse.md`).
4. **Learning-rate warmup and schedule, and a larger batch** with a scaled learning rate. Never run, while the launch plateau costs 2,000 to 3,000 steps.
5. **Strong dropout and weight decay on a small model at 1,200 or more epochs.**
6. **Linear probe on frozen embeddings, then fine-tune.**
7. **Distribution term:** lowest priority, because it acts against the over-dispersion that is actually measured.

## 5. TOP THREE RECOMMENDATIONS

- **R1. Fast harness (about 1 GPU-day).**
  - Experiment: set `train_eval_every` to 100, validate every 5 epochs, use batch 128, and add a trunk-free arm at about 0.3M parameters.
  - Control: v19 K_expr on splits 0 to 2.
  - Hypothesis (untested): about 10× less wall time per run. The parts that are measured: train-set eval is 20% of epoch time, and size does not affect time per epoch.
  - Stop: adopt the harness if the trunk-free arm comes within 0.02 of K_expr's window score (sd of paired differences 0.030, n=11).
- **R2. Learning curve and the cause of the ceiling (about 1 GPU-day with R1's harness).**
  - Experiment: trunk-free model on 25, 50 and 100% of strains, 3 splits each, plus a label-shuffle control.
  - Hypothesis (untested): score rises roughly in proportion to log(n), which would mean the limit is data, not optimization.
  - Stop: if the slope is not positive, attack generalization to unseen deletion genes (embeddings, a pair term), not training.
- **R3. Screening protocol, then confirmation.**
  - Screen: 0.3 to 0.7M models, 300 epochs (past the launch p90 of at most 115 epochs). Score is the mean validation Pearson over epochs 200 to 300, taken as a paired difference against the reference on 3 splits.
  - Cost and resolution: 100 configs × 3 splits = 300 runs, about 2 GPU-days if R1 delivers 10×. Without R1, at 17 s per epoch alone on one card, it is about 18 GPU-days. The resolution is about 0.04 to 0.05.
  - Confirm: the top 3 configs plus the reference at 1,200 epochs on 6 splits, about 6 card-hours per run at three per card, roughly 6 GPU-days.

**Judgment.** The dominant limit is too few strains combined with having to generalize to deletion genes never seen in training. Embedding content is worth +0.068 and trunk size −0.012; train fit caps near 0.78; the interaction models train on 270 to 950 times more data. The second problem is cost: an optimization plateau of about 80 epochs and fixed per-step overheads. Neither the objective nor capacity is the main problem. R2 is the cheap test that tells these apart.

Files:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/round_leaderboards.csv
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_objective_diagnosis.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v10_grid_factorial.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v15_wd_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/torchcell/losses/point_dist_graph_reg.py
