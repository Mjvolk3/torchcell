# Reviewer 1 of 10: expression rounds up to the SIMB retrospective (2026-10-04)
Read-only audit by an independent agent; every unmeasured statement is labeled as a hypothesis.

## 1. ESTABLISHED

| round/wave | varied | runs, seeds | epochs | result (roll_max) | valid? |
|---|---|---|---|---|---|
| expr, v2, v3, v5, multitask | capacity, norm, head | 26/120/60/35/496, seed 0 | ≤151 | ≤0.130 | no: truncated, split = seed |
| v6 | head, L, lr, width | 158, seed 0 | ≤199 | 0.163 | no |
| v7 (Optuna) | head × lr × dropout | 295, seed 0 | ≤199 | 0.163; median 0.034 | no |
| v8 waves 1 to 4b | 21 decoder arms | 67, seeds 0/1/2/42 | 18 to 300 | 0.13 to 0.175 (seed 0) | no: split confound |
| v8 wave 5 | mixing, mask layer, lr | 8 to 13 seeds, split pinned | 399 | W_ref 0.1511 ± 0.0054 | valid at 399 epochs only |
| v8 wave 6 | pair-rank ladder, regularization | 12 arms, n=1 | 4,091 to 4,372 | 0.138 to 0.228 | unreplicated |
| v9 | 8 mask schedules | n=1 each | 9,900 | 0.166 to 0.238 | unreplicated |

Sources: a groupby of /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/round_leaderboards.csv, and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/decoder_arms_torchcell_019_expr_v8.csv.

- **Training budget drives the score, measured within single runs.** On the 5-point smoothed running max for b50f93ju, the score is 0.1466 through epoch 500, 0.2019 at 2,000 and 0.2304 at 4,000 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_curve_b50f93ju.csv). hx8pxdic reaches 0.1268 at 500 and 0.239 at 9,500 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_curve_hx8pxdic.csv).
- **Init-only noise is small at 399 epochs.** With the split pinned, W_ref has sd 0.0054 (n=13) and X_mix has sd 0.0053 (n=8) (round_leaderboards.csv, stage-wave5). Three seed-0 W_ref runs on different machines span only 0.1494 to 0.1507.
- **The data split dominates waves 1 to 4b.** Seed 0 sits +0.0935 above the mean of the other seeds, averaged over 9 arm blocks (computed from decoder_arms_torchcell_019_expr_v8.csv). In those waves `seed` also set the partition (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py L2663-2680).
- **Dropout 0.1 beats 0** at 1,615 epochs: paired +0.0165 and +0.0545 (n=2; round_leaderboards.csv, H1_ref vs H1_nodrop).
- **The additive operator has no pair term**, shown as an exact identity (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/perturbation_selector_degeneracy.json).
- **Head choice decides collapse at about 9,890 epochs.** Q_point collapsed in 3 of 3 runs (0.003 to 0.011). Q_crps collapsed in 2 of 3. Q_laplace scored 0.187, 0.195 and 0.213 (round_leaderboards.csv, stage-launch).

## 2. NOT ESTABLISHED OR CONTRADICTED

- **"Eight identical-config replicates, 0.196 ± 0.022"** (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex L34-46, keybox L617-623; /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/1-summary.tex item 2). The round_leaderboards.csv tags show these are 8 different arms: M_sched, M_lo, M_hi, M_fine, M_coarse, M_nomix, M_off and M_gate_rezero. /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex already carries this correction (2026-09-08). Same-seed reruns agree within 0.001, so the 0.0222 is a spread across arms, not run-to-run noise.
- **"Only two rungs trained, 2.4× budget gap"** (2-expression.tex L308-313). In fact all 12 wave-6 arms ran to between 4,091 and 4,372 epochs (round_leaderboards.csv, stage-wave6). The ladder is not monotone in rank: V_basis64 0.2276, V_hadamard_add 0.2155, V_ref 0.2100, V_sink 0.2004, V_basis32 0.1925, each n=1.
- **"No maximum ever observed"** (2-expression.tex L104) contradicts L49-54 in the same section, where five of eight runs peak by epoch 4,109. Run d94cy5az also peaks at epoch 16,785 of 18,990 (round_leaderboards.csv).
- **"1,600 to 10,000 epochs moved the score from 0.204 to 0.238"** (2-expression.tex L101-104). Depth (L 4 to 6), batch size (8 to 32) and the mask objective all changed between those runs (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_012.yaml header). The leaderboard also gives 1vhu95lc as 0.1997, not 0.204.
- **"Distributional axis is not the lever"** (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.wave6-design.md §6). This was decided at ≤199 epochs in a mostly collapsed regime (v7 median 0.034, n=295), and the long-budget launch round contradicts it.
- **Hard graph mask.** Its accuracy was never measured; it was frozen on speed alone (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_010.yaml L37-41).

## 3. ERRORS AND INCONSISTENCIES

- 1-summary.tex L33 says the quoted rounds pin `split_seed`. None of the 67 rows in decoder_arms_torchcell_019_expr_v8.csv are from pinned runs: `split_seed` arrived in wave 5 (commit 727b5bbb8).
- In 10-launch.tex, L25-27 still claim the spread is "nondeterminism and nothing else", directly above the section's own correction.
- /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/launch_plan_evidence.json names this spread `replicate_spread`, and its power table rests on it. The launcher comment in /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh (~L395-410) also justifies dropping quantile from the launch round as a "known quantity" on the same basis.
- The keybox (2-expression.tex L626) says "loss and metric diverge… in opposite directions". This contradicts the correction at L145-147: only the pinball loss diverges, while squared error agrees with Pearson.
- In decoder_arms_torchcell_019_expr_v8.csv, A1_ref seed 0 (lw8zgco6) and A0_baseline wave 3 (w5ob5tku) are bit-identical (0.15759531855583192, argmax 171). One trajectory is counted twice.
- /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/tables/t-expr-epoch-budget.tex lists 16 v9 runs; round_leaderboards.csv has 70. It also compares epochs across batch 8 and batch 32, which are not commensurable.
- The ±0.0076 attached to the mixing delta has no traceable source (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-round-retrospective.md L161-170). The round_leaderboards.csv difference is −0.0022 (X_mix n=8 vs W_ref n=13), which agrees with the quoted −0.0023.

## 4. UNTRIED OR UNDER-BUDGETED (ranked)

1. **Init spread at ≥4,000 epochs on a pinned split.** Never measured, so no ladder gap is interpretable.
2. **Splits other than split 0 at long budget.** Every long run used split 0, the high draw.
3. **lr re-check after the batch went 8 to 32.** The batch change cut optimizer steps per epoch 4× (cgt_expr_012.yaml header). lr was frozen from v6 at ≤100 epochs, and S1_warmup only reached 122 epochs.
4. **Pair-rank arms and weight decay.** Each was n=1. V_basis64, V_hadamard and V_wd1e4 all peaked in their final 2% of epochs.
5. **Null sink.** Wave 3 stopped at 186 to 189 epochs with the gate shut (−4.0 to −3.92); wave 4b stopped at 300; wave 6 has n=1.
6. **Graph propagation.** A6/A4 gave the largest early seed-0 numbers (0.175, 0.172 at 140 epochs) but never ran past 140 epochs, because they were excluded on organism-transfer grounds.
7. **Graph conditions.** Mask vs KL vs no graph has never been compared on accuracy.
8. **Never run at all:** post-perturbation graph masking (C1), G1_layers6/8 (died at 0.02 h), the A5_sham null, and ProtT5 embeddings past 70 epochs.

## 5. TOP THREE RECOMMENDATIONS

Measured wall time for costing: about 42 s/epoch with 3 runs packed per A100, i.e. about 0.65 GPU-days per 4,000-epoch run (round_leaderboards.csv wave 6: 47.9 h for 4,105 epochs).

1. **A defensible number for Figure 3.**
   - **Experiment:** V_ref, 4,000 epochs, split seeds {0, 1, 2} × 2 init seeds.
   - **Control:** the kNN and linear baselines on the same three partitions, run on CPU.
   - **Expected effect:** Hypothesis (untested): the cross-split mean falls below 0.21. The post-SIMB /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json (partial run) supports this direction, with split 0 at 0.185 against 0.127 to 0.156 for the others.
   - **Cost:** about 4 GPU-days.
   - **Stop if:** the model-minus-baseline difference, paired by split, has a CI that includes 0. Then report parity and end the mechanism work.
2. **Replicate the top of the ladder.**
   - **Experiment:** V_basis64 and V_hadamard_add, 3 init seeds on splits 0 and 1, 4,000 epochs.
   - **Control:** the V_ref runs from recommendation 1.
   - **Expected effect:** Hypothesis (untested): about +0.018, which is what n=1 showed.
   - **Cost:** about 8 GPU-days.
   - **Stop if:** the paired mean is below 0.01, or its CI includes 0 after n=3 per split.
3. **lr and warmup at batch 32.**
   - **Experiment:** lr 6e-4 and 1e-3, each with warmup, 2 seeds, 2,000 epochs.
   - **Control:** lr 3e-4 on the same seeds, plus b50f93ju's 0.2019 at epoch 2,000.
   - **Expected effect:** Hypothesis (untested): the slow climb is partly an under-stepped optimizer, so the 0.20 level would come earlier.
   - **Cost:** about 2 GPU-days.
   - **Stop if:** either arm falls below 0.05 by epoch 500, or does not exceed the control by more than 0.012 at epoch 1,000.

Example runs:

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/b50f93ju

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/d94cy5az

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/sm6efleg

Files:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/1-summary.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/tables/t-expr-epoch-budget.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/round_leaderboards.csv
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/decoder_arms_torchcell_019_expr_v8.csv
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/launch_plan_evidence.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_010.yaml
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_012.yaml
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.wave6-design.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-round-retrospective.md
