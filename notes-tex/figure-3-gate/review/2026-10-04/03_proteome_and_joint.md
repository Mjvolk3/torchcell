# Reviewer 3 of 10: proteome and joint rounds (2026-10-04)

Read-only audit by an independent agent; every unmeasured statement is labeled as a hypothesis.

Path legend: WT = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective ; R = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results ; S = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts ; REV = /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review

## 1. ESTABLISHED

- The v19 readout matches the registration: windows 200 to 400 and 1,000 to 1,199, paired over split, only the 11 partitions with all three arms at 1,199 included, all 36 runs have seed 0 (W&B config) (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/joint_checkpoint_readout.py:52, 277-281).
- H1a joint minus single, expression: -0.0012, SE 0.0091, 5/11 positive, p 0.55. H1b proteome: -0.0172, SE 0.0059, 1/11, p 0.64. Neither rejects (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json `v19.tests`). The /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/6-checkpoint.tex conclusion is supported.
- The "delay" reading is supported: rolling maximum +0.0018 (5/11), and (my W&B query of first epoch with `val/proteome/pred_sd_ratio` > 0.05) the joint proteome head launches at epochs 115 to 177 against 72 to 94 single-head (12 runs each). All v19 heads passed the launch guard; no relaunch was needed.
- `_conditioned_step` conditions validation and test (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py:1459-1461, 1581-1587); train-only standardization (same file :657-835); the revealed head is removed from loss and metrics (:1438). W&B: about 1,795 proteins revealed in val for C_expr, 6,127 genes for C_prot. Forward passes are equal: 2 per step in K (masked step with schedule [0]) and C (:1254-1283, :1417-1431).
- v14: concat minus ref +0.0008 (sd 0.0057, 5/8) on the rolling maximum; NMSE minimum at epochs 46 to 147, NMSE at end 1.10 to 1.15, worse than the mean predictor (16 runs; /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v14_proteome_readout.json).

## 2. NOT ESTABLISHED OR CONTRADICTED

- The proteome window premise "peaks at a median epoch of 140 to 259" (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v19_joint_clean.yaml:44) fails on the 1,349-strain store: K_prot peaks at a median epoch of 417 (151 to 1,190, n=11), 4/11 after epoch 900 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json `runs`).
- "Proteome head budget-saturated at 500" (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/5-joint-plan.tex) is contradicted by the finished v16 continuation (computed by me from W&B, in no committed readout; Appendix A): J_ref_s1 peaks at 1,028 and 1,124 and gains +0.015 and +0.018 from window 200-400 to 1,000-1,200 (n=2). Joint minus ref, rolling maximum over 1,200 epochs: -0.0118, 0/6.
- Registered secondaries are absent: K_perm (never run), the test score at each head's own checkpoint, reliability-weighted proteome Pearson, NMSE (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/joint_checkpoint_readout.json).
- No reliability-restricted metric has ever been computed (proposed as E4, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/01_data_ceilings.md:384). Hypothesis (untested): the decay is fitting the 28% zero-reliability proteins (same file :83).
- The v20 header cites "-0.006 / -0.018" and a single-head peak at "380" (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v20_conditioned.yaml:12, 32). The 11-partition readout gives -0.0012 and -0.0172; 380 is the median only with the incomplete split 11 run (peak epoch 103) included.

## 3. ERRORS, BUGS AND INCONSISTENCIES

- [HIGH, live v20] /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh:695 sets `persistent_workers=true` for `L_*|Y_*|J_*|K_*` but not `C_*`. W&B confirms False on all 8 v20 runs and True on all 36 v19 runs. This is the 60 s file-lock crash mode that killed 4 v15 runs; v20 also runs about 36 epochs/h (157 in 4.4 h) against 60 to 87 for v19 (with 4 against 3 runs per card).
- [MEDIUM, framing] The v20-against-K comparison is not like for like: different head slot (`per_gene_aux` against `per_gene`), parameters 6,681,548 against 6,671,629 (a shifted init draw), different loader setting, encoder input [0,0] everywhere in K against [value,1] in C. Make C minus Cperm, identical except for the input, the primary contrast.
- [MEDIUM] The "zero input is identity" claims (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py:1190; encoder docstring /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py:962-976) are false: `proj([0,0])` = W2 ReLU(b1) + b2, a learned offset on every unobserved token.
- [LOW] The permutation is `roll(1)` inside each batch (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py:1426). In val and test (no shuffle) the partner is fixed to the adjacent record, and a size-1 batch would get its own labels. Missing proteins (NaN) look identical to genes not on the panel (:1428).
- [LOW] The key `concat_minus_ref` holds wd1e2 minus ref and J_joint minus J_ref (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v15_wd_readout.json, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v16_joint_readout.json).
- [PROVENANCE] 0.226 and 0.218 have no committed script (part2.json is not in the repo); the triangle table mixes means (gene sides) with medians (morphology sides).

## 4. UNTRIED OR UNDER-BUDGETED (ranked)

1. A per-partition ridge bar, refit on each partition's train rows and scored on the same val rows and features.
2. The reliability-restricted proteome metric.
3. Joint on the union store against an undiluted expression-only control. The v16 continuation reaches 0.118 to 0.130 on the expression window (6 runs) against v19 K_expr 0.069 to 0.117 on splits 0 to 2 (unpaired, different val sets).
4. Test scores from the saved checkpoints.

## 5. TOP THREE RECOMMENDATIONS

- A. Finish v20 with C minus Cperm as the primary contrast, persistent workers on for the remaining tasks, proteome arms stopped at 600 epochs. Control: Cperm, and the genotype-plus-observed ridge bar of 0.236 (expression) and 0.214 (proteome) (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/01_data_ceilings.md:144-147). Hypothesis (untested): C_prot minus C_protperm about +0.08 (partial at epoch 157: +0.087, +0.085, n=2). Cost about 12 GPU-days (extrapolated from 4.4 h per 157 epochs). Stop if C minus Cperm < 0.02 at 6 partitions or C stays below the per-partition ridge.
- B. Zero-GPU readouts: per-partition ridge bar, reliability-restricted metric, test score per head, and a committed v16 continuation readout. About 0.5 GPU-day of checkpoint evaluation. Stop if the reliability-restricted contrast is also null.
- C. Price the data-quantity term: joint on the union store against expression-only on the 1,554 expression strains, 6 partitions, 1,200 epochs. Hypothesis (untested): about +0.02 on expression. Cost about 8 GPU-days (assumes about 3.3x v19 steps per epoch at 14 to 20 h per v19 run). Stop if the difference is < 0.01 or the control fails the launch guard.

W&B example runs:

https://wandb.ai/zhao-group/torchcell_019_prot_v19/runs/olkikplr

https://wandb.ai/zhao-group/torchcell_019_prot_v20/runs/7tiaq7pe

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/94437cq7

## Appendix A. v16 continuation (019-v16w2-joint), computed by this reviewer

How it was computed: a read-only W&B public API query of project zhao-group/torchcell_019_prot_v16. For each continuation run (config `wandb.resumed_from` set), the history of `val/<head>/pearson_per_feature` was concatenated with its source run's history; on a duplicated epoch the continuation's value was kept, and the last logged value per epoch was used. w200_400 and w1000_1200 are plain means over those epoch ranges, inclusive. rollmax is the maximum of a centered 5-epoch rolling mean, an upward-biased order statistic, and rmax_ep is its epoch. One row per run and head; 18 runs (3 splits x 2 seeds x 3 arms), each at epoch 1,199 or 1,200. No script was committed (read-only audit); the scratch output was /private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/0e7a8bd2-1c56-433f-848a-c25de6ee5824/scratchpad/v16w2.csv.

| arm | seed | head | last | w200_400 | w1000_1200 | rollmax | rmax_ep |
|---|---|---|---|---|---|---|---|
| J_expr_s0 | 0 | expression | 1200 | -0.0001 | -0.0001 | 0.0246 | 388 |
| J_expr_s0 | 1 | expression | 1200 | 0.0872 | 0.0887 | 0.1054 | 230 |
| J_expr_s1 | 0 | expression | 1200 | -0.0006 | -0.0000 | 0.0199 | 1039 |
| J_expr_s1 | 1 | expression | 1200 | 0.0007 | -0.0005 | 0.0207 | 641 |
| J_expr_s2 | 0 | expression | 1200 | 0.0731 | 0.0384 | 0.0927 | 282 |
| J_expr_s2 | 1 | expression | 1199 | 0.0013 | -0.0006 | 0.0257 | 349 |
| J_joint_s0 | 0 | expression | 1200 | 0.0633 | 0.1296 | 0.1395 | 1187 |
| J_joint_s0 | 1 | expression | 1200 | 0.0699 | 0.1291 | 0.1363 | 1146 |
| J_joint_s1 | 0 | expression | 1200 | 0.0634 | 0.1211 | 0.1305 | 607 |
| J_joint_s1 | 1 | expression | 1200 | 0.0902 | 0.1304 | 0.1371 | 1119 |
| J_joint_s2 | 0 | expression | 1200 | 0.0476 | 0.1294 | 0.1377 | 1171 |
| J_joint_s2 | 1 | expression | 1199 | 0.0548 | 0.1179 | 0.1251 | 1021 |
| J_joint_s0 | 0 | proteome | 1200 | 0.0988 | 0.0923 | 0.1129 | 253 |
| J_joint_s0 | 1 | proteome | 1200 | 0.0742 | 0.0986 | 0.1075 | 1054 |
| J_joint_s1 | 0 | proteome | 1200 | 0.0606 | 0.0750 | 0.0864 | 780 |
| J_joint_s1 | 1 | proteome | 1200 | 0.0546 | 0.0638 | 0.0698 | 1147 |
| J_joint_s2 | 0 | proteome | 1200 | 0.0661 | 0.0712 | 0.0849 | 678 |
| J_joint_s2 | 1 | proteome | 1199 | 0.0796 | 0.0839 | 0.1038 | 437 |
| J_ref_s0 | 0 | proteome | 1200 | 0.0925 | 0.0837 | 0.1155 | 181 |
| J_ref_s0 | 1 | proteome | 1200 | 0.0769 | 0.0756 | 0.1116 | 136 |
| J_ref_s1 | 0 | proteome | 1200 | 0.0750 | 0.0897 | 0.0950 | 1028 |
| J_ref_s1 | 1 | proteome | 1200 | 0.0668 | 0.0847 | 0.0890 | 1124 |
| J_ref_s2 | 0 | proteome | 1200 | 0.0908 | 0.0884 | 0.1112 | 202 |
| J_ref_s2 | 1 | proteome | 1200 | 0.0838 | 0.1018 | 0.1135 | 93 |

Derived paired contrasts (same split and seed, n=6): J_joint minus J_ref on the proteome is -0.0087 on window 200-400, -0.0065 on window 1,000-1,200 (2/6 positive), and -0.0118 on the rolling maximum (0/6). Four of six J_expr runs never launched (window values near zero), so J_joint minus J_expr on expression is confounded, as documented in /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/review/2026-09-27-joint-review/ADDENDUM_0207.md.
