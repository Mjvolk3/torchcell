# Reviewer 2 of 10: post-SIMB expression rounds (2026-10-04)

Read-only audit by an independent agent; every unmeasured statement is labeled as a hypothesis.

## 1. ESTABLISHED

- **Embedding content vs random (v10 grid).** ProtT5 beats width-matched random vectors by +0.068 (t 7.8, 16 v 16, split 0, epochs ≤990). Dropping the one stuck run gives +0.076. /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v10_grid_factorial.json `matched_all`, `matched_healthy`. This compares against random vectors, not against the incumbent's `calm` embedding.
- **v11 embedding round.** E_calm_ptt5 minus E_ptt5 is +0.030 (3 of 3 seeds). E_full minus E_calm is +0.0012 on the 2 clean seeds. Source is /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md:31,97 only; no v11 results file exists.
- **Budget.** Going from epoch 1,000 to 3,700 within the same run gains +0.0156 (sd 0.0164, 18 of 23 positive). Source /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md:36.
- **Masked objective, the only split-round contrast whose confidence interval excludes zero.** Removing it (Y_k0, v18) costs −0.0122 on the registered window statistic, sd 0.0173, 12 pairs, CI (−0.023, −0.001). It is negative on all 4 partitions, and on test it is −0.015 (3 of 11 positive). /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v18_hygiene_readout.json `concat_minus_ref.pairs[].diff_window`. This contradicts the v9 config header's expectation of no gain at k=0 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v9_mask.yaml:31-35).
- **H_concat readout.** +0.0135, 4 of 4 seeds, paired t≈10, but on split 0 only at 1,399 epochs. /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json.
- **Collapse.** The point head collapsed in 6 of 6 runs, the quantile head in 0 of 20. /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/1-findings.tex:152-155.
- **Partition dominates every arm effect.** Between-partition sd is 0.028 (v13), 0.030 (v17) and 0.020 (v18), against pooled within-partition sd of 0.010, 0.020 and 0.017. `partition` key in /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json, v17_locality_readout.json and v18_hygiene_readout.json.
- **Incumbent replicate spread.** It has never been measured at 9,900 epochs. True same-config spreads that do exist:
  - H_ref, 4 seeds at 1,399 epochs: sd 0.0076 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json).
  - R_ref, 2 seeds at ≤4,079 epochs: 0.188 and 0.169 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/mech_round_readout.json).

No lever has cleared the registered bar (+0.02, 3 of 4 partitions, test sign agreeing).

## 2. NOT ESTABLISHED OR CONTRADICTED

- **The 0.196 ± 0.022 incumbent** is the mean of the 8 arms of the v9 mask-schedule round, not replicates (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/short_budget_spread.json; /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/1-findings.tex:33-41). All 8 are seed 0 on split 0, scored by roll_max over 9,900 epochs (bias +0.007 to +0.013, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md:40).
  - Split 0's validation draw is +1.72 sd above the mean of 12 baseline draws (same note, line 35).
  - 21 of its 155 validation strains share a deleted gene with training and score 0.398 against 0.178 for the rest (same note, line 45).
- **0.238 is one draw of one arm (M_fine).** The Blom expected-maximum argument assumes iid draws, and these are 8 different arms, so it does not apply.
- **Rank-64 "best per compute" (0.2274 in 2.0 days vs 0.2362 in 3.8)** is a v8 genotype-only run against a v9 masked run: different projects, one maximum each, 42.0 vs 32.9 s/epoch (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex:150-170). The "40% of compute" figure is epochs (41%); wall-clock is 52%. At matched budgets the claim fails:
  - R_basis64: +0.008, 2 pairs, signs disagree (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/mech_round_readout.json `paired_contrasts["4079"]`).
  - H_basis64: −0.021 over 4 seeds, one collapsed (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json).
  - Wave-6 rank ladder: flat at n=1 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md:32).
- **H_concat across partitions** reads +0.011 on the window statistic or +0.007 on roll_max over 11 clean pairs (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json). It does not replicate.
- **L_prop2** reads +0.012 on the window statistic, CI (−0.009, +0.033), and +0.003 on test (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v17_locality_readout.json).
- **Pearson and ListMLE objectives** do not beat the band. Final ListMLE scores are 0.177, 0.150, 0.153 and 0.161 at about 6,000 epochs, against a band of 0.192 ± 0.018 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/listmle_round_readout.json, read 2026-09-11).

## 3. ERRORS AND INCONSISTENCIES

- **"Identical-config replicates" and Blom still asserted.** /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/1-summary.tex:189-193 and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/2-expression.tex:34-44 make the claim; /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/1-findings.tex:33-41 correct it. Also /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/10-launch.tex:36 prints 0.2010 where the JSON has 0.2008.
- **Rank-64 summary statements** repeat the uncorrected per-compute claim: /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/1-summary.tex:266-267 and /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/9-campaign.tex:206.
- **/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/4-split-rounds.tex:41.** "Eleven clean pairs +0.007" is the roll_max number inside a paragraph scored on the window statistic, which gives +0.011 (6 of 11). The fit-review note gives a third number for the same contrast, +0.0056.
- **/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/4-split-rounds.tex:26-30.**
  - The "reference-arm partition means" are actually means over all arms; the reference arm alone spans 0.107 to 0.168 in v17.
  - The 0.164 → 0.213 lift from the 90/10 fold includes the collapsed run 825on260. Paired and clean it is +0.031 (n=3).
- **Split-round JSON keys.** The summary `mean/sd/t` are computed on roll_max (`diff`), not the registered `diff_window`: v17 prop2 shows t 2.03 in the JSON against 1.28 on the registered statistic. The `concat_minus_ref` key holds L_self and Y_k0 in v17 and v18 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/v13_split_readout.py).
- **/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/2-readouts.tex has stale and overstated claims.**
  - The Pearson and ListMLE tables are the day-one partial reads; the results files hold the final ones.
  - Line 357 calls H_concat "resolved", which v13 did not bear out.
  - Lines 362-364 claim "embedding effect seen again" from 0.174 vs 0.167, which is inside the v9 band sd of 0.0117.
- **/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/0-strand.tex:139-142.** It says 0.7799 is "scored on the matrix it was fit to". In the JSON that value is `ceiling_train_basis` (basis fit on training strains); the self-fit value is 0.928 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/lowrank_output_ceiling.json).
- **/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/1-findings.tex:157.** The 0.0005 head gap compares runs of up to 18,990 epochs against runs of at most 9,896 under an order statistic, so the budgets are not matched.
- **No v11 results file exists, but v12 to v16 pinned E_full on it.** This breaks the project's rule that every artifact comes from a committed script in the experiment folder.

## 4. UNTRIED OR UNDER-BUDGETED (ranked)

1. Graph adjacency of the deleted gene as model input. Hypothesis (untested): it lifts the model. A parameter-free graph-neighbor average (B4) already matches the model on gene-disjoint strains: 0.165 to 0.194 val against CGT 0.100 to 0.178, n=2 dumps (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md:118).
2. Perturbation-derived embeddings (SGA interaction, chemogenomic profiles). There is zero coverage, and none has passed the CPU gate.
3. Learning-rate warmup and schedule. Every high-learning-rate arm started cold (same note, line 28).
4. Per-strain magnitude head. Its oracle bound is +0.071 (same note, line 44).
5. Reading the v13 and v14 best-validation checkpoints on test. They are pulled and unread.
6. Self-indicator injected before the encoder.

## 5. TOP THREE (under 20 days)

**A. Graph-adjacency input (union SVD-128) vs L_ref, keeping the masked objective.**

- Design: 4 partitions × 3 seeds, 1,200 epochs, test read on gene-disjoint strains.
- Hypothesis (untested): ≥ +0.02.
- Cost: about 12 card-days, from v12's measured 1,400 epochs in 1d10h to 1d17h at four runs per card (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/2-readouts.tex:311-316).
- Stop if the paired window gain is under +0.02, or if B4 alone matches the arm on test in 3 of 4 partitions. Figure 3 then reports CGT as being at parity with B4.

**B. Final-configuration replicate set (ref + mask + L_self + L_prop2) vs L_ref.**

- Same template, reported against B2, B3 and B4 on test.
- Cost: about 12 card-days.
- Stop after one round regardless of outcome, because this set is the Figure 3 number.

**C. Two CPU steps first, zero GPU.** Read the unread v13/v14 test checkpoints, and run the SGA-profile retrieval gate (Spearman ρ > 0.02 against ProtT5's +0.017).

- Only if the gate passes: one GPU arm, about 12 card-days.
- Stop if ρ ≤ 0.02.

Full paths:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v18_hygiene_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v17_locality_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/mech_round_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes/experiments.019-simb-multimodal.expression-fit-review.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal/sections/1-summary.tex
