---
id: 91hbba8c1ug487r4q0wtoh0
title: '37'
desc: ''
updated: 1788904951158
created: 1788904951158
---

## 2026.09.08

- [x] **The v10 generalization-gap grid on Delta is read out: embedding content is the only factor that moves the score, +0.068 at three times the replicate spread; trunk, readout and weight decay are nulls at about 0.02.** 32 runs at a matched budget of epochs <= 990, three of eight array tasks having hit the two-day wall. Neither grid level is the incumbent's `calm` embedding, by design, so the grid ranks content and not the incumbent's choice [[experiments.019-simb-multimodal.scripts.v10_grid_factorial]]
- [x] One grid run in 32 never learned (cell 1 seed 0 at 0.019 while its twin is the best run at 0.169); its cause is not measured and both readings, with and without it, are recorded [[experiments.019-simb-multimodal.scripts.v10_grid_factorial]]
- [x] **The mechanism round is read out at epochs <= 4,079: the pair term alone is a null with sign-disagreeing pairs, the per-gene readout arms average +0.027 over `R_ref` with the combined arm positive at both seeds, under the round's ~0.06 resolution.** The per-gene arms peak early and give part of it back by 8,500 [[experiments.019-simb-multimodal.scripts.mech_round_readout]]
- [x] All four packed five-day mechanism tasks died in the cgroup out-of-memory handler at 61 GB host RSS against 60 GB requested, one run per task; the Pearson-round packed tasks read 21 GB at day 1.9. Cause not identified; the `file_system` sharing strategy is the unverified suspect [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] Corrected the strand record: `vqek7ali` and `3qy1rh0o` are `R_pergene` and `R_pergene_basis64` seed 1, not quantile replicates; every mechanism-round `R_ref` run is tagged `stage-wave4b` because a wave-4b `case` branch shadowed the wave-5 one, now removed [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] **The metric-aligned round at day two: five of six batch-32 runs collapsed to constant outputs (pure Pearson by epochs 584 to 680 at every seed, the MSE-anchored arm at two of three), while the loss kept reading 0.56 because it drops constant columns as invalid.** The two solo batch-64 pure-Pearson runs are alive at 6,050 epochs on the incumbent band (0.194, 0.186 vs 0.192 +/- 0.018), not above it; batch size and packing are confounded there [[experiments.019-simb-multimodal.scripts.pearson_round_readout]]
- [x] **New notes-tex document `019-simb-multimodal-expression`: the expression strand after the sprint, claim by claim with script and sample size, the 2026-09-08 readouts, and the open decisions.** The readouts section moved out of the six-strand retrospective, which stays as reviewed with its design and launch sections; `make check` clean on both [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] **Correction: the eight "identical-config replicates" behind 0.1965 +/- 0.0222 are the eight v9 mask-schedule arms (`M_sched`, `M_lo`, `M_hi`, `M_fine`, `M_coarse`, `M_nomix`, `M_gate_rezero`, `M_off`), verified in their W&B configs; the leaderboard's config columns cannot see mask schedule, mixing or gate.** The spread bounds nondeterminism from above; within-round contrasts stand; 0.2382 is one draw of `M_fine`, and `M_off` with no masked objective scores 0.2008 inside the same spread [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] `wandb_run_index.py` links all 74 runs behind the expression document to their W&B pages, generated from the same result files the sections read [[experiments.019-simb-multimodal.scripts.wandb_run_index]]
- [x] Both 019 notes-tex documents renamed for their directory (`019-simb-multimodal.pdf`, `019-simb-multimodal-expression.pdf`), matching main's convention [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] `loss_min_vs_pearson_peak.py` now writes a stable figure name and excludes the Pearson-objective arms, whose loss and metric coincide by construction; the 23-run medians are unchanged (loss minimum at 481, Pearson peak at 2,488) [[experiments.019-simb-multimodal.scripts.loss_min_vs_pearson_peak]]

## 2026.09.09

- [x] **Expression document retitled "Knockout Expression" and made self-contained**: new section 0 carries the task, model, pair-term degeneracy, head and decoder-family tables, baselines, GEARS/State/benchmark reading, oracle and campaign arithmetic over from the SIMB retrospective; SIMB document untouched [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] Answered "did we compare heads / GEARS": distributional heads yes (objective round); decoder families (GEARS cross-gene, bilinear, Perceiver, basis, concat) only at 50 to 276 epochs in v8, never a comparison; new generated table [[experiments.019-simb-multimodal.scripts.decoder_arms_table]]
- [x] Correction: W&B `finished` on offline-synced runs means synced, not ended; all seven surviving Pearson/ListMLE runs still RUNNING at 18:10 CT, document and note corrected [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] ListMLE round day-one readout (`--round listmle`, Spearman columns): no collapse in pure arms, 0.150 to 0.168 at 1,800 to 2,400 epochs, one seed inside the band [[experiments.019-simb-multimodal.scripts.pearson_round_readout]]
- [x] CLS is wild-type for every strain (measured sd 0.0 in the code comment); three encoder-side perturbation designs written up as untested options, incl. typed perturbation tokens for environmental perturbations [[experiments.019-simb-multimodal.expression-strand-retrospective]]
- [x] Loss-minimum figure: clipped x labels fixed (58 mm); ListMLE figure on the standard; document 17 pages, `make check` clean [[experiments.019-simb-multimodal.scripts.loss_min_vs_pearson_peak]]
- [ ] Launch discussion: Delta `bbub` 1,000-epoch arrays (calm vs ProtT5; R_ref vs R_pergene, 3 seeds each), one batch-32 solo Pearson seed on cabbi when the batch-64 tasks end
