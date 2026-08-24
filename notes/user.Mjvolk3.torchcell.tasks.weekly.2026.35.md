---
id: tqzn9662f7ym3zsbuuj8jk3
title: '35'
desc: ''
updated: 1787586795736
created: 1787586795736
---

## 2026.08.24

- [x] **Build list for the first trigenic round is final: 45 strains, 25 doubles + 20 triples, 65 on plate.** Bench sheet [[experiments.W019-echo-crispr-array.build-list]], PDF `notes/assets/pdf-output/experiments.W019-echo-crispr-array.build-list.pdf`, rationale [[experiments.W019-echo-crispr-array.next-strains-to-construct]]. Every strain builds in parallel; T01-T15 have one parent route each so start those first
- [x] Chose `capped` over five alternatives because it is the only design that holds exposure to the two zero-trigenic-data genes (YER079W, YLR312C-B) to a set level, 6 of 20, while still covering all eleven panel genes; it costs 45 strains against 40 for `rank` [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]]
- [x] Corrected the target basis from 10 genes / 31 targets to 11 / 39: YLR104W (LCL2) is both built and a panel-12 prediction node, and the frozen TEN set predated run 4's decision to keep LCL2 [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]]
- [x] Audited the whole round with an independent checker that re-derives inventory, basis and selection from the pinned inputs without importing the design script: **52 checks, 0 fail**. The load-bearing one is tau-computability, since a triple yields nothing unless all three of its doubles and all three of its singles are on the same plate [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]
- [x] Found and fixed a real error in both build notes: 17 new doubles are still needed if YER079W and YLR312C-B are dropped, not 18, and the clean core is 31 strains rather than the 32 stated or the 34 also claimed. `YLR104W + YPL081W` serves only rank 3 and is orphaned by the drop [[experiments.W019-echo-crispr-array.build-list]]
- [x] Fixed the Kuzmin replicate-reproducibility numbers in the run-4 handoff to the SI text rather than a figure read: digenic eps and raw trigenic r = 0.90-0.91, adjusted trigenic tau r = 0.74-0.81, sha256-pinned. The old 0.59 for adjusted tau understated their reproducibility by 0.15-0.22 [[experiments.W019-echo-crispr-array.run4-handoff]]
- [x] Labeled the per-gene digenic column as not training data; it comes from `dmi_kuzmin{2018,2020}`, which `001_small_build.cql` excludes, and the trigenic counts were read from the LMDBs rather than quoted [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [x] Deprecated the 10-gene / 31-target artifacts, 19 files, to `/scratch/projects/torchcell-deprecated/2026-08-24_104110__*/` with per-file provenance manifests [[experiments.W019-echo-crispr-array.session-handoff-2026-08-23]]
- [ ] **Settle YLR312C-B in or out before starting the starred strains** (D07, D10, D11, D13, D19, D22, D23 and T01-T06). SGD "ORF, Merged", no discrete protein, zero trigenic training data, 0/10 significant in-panel digenics; the swap to SPH1 was recommended and never actioned #high [[experiments.010-kuzmin-tmi.inference-dataset-3]]
- [ ] **tau is not callable at current replication.** SE(tau) ~ sqrt(7)*s and at 2 picks / 3 plates s ~ 0.121, so the smallest callable |tau| is 0.63 against Kuzmin's 0.08 / 0.12 / 0.20 thresholds and their median SE(tau) of 0.031. 0 of 39 targets clear our floor; closing it is a pick/plate problem, not a model problem #high [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [ ] Decide the round order: the pick-replication round (26 strains / 57 tubes) and this triple round (65 strains / 65 tubes) compete for the same plate. Recommended is replication first, constructing these 45 during it #high [[experiments.W019-echo-crispr-array.build-list]]
- [ ] Check the 0.43 calibration slope against the inference parquet; it assumes mu_pred ~ mu_y and one read would settle it #medium [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [ ] Record which variance decomposition gives the 6-plate floor of 0.55; scaling s by 1/sqrt(2) from 3 to 6 plates gives 0.44, and a nested model where only the plate term shrinks is the more careful assumption but is not shown #low [[experiments.W019-echo-crispr-array.session-handoff-2026-08-23]]
- [ ] If anyone retries `YKL033W-A x YJR060W`, record the attempt date and outcome; the original failure has no recorded date, and it blocks 0 of the 39 targets so it stays off the list until then #low [[experiments.W019-echo-crispr-array.build-list]]
