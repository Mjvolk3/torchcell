---
id: ogvbzxpeltnot5eyaykjuzv
title: '37'
desc: ''
updated: 1788913645350
created: 1788913645350
---

## 2026.09.08

- [x] 010 positive-panel report reviewed and corrected: the 0.877 calibration slope had no generating script and reproduces exactly on the whole test split while the nominated tail runs at 0.24; "two thirds real calls" was magnitude-only (2 of 18 under the Kuzmin call); rescue ordering is 82 percent published rungs; three 20-strain designs tie on the median objective ([[experiments.010-kuzmin-tmi.scripts.calibration_on_test_split]], [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_rescue_decomposition]])
- [x] Query-pair-disjoint arm had never run: job 1599 died on `plot_edge_recovery_every_n_epochs: 0` (fixed b0b05419a). Launched the matched pair 1607 (random) and 1608 (disjoint), which differ only in the split file. Not the Table 10 row of the additive-baselines report: different partition (331/43/46 vs 285/67/68 query pairs), single split vs five-fold, mask arm vs KL
- [x] Draft figure "not what you learn, how you learn": the graph penalty never reaches the prediction (inference swap 1e-6) and decides whether training happens (lambda = 0 fails; hard mask, the lambda to infinity limit, reaches 0.31 at epoch 1 then predicts the mean; soft KL 0.446). Candidate for SI `fig:graphreg-empirical` [[experiments.025-solid-growth.scripts.graph_regularization_how_not_what]]
- [x] Mock-up `notes/assets/drawio/FigS-graph-regularization-sweep.drawio` (+ `.drawio.png`, generator `.gen.py`): the λ sweep with no-penalty and hard-mask as the extremes (panel b), gradient budget at probe epochs (c), discovery defined on graphs already in hand (d: regulatory head vs TFLink-only edges), overruling defined with Costanzo digenic ε (e), degree-matched random-graph control (f), and the Delta design strip (24 runs, ~2,016 GPU-h). Headless export recipe recorded in memory `drawio-headless-export-gilahyper`
- [x] Delta package for the graph-regularization sweep on the 010 build (the 025 full build is 3.2 TB; the 010 build is 1.5 GB with bit-identical labels): configs `cgt_010b_r_kl_005` / `r_mask_006` / `q_kl_007`, 010-native index artifacts, `delta_cgt.slurm`, preflight, sweep submitter, and the GilaHyper to Delta sync ([[experiments.025-solid-growth.scripts.delta_cgt]], [[experiments.025-solid-growth.scripts.make_010build_index_artifacts]])

## 2026.09.09

- [x] The four panels of the 010 additive-baselines report brought to the repo figure standard (panel letters at the outer top-left, framed legends, palette-order series, palette density ramp with a shared colorbar) and the PDF rebuilt; `panel_label` and the framed-legend rcParams added to `torchcell.utils` ([[experiments.010-kuzmin-tmi.scripts.additive_baseline_gene_interaction]], [[experiments.010-kuzmin-tmi.scripts.paired_prediction_agreement]], [[experiments.010-kuzmin-tmi.scripts.inference_run_consistency]], [[experiments.010-kuzmin-tmi.scripts.query_pair_disjoint_cv]])
- [ ] Decide: cancel job 1608 (hard mask on the disjoint split; the mask arm already collapses on the random split, so 1608 cannot answer the split question) and queue `cgt_s0_q_kl_004` (soft KL on the disjoint split) in its place
- [ ] Delta canary before the sweep: one KL run to measure minutes per epoch with zero dataloader workers (GH RTX 6000 Ada: 19.4 min/epoch KL, 14 min mask; IGB: 58 min/epoch KL, 18 min λ = 0)
- [x] Delta package for the graph-regularization sweep on the 010 build (the 025 full build is 3.2 TB; the 010 build is 1.5 GB with bit-identical labels): configs `cgt_010b_r_kl_005` / `r_mask_006` / `q_kl_007`, 010-native index artifacts, `delta_cgt.slurm`, preflight, sweep submitter, and the GilaHyper to Delta sync ([[experiments.025-solid-growth.scripts.delta_cgt]], [[experiments.025-solid-growth.scripts.make_010build_index_artifacts]])

## 2026.09.09

- [x] 010 positive-panel report landed and published to Zotero for review (`torchcell/notes-tex/010-positive-panel`, version `2026-09-09-19-00-02_2ba91614`). Sections 10 to 12 are the new material: why regulation acting on metabolism, the `inference_4` ranking, and the 20-strain pyruvate-node panel with its measured-overlap diagnostic ([[experiments.010-kuzmin-tmi.scripts.inference_4_panel_design]], [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_score]], [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_overlap]], [[experiments.010-kuzmin-tmi.scripts.inference_4_measured_diagnostic]])
- [x] Every number in the new sections re-checked against its result file: panel scores, the stratum table, the rescue decomposition (82.4 percent measured share), the engineering axis subsystems, and the Marquez-Zavala quotes against the mirror at sha256 `952805b4`. All reproduce
- [x] Section cross-references converted to `\label` / `\ref`. Inserting sections 10 and 12 had shifted the numbering, so the abstract pointed at Section 13 for a panel that renders as Section 12, and three more references were off by the same amount
