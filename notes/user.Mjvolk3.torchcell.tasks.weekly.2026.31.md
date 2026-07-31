---
id: 3ctlcwkeiclelxddckv4lyy
title: '31'
desc: ''
updated: 1785532549470
created: 1785532549470
---

## 2026.07.31

- [x] Established that the 019 expression round contains no valid arm comparison: all 67 scored runs stopped at or below 300 epochs while the curve is still climbing past epoch 1,900, so every mechanism verdict is void rather than null [[experiments.019-simb-multimodal.expression-round-retrospective]]
- [x] Replaced the graph-regularization KL with hard edge masking as the frozen default, which is 1.5-1.7x faster per epoch because the KL needs attention weights and so forces a materialized [1,9,6608,6608] matmul that blocks the fused SDPA kernel [[torchcell.models.equivariant_cell_graph_transformer]]
- [x] Built the pair-rank decoder ladder and Hadamard perturbation operator to test how much (perturbation, gene) interaction the architecture can express, after measuring that the operator is exactly degenerate on the 95.4% of the build with a single perturbation [[experiments.019-simb-multimodal.scripts.perturbation_selector_degeneracy]] [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]]
- [x] Added the teacher-forced masked-label objective, with revealed genes taking exactly zero gradient so the model cannot be rewarded for copying its input to its output [[experiments.019-simb-multimodal.scripts.verify_masked_objective]]
- [x] Showed the masked-conditioning gain is orthogonal to genotype -- 97.5-100.6% of it survives removing a genotype predictor -- so the objective optimizes a channel that is switched off at the point we score [[experiments.019-simb-multimodal.scripts.conditioning_gain_after_genotype]] [[experiments.019-simb-multimodal.scripts.cross_study_conditioning_oracle]]
- [x] Rebuilt the scoring apparatus after three pooling bugs each silently voided numbers, adding eval-mode train metrics and best-by-metric checkpointing; the latter matters because best-by-loss lands 1,400 epochs earlier than best-by-metric [[experiments.019-simb-multimodal.scripts.score_decoder_arms]] [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]]
- [ ] Wire `ckpt_path` resume into `trainer.fit` so runs can chain past a walltime kill; without it IGB wave 6 is capped near 4,000 epochs and cannot be extended, since we have no root there #high [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]]
- [ ] Pre-adapter cleanup + canonical rebuild for the β-carotene (Ozaydin 2013) and betaxanthin (Cachera 2023) datasets before BioCypher adapters: fix stale metabolite-verifier test, add sha256 verification to both loaders, document the intentional Cachera gene-drop (AAD6/CRS5/FLO8), add build-smoke tests, rebuild both stale LMDBs, promote the adopted cassette design in the notes #high [[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]
- [ ] Commit the gilahyper phenotype/gzip script so the phenotype-dataset and integrated-graph tables come from a script instead of a hand transcription; they are the last numbers in the paper without a generating script, and one of them is sourced from a scratch note #high [[paper.information-accounting]]
- [ ] Re-examine the SVR interaction fits at random (d=1000), where a CV s.d. of 0.383 against a mean of 0.458 sits in the same cell that produced a diverged MSE #medium [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [ ] Reconcile the paper-facing classical-ML plot script with the new figure standard, since its PNG output conflicts with the palette SVG route that now feeds the classical-ML figure #medium [[experiments.smf-dmf-tmf-001.traditional_ml-plot_paper]]
- [ ] `#73` caudal2024 `SACE_`-prefixed FASTA headers -- silent `continue` would drop an isolate's entire genotype; latent today (none of the 93 fall in the built 943)
- [ ] Reconcile stale curated genotype counts the generator surfaced (Kuzmin 2020 dmf 632,797 vs 256,862; Kemmeren 1,450; Zelezniak metabolome built) into [[paper.supported-datasets-and-databases]]
- [ ] Implement CI/quality foundation -- ruff (E,F,I,UP,D), runnable mypy (defer 7,601-error cleanup), pytest/coverage config, repair broken `src/`-path CI; supersedes `feat/literature-zotero-ocr` tooling [[plan.ci-foundation-ruff-mypy-pytest.2026.06.18]]
- [ ] Make the pytest CI job blocking (`#16`): harden 4 CI-fragile tests (test_s288c module-level DATA_ROOT guard, wall-clock benchmark `@pytest.mark.gpu`, targeted DATA_ROOT skipif, filelock cleanup rewrite) + CPU-only wheel install (torch/scatter from CPU indexes) + remove `continue-on-error`, then add `pytest-coverage` to main's required checks post-merge [[plan.pytest-ci-blocking.2026.07.01]]
