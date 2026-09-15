---
id: 9dg7pu4ifig81llpd5093qy
title: Build_fig3_proteome
desc: ''
updated: 1789443181445
created: 1789443181445
---

## 2026.09.15 - Building fig3_proteome on the GilaHyper served store

`experiments/019-simb-multimodal/scripts/build_fig3_proteome.py` builds `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_proteome` from `queries/fig3_proteome.cql` (Kemmeren 2014, Sameith 2015 Sm and Dm, Messner 2023; all `graph_level = 'node'`), with `ProteinAbundanceLog2RatioConverter` in the conversion step, then the mean deduplicator, the genotype aggregator and the `Perturbation` processor, the same pipeline as `fig3_core`. The query file is read from the script's own `queries/` directory so the build runs from a worktree before the file lands on `main`; the training script reads the same file under `EXPERIMENT_ROOT`.

The served store is the GilaHyper `tc-neo4j-readonly` container (`TC_NEO4J_URI`, default `bolt://localhost:7687`); the URI and host are recorded in `results/fig3_proteome_build_census.json` with the record counts per label, the genotypes carrying both proteome and expression, the raw-versus-converted record count (the converter's base `process()` skips a record it cannot convert, so an unequal count is the failure), and the log2 range of the first proteome record.

Related: [[experiments.019-simb-multimodal.conf.cgt_expr_v14_proteome]], [[torchcell.datamodels.protein_abundance_log2_ratio_conversion]], [[experiments.019-simb-multimodal.scripts.make_split_indices]] (`--dataset-tag fig3_proteome --label protein_abundance`), [[experiments.019-simb-multimodal.scripts.expression_baselines_split]] (same two flags).
