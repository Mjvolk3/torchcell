---
id: iplas96fdgd1kg2qgq7wlbp
title: 043-expression-proteome-morphology
desc: ''
updated: 1792393327949
created: 1792393327949
---

## 2026.10.19

- [x] Experiment 043 stage 1: one 15-dataset expression, proteome and morphology query over KG 4.0, registered as a supported query and validated on `2026.10.17-4295f7ca`, with a dev-tree census of all 173,805 records [[experiments.043-expression-proteome-morphology]]
- [x] The raw store build (job 5851) fails in `Neo4jQueryRaw.extract_systematic_gene_names` on the Boocock and Nguessan segregant genotypes; needs a torchcell fix or an owner decision before a rebuild [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
- [x] Fixed the raw stage for genotypes without a perturbation list (segregants), kept the zero-perturbation isolate, and rebuilt the 043 raw store: 173,617 samples, 188 short of the served 173,805 because the `all` gene clause drops records naming genes outside the genome gene set [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
- [ ] Owner decisions from the rebuild: the 188 gene-set drops, the doubled `expression_log2_ratio` label in the Perturbation processor, and the perturbation-count index that omits every segregant [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
