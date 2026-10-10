---
id: yuqliky1g8ob12wwqivbcz2
title: Cgt_morph_v24
desc: ''
updated: 1791623053392
created: 1791623053392
---

## 2026.10.10 - The morphology round v24

Two questions in the author's order: can the model predict single-deletion morphology from the genotype, and does a revealed gene-level modality help. Target: the 116 moving CalMorph features (`calmorph_moving_features.yaml`), Yeo-Johnson normalized, scored as `val/morphology/pearson_per_feature`. Readout: the CLS token after the perturbation operator (`model.perturb_cls: true`, ported from the 025 line into `torchcell/models/equivariant_cell_graph_transformer.py`; the test `test_perturb_cls_makes_the_global_readout_strain_specific` shows the gene rows unchanged and the CLS strain-specific), with `use_gene_pool: false` so the head reads the cell state alone. Store `fig3_core` (4,695 calmorph genotypes, 1,554 with expression). Trunk and recipe from v22: ProtT5 only, width 90, six layers, batch 128, lr 3e-4, 1,200 epochs, no mask schedule.

Arms (`gh_expr_008_arm.sh`): `M_cls` genotype only on every calmorph genotype; `CM_expr` expression revealed in full (the v20 conditioning step) on the genotypes carrying both labels; `CM_exprperm` the permuted control; `M_pool` the CLS plus the mean gene pool. Launched 2026.10.10 as IGB stages `morph_v24` (cabbi, 36 tasks, one run per card) and `morph_v24_pool` (gpu, 3 tasks). The proteome-revealed arm waits on a store joining Messner 2023 with Ohya 2005.
