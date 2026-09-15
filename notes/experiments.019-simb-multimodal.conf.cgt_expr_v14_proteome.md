---
id: 9kwyizzi0iuumr25mb5z7k4
title: Cgt_expr_v14_proteome
desc: 'v14 proteome round config: the v13 split design on the Messner 2023 knockout proteome'
updated: 1789443188803
created: 1789443188803
---

## 2026.09.15 - The proteome round: v13's design on the Messner knockout proteome

Config: `experiments/019-simb-multimodal/conf/cgt_expr_v14_proteome.yaml`, inherits [[experiments.019-simb-multimodal.conf.cgt_expr_v13_split]]. Launcher stage `proteome` in [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]]; arms `P_ref_s<k>` and `P_concat_s<k>` in `gh_expr_008_arm.sh`.

**What changes against v13, and nothing else.**

- Data: `fig3_proteome` (query `queries/fig3_proteome.cql`, built by [[experiments.019-simb-multimodal.scripts.build_fig3_proteome]]), the three expression panels plus `ProteomeMessner2023Dataset`. Proteome values are `log2(strain / HIS3 reference)` per protein, converted at build time by [[torchcell.datamodels.protein_abundance_log2_ratio_conversion]].
- `cell_dataset.require_modalities: [protein_abundance]`: only genotypes carrying the proteome enter train, val and test.
- `multitask.head_phenotypes.per_gene: [protein_abundance]`: the expression head reads the proteome label. `multitask.head_phenotype_names.per_gene: proteome` names the metric namespace, so the score is `val/proteome/pearson_per_feature` (new config key, read in `train_cgt_multitask.py` right after `head_phenotypes`).
- `trainer.max_epochs: 2000` (about v13's 6,000 in gradient steps, since the proteome train set is about 2.9x the records) and the checkpoint monitor follows the namespace.
- W&B project `torchcell_019_prot_v14`.

**Design.** 16 runs: {P_ref, P_concat} x split seeds {0,1,2,3} x init seeds {0,1}, four per RTX 6000 Ada card on cabbi, one split per card, so ref/concat is paired within card and seed as in v13. The per-feature z-score of the target (`standardize_per_feature_target: [per_gene]`, inherited) removes any per-protein offset, so the log2 ratio and log2 abundance would score identically on Pearson; the ratio is stored because it is what the label means.

**Why the proteome.** Per-strain agreement with Kemmeren on the 1,350 shared deletions is 0.04 and the gene co-variation Spearman is 0.31 (expression document, section "The knockout proteome against three expression panels"); it is a target of its own. Three times the strains of Kemmeren, one measurement per strain, no replicate ceiling of its own beyond the 29 SGA-background re-measurements (Spearman 0.72 to -0.19).

**Hypotheses, unmeasured.** Throughput at four per card about 13 epochs/h/run, 2,000 epochs in about 6.5 days. Whether the trained arms beat the bilinear ridge on ProtT5 on the proteome partitions is what the round measures; the baselines run on GilaHyper CPU with `gh_expression_baselines_split.slurm` under `BASELINES_TAG=fig3_proteome BASELINES_LABEL=protein_abundance`.
