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

**Sparse target.** The proteome vector spans the 1,850-protein union with NaN where a strain did not quantify a protein (median 1,820 measured per strain); loss, metrics and masked conditioning score finite entries only. Details and the verification in [[torchcell.datamodels.protein_abundance_log2_ratio_conversion]]. `val/proteome/n_scored_genes@k0` on W&B says how many proteins survive the 3-pair floor each epoch.

## 2026.09.16 - First read at the matched epoch 233 (PARTIAL, 16 runs, job 2400350)

`v13_split_readout.py --round v14`, `results/v14_proteome_readout.json`. Rate: task 0 at 234 to 239 epochs after 20 h (about 12 per hour per run), tasks 1 to 3 at 327 to 342 after 19 h (about 17 per hour); 2,000 epochs lands at 5 to 7 days.

- Partition means (both readouts, both seeds): split 0 0.114, split 1 0.078, split 2 0.112, split 3 0.085; between-split sd 0.019, range 0.036, within-split sd 0.004. The partition axis is as large on the proteome as on expression, and the order differs from Kemmeren's (split 2 is high here, low there).
- P_concat minus P_ref: mean 0.000, sd 0.006 over 8 pairs, 5 of 8 positive. Nothing.
- Against the split-matched linear baselines (val): the arms lead by 0.03 to 0.05 on splits 0, 2 and 3 (e.g. 0.114 against B3 0.071 on split 0) and by nothing on split 1 (0.079 against B3 0.082).
- The curves peak EARLY and fall: `roll_max` epochs 66 to 207 with last values 0.02 to 0.03 below the peak by epoch 233 (e.g. `P_concat_s3` seed 0 peak 0.085 at epoch 66, last 0.050). The expression curves rise to thousands of epochs; the proteome's do not, at least at this point. Hypothesis (untested): with one measurement per strain and a ceiling of 0.40 to 0.70, the per-protein z-scored target has less recoverable signal per gradient step, and the model starts fitting noise early; the best-val checkpoint (monitor `val/proteome/pearson_per_feature`) will hold the peak, and the test read at that checkpoint is the number to trust.

Report and ranked Charts view: `wandb_v13_report.py --round v14` (saved view `nw=yxor02gt7y9`). Task job ids are not contiguous (task0 2400351, task1 2401107, task2 2401108, task3 2400350); sync with `sync_019_live.sh <ids>` on the login node.
