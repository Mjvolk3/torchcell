---
id: wxg3l12786771nxkt15o4zt
title: Cgt_expr_v16_joint
desc: ''
updated: 1789698582818
created: 1789698582818
---

## 2026.09.17 - The joint round: transcriptome as a second per-gene head beside the proteome

Config `experiments/019-simb-multimodal/conf/cgt_expr_v16_joint.yaml` (inherits v14). The question: does supervising the trunk with the Kemmeren/Sameith knockout transcriptome move the proteome score? Motivation measured in [[experiments.019-simb-multimodal.scripts.proteome_ceiling_replicate]]: v14 sits at a fifth of the proteome's genotype ceiling, the two panels agree on gene co-variation at 0.31, and 1,349 strains carry both labels.

What changed in code (commit on this branch):

- model: `per_gene_aux`, a second `PerGeneHead` over the same head input as `per_gene`, built when the heads config names it
- trainer: every `head == "per_gene"` special case became `head.startswith("per_gene")` (spec knobs, col_idx alignment, output_dim check); `multitask.mask_head` names the head the reveal schedule teacher-forces (default `per_gene`), the other per-gene head is supervised on all its genes at every step
- trainer: `trainer.eval_ckpt_path` evaluation-only path (load weights, validate, dump val and test predictions), used by [[experiments.019-simb-multimodal.scripts.gh_eval_ckpt_predictions]]

Design decisions:

- the partition is v14's: `require_modalities` filtered each split after the draw, so dropping it keeps every proteome validation and test strain and adds the 205 expression-only genotypes (train 3,581 to 3,722 supervised rows); the gain is the expression label on the 1,349 shared strains
- the reveal schedule stays on the proteome head only; the expression head is never revealed, so at k = 0 both heads are unconditioned and `val/expression/pearson_per_feature` is comparable to the expression rounds on these strains (not the same strains as v13)
- first checkpoint monitors `val/proteome/pearson_per_feature` (what the test pass loads), second `val/expression/pearson_per_feature`; two ModelCheckpoints on one key is a Lightning error, which the smoke test found
- arms `J_ref_s<k>` (aux head off, proteome strains only, `metric_monitor` back to the mean: byte-for-byte v14's reference at 500 epochs), `J_joint_s<k>` (both heads, weight 1), `J_joint05_s<k>` (aux weight 0.5, not in the first submission); stage `joint` in `igb_expr_wave5.slurm`: splits 0, 1, 2, two seeds, four per card, three cabbi cards, 500 epochs

Hypothesis (untested): the auxiliary label changes the proteome score by less than the within-partition sd of 0.005 at 500 epochs; the round is sized to see 0.01.

## 2026.09.18 - Expression-only arm on the same partition, the split audit, and the loss checkpoint

- Split audit ([[experiments.019-simb-multimodal.scripts.split_gene_overlap_audit]], report 11 under the review results folder): the draw shuffles record indices per index key and never looks at gene identity; one record is one deletion gene set, so exact duplicates cannot cross a split (0 on every seed). The proteome rows of every seed are exactly v14's (3,581 / 448 / 447), gene-disjoint by construction. Of the 155 held-out expression genotypes per seed, 17 to 20 share a deleted gene with train, all single-versus-double relations among the 72 Sameith doubles. v13's partition is not reproducible on fig3_proteome (same 1,554 genotypes in a different record order; held-out overlap with v13 at the same seed 6 to 15%, chance 10%).
- New arm `J_expr_s<k>`: the expression-only reference on the same partition, the masked `per_gene` head re-pointed at `expression_log2_ratio` with no aux head and `require_modalities []`, so the 3,127 proteome-only genotypes ride along with the row mask off. One-epoch CPU smoke on GilaHyper (job 2366) trains, validates on 6,127 genes and writes its three checkpoints; the joint config's own one-epoch smoke (job 2365) does the same with proteome-only genotypes carrying no expression label. The round is now three arms per (split, seed): J_ref, J_expr, J_joint, eighteen runs as six three-run tasks, sent to cabbi and the gpu partition as cards free.
- `trainer.checkpoint.save_loss_min: true`: a fourth ModelCheckpoint on `val/loss` (mode min), so the early loss-minimum model exists beside the best-metric model; v13 and v14 saved none, so the question "is the loss-minimum model better on the loud features or better calibrated" had no file to answer it.
- Readout rule: the expression head is compared to J_expr, not to v13, on all held-out expression rows and on the gene-disjoint rows (val 137 / 138 / 136, test 140 / 135 / 138 for seeds 0 to 2); the proteome head to J_ref, paired with v14.
