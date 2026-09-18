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
