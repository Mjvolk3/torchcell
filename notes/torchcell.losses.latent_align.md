---
id: sydoyhou2fxcyq09bf7qohw
title: Latent_align
desc: ''
updated: 1791814441401
created: 1791814441401
---

## 2026.10.12 - Latent alignment for the inductive rounds

`torchcell/losses/latent_align.py` holds an auxiliary term that shapes a strain's latent
vector by the similarity of its labels. For every pair of supervised strains in a batch,
the cosine similarity of their latent vectors is pulled toward the Pearson correlation of
their label vectors (the 116 moving CalMorph features, or the measured proteins), with a
squared gap averaged over the pairs. The label correlation runs over the features both
strains have measured, so a sparse target (NaN where unmeasured) needs no imputation; a
pair sharing fewer than `min_overlap` features is left out.

Why: in rounds v24 and v25 of experiment 019 the genotype-only morphology runs sit on a
plateau where every strain gets the same readout vector and the head predicts the
per-feature marginal; the pinball loss barely moves across an escape (0.195 to 0.187
while the training Pearson goes 0.06 to 0.34), so nothing in the data loss pushes the
latents apart. The marginal solution scores a cosine of one for every pair against a
spread of label correlations, so this term penalizes the plateau from the first step.

- `masked_pairwise_pearson(targets, min_overlap)`: [n, n] correlations and a validity
  mask, vectorized through masked matrix products.
- `latent_align_loss(latent, targets, row_mask, min_overlap, center)`: the loss and the
  pair count; `center` subtracts the batch mean of the latents before the cosine.
- `strain_latent(reps, idx, idx_batch, source, pooling)`: the vector it shapes, read from
  the model's reps: `cls` (the perturbed CLS, needs `model.perturb_cls`), `zs` (the pool
  over the strain's perturbed gene tokens, the PerturbationHead input), `cls_zs` (both).

Wired into `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py` through
`multitask.latent_align` (declared off in `conf/cgt_expr_008.yaml`): the term is added into
the step's total beside the data loss and logged as `<stage>/latent_align/loss` and
`<stage>/latent_align/pairs`; it runs on the plain step only (mask schedule and condition
head null). Launched as stage `morph_v26` (morphology, project `torchcell_019_morph_v25`)
and `prot_v26` (knockout proteome, project `torchcell_019_prot_v26`) of
`igb_expr_wave5.slurm`, and as `morph_v26d` of `delta_morph_v25_small.slurm`.
Tests: `tests/torchcell/losses/test_latent_align.py`.
