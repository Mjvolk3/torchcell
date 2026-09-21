---
id: qa5fqy7mlyazkih71jt6nw6
title: Cgt_expr_v18_hygiene
desc: ''
updated: 1789959305127
created: 1789959305127
---

## 2026.09.20 - The hygiene and readout round, queued as IGB 2409571

The v13 reference against the two levers that need no code change and have never been tested at this trunk. Queued on cabbi behind the weight-decay round and two other groups' jobs, at the author's instruction to add to the queue rather than free cards.

| arm | override | question |
|---|---|---|
| `Y_ref_s<k>` | none | the v13 reference at 1,200 epochs, the shared control |
| `Y_k0_s<k>` | `multitask.mask_schedule=[0]` | is the masked objective doing anything, and what does it cost |
| `Y_ctx_s<k>` | `multitask.context_readout=true` | does a per-gene affine row over the strain context help |

**The mask arm.** Every run from v10 through v17, 166 of them, carries `mask_schedule: [0, 10, 100, 1000]`, and no no-mask control exists at this trunk. The expression-fit review measured both sides. On the training set the k3 minus k0 gap is +0.245 at epoch 100, +0.034 at 500, +0.002 at 1,000 and 0.000 after, so the objective is inert across the band this round scores. On validation, where nothing is revealed, the imputation capability peaks at 0.48 around epoch 1,200 and decays to 0.32 by the selected checkpoint. The cost is measured at +24% wall from two co-resident 9,999-epoch runs, 222,666 s against 276,815 s. Setting the schedule to `[0]` removes the objective and leaves the plumbing alone, so the arms stay comparable. This arm is judged on two axes: the score at matched epochs, where a tie is the expected outcome, and the score at matched wall clock, where it should reach about a quarter more epochs for the same card time.

**The readout arm.** `context_readout` gives every gene its own affine row over the strain context, added to the shared head's output. Both State's gene reconstruction head and the Ahlmann-Eltze benchmark's decoder read each gene off a strain vector through a gene-specific row and never off the gene's own token; this is that form. The row is zero-gated, so at step 0 the head is exactly the shared multilayer perceptron and the arm is a clean ablation. The v12 head round tested the GEARS-style sibling, which reads a function of the gene token and the context, and found it worth -0.0085 at 1,399 epochs but +0.032 at 500, a speed lever rather than a ceiling lever. The strain-only form has never run.

**Design.** Split seeds 0 to 3, init seeds 0 to 2, the three arms co-resident per card, 36 runs as twelve tasks of three, 1,200 epochs, the loss-minimum checkpoint kept beside the best-metric one. Scored as the fixed-window mean with the closing test read; adopt only above +0.02 with the 12-pair confidence interval excluding zero, three of four partitions positive and the test sign agreeing.

**Verification.** Both arms fast-dev-run on CPU. The context arm reports 7.3M trainable parameters against the reference's 6.7M, which is the 557,558 its formula predicts for 6,127 genes at hidden dimension 90. The mask arm logs a single `k0` branch with `n_revealed@k0` at zero, confirming the schedule collapsed rather than the metric merely being hidden.

Related: [[experiments.019-simb-multimodal.expression-fit-review]], [[experiments.019-simb-multimodal.conf.cgt_expr_v17_locality]], [[experiments.019-simb-multimodal.scripts.campaign_status]].
