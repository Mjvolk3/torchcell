---
id: st2tfkmvql5dajzwvsgye8t
title: Per Head Convergence and Early Exit
desc: ''
updated: 1790496546553
created: 1790496546553
---

## 2026.09.27 - Idea, recorded and parked: heads on one trunk peak at different times

Recorded at the author's request; NOT to be acted on now. It is a late-stage inference
question, after the joint rounds have read out.

**The observation.** In v16 the proteome head on the joint trunk peaks by epoch 150 to
260 and decays afterward (validation loss rising from epoch 180 to 230 on every
proteome-only run; the joint arm's proteome head peaks at 238 to 499, later than alone),
while the expression head on the same trunk is still rising at epoch 499 and, from v13
and v17, keeps rising past 1,200. One trunk, two heads, two convergence times. The v19
design answers it for scoring by giving each head its own window (proteome epochs 200 to
400, expression 1,000 to 1,200) and its own checkpoint; see
`conf/cgt_expr_v19_joint_clean.yaml` and the wireframe panel e in
[[experiments.019-simb-multimodal.scripts.v19_joint_mockup]].

**The idea.** At inference, serve each head from the checkpoint at which that head peaked
rather than from the trunk's final state, which is the early-exit family of ideas applied
across heads rather than across depth: a head that has converged exits early, a head that
has not keeps training. Questions this raises, none measured:

- Whether the proteome head's decay after its peak is overfitting of the head alone (a
  9,919-parameter MLP) or drift of the shared trunk toward the expression objective; the
  train fit deficit of -0.095 on the joint proteome head (review of 2026-09-27) says the
  trunk is involved.
- Whether freezing a head at its peak while the trunk continues (a per-head stop rule)
  preserves that head's score, or whether the trunk's later movement invalidates the frozen
  head's input distribution.
- Whether per-head learning rates or loss weights that decay after a head's peak reach the
  same end without checkpoint surgery.
- What the serving cost is: two checkpoints of one trunk against one, or one trunk plus
  per-head adapters.

**When to return to it.** After v19 gives per-head curves on twelve partitions, which is
the data that says how far apart the peaks are and how much each head loses by being read
at the other's time.
