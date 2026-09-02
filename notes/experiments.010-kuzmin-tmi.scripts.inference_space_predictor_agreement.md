---
id: xq2ijhjqq02p8hcgkbr0vp5
title: Inference_space_predictor_agreement
desc: ''
updated: 1788312504142
created: 1788312504142
---

## 2026.09.01 - Do the Three 010 Checkpoints Agree on Unmeasured Triples?

Joins the `inference_1` predictions of all three 010 checkpoints on the gene triple,
adds the additive per-gene ridge prediction, and reports the full correlation matrix
plus top-K and bottom-K selection overlap for every pair.

On the 3,132,471 triples whose genes all carry an additive coefficient, checkpoint
to checkpoint Pearson is 0.524 to 0.560, which is no higher than each checkpoint's
0.511 to 0.579 agreement with the additive model. Top-100 overlap between two
checkpoints is 0.21 to 0.35.

Two independent training runs of the same architecture on the same split therefore
nominate largely different extreme triples, so a selection taken from one
checkpoint's tail is not reproducible across seeds.

![](assets/images/010-kuzmin-tmi/inference_space_predictor_agreement.png)

Findings: [[experiments.010-kuzmin-tmi.additive-baseline-analysis]]
