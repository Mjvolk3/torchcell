---
id: mzcq23e6lnihd0fe4yl981o
title: Paired_prediction_agreement
desc: ''
updated: 1788318029124
created: 1788318029124
---

## 2026.09.01 - Record-by-Record Agreement on the Test Split

Joins the baseline test predictions with the per-record transformer predictions
from [[experiments.010-kuzmin-tmi.scripts.score_010_checkpoints_directly]] and
compares the predictions themselves rather than their summary statistics, which
is the form the DeWitt preprint's Figure 1 takes.

Prediction correlation on the 37,673 test records:

| | CGT M01 | CGT M02 | CGT M03 |
|---|---|---|---|
| B1 additive ridge | 0.752 | 0.726 | 0.759 |
| B5 embedding MLP | 0.788 | 0.784 | 0.810 |
| CGT M01 | | 0.788 | 0.821 |
| CGT M02 | 0.788 | | 0.804 |

The 010 transformer does NOT reproduce the additive model. DeWitt reported
r > 0.999 for MULTI-evolve; here it is 0.73 to 0.76. That is the opposite
finding and it should be stated plainly.

But two training runs of the same architecture agree with each other at 0.788 to
0.821, barely above a checkpoint's agreement with the additive model and no
better than its agreement with the nonlinear baseline. So the non-additive part
of the output is largely not reproducible across seeds.

Residual correlation between B1 and the checkpoints is 0.907, 0.897 and 0.918,
so the models are wrong in the same places.

Paired bootstrap over records, 2,000 resamples, for the Pearson gain over B1:
M01 +0.043 [+0.023, +0.063], M02 +0.038 [+0.019, +0.058], M03 +0.055
[+0.034, +0.074], B5 +0.025 [+0.017, +0.033]. All intervals exclude zero.

Top-100 overlap on the most negative predicted interaction: B1 and M03 share 31,
B5 and M03 share 45, B1 and B5 share 46. At K = 10,000 every pair exceeds 0.91.
The models agree about the bulk and disagree about the tail.

![](assets/images/010-kuzmin-tmi/paired_prediction_agreement.svg)

Findings: [[experiments.010-kuzmin-tmi.additive-baseline-analysis]]
