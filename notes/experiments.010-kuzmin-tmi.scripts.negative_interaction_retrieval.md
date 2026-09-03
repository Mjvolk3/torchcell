---
id: 74xv7pd8g2hyh31pgvmvzyo
title: Negative_interaction_retrieval
desc: ''
updated: 1788414573343
created: 1788414573343
---

## 2026.09.03 - The model is a usable retriever of negative interactions

Every confident prediction the model makes about the construction panel is
negative, which invites using it to find negative interactions rather than
positive ones. That needs testing rather than assuming, because the label is
already skewed negative: 59.4 percent of the 376,732 records have tau below zero
and the mean is -0.0080, so a squared-error fit that had learned nothing beyond
the mean would still emit mostly negative predictions.

The test is retrieval. Among the K records a model calls most negative, what
fraction really are strong negatives, and does that beat the base rate and the
additive null.

### On the published split it works, and the transformer beats the null

Held-out test, 37,673 records. Precision at K = 100, with enrichment over the
base rate in parentheses.

| model | tau < -0.08 (base 8.3%) | tau < -0.10 (base 5.1%) | tau < -0.20 (base 0.88%) |
|---|---|---|---|
| B1 additive ridge | 0.560 (6.8x) | 0.520 (10.1x) | 0.240 (27x) |
| B5 nonlinear MLP | 0.700 (8.5x) | 0.670 (13.1x) | 0.400 (45x) |
| CGT M01 | 0.670 (8.1x) | 0.630 (12.3x) | 0.400 (45x) |
| CGT M02 | 0.750 (9.1x) | 0.700 (13.7x) | 0.420 (48x) |
| CGT M03 | 0.710 (8.6x) | 0.670 (13.1x) | 0.450 (51x) |

Average precision at tau < -0.20 is 0.096 for B1 against 0.182 to 0.191 for the
three checkpoints, so the transformer's advantage over the additive null is
proportionally larger on this task than on Pearson, where it was 0.400 against
0.438 to 0.455. Retrieval of the extreme tail is what the architecture buys.

### It survives the disjoint split, weakened, for the models measured there

The transformer has not been retrained on the query-pair-disjoint split, so only
the baselines have a row. At tau < -0.10 the disjoint test base rate is 9.6
percent, and precision at 100 is 0.26 for B1 (2.7x) and 0.33 for B5 (3.4x),
against 10.1x and 13.1x on the random split. So retrieval is real but several
times weaker once query doubles cannot be shared, and the absolute precisions are
not comparable across splits because the base rates differ.

![](./assets/images/010-kuzmin-tmi/negative_interaction_retrieval.svg)

### What this supports

Using the model to nominate candidate negative trigenic interactions is defensible
on the evidence, and is a better fit to what it can do than nominating positives:
on the build list 17 of 20 triples have all three checkpoints agreeing and every
one of those is negative, while the single positive is not agreed.

What is not established is the enrichment a transformer would retain on a
query-pair-disjoint split, since it has not been refit there. The baseline drop
from roughly 10x to 3x is the honest prior for what to expect.
