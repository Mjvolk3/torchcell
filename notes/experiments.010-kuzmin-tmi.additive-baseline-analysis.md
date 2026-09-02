---
id: 6vv13mcj0t0m2070ui8rfsd
title: Additive Baseline Analysis
desc: ''
updated: 1788312482240
created: 1788312482240
---

## 2026.09.01 - Additive Null Baselines for the 010 Trigenic Interaction Score

### Why

A colleague asked whether the 010 trigenic interaction predictions could be
matched by a simpler model, pointing at Visani, Verma and DeWitt, "Additive
baselines furnish no evidence for epistasis learning by MULTI-evolve"
(bioRxiv 2026.04.23.719915). That preprint shows a neural network claimed to
learn epistasis is reproduced almost exactly by a ridge additive model over
mutation indicators, so its engineering wins reduce to stacking beneficial
single mutations. Their standard is that a model claiming to learn interaction
must be benchmarked against a null with no capacity to represent interaction.

Prior traditional-ML baselines in this repo cover `002-dmi-tmi` (mixed doubles
and triples) and `smf-dmf-tmf-001` (fitness). Nothing had been run against the
010 triples-only build.

### Setup

Every baseline uses the same build, split, and label the transformer used, read
straight from the pinned build rather than through the loaders:

- build `$DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/001-small-build`
- label `gene_interaction`, the Kuzmin adjusted trigenic interaction score tau
- 376,732 records, all perturbation count 3, over 4,352 genes
- split `index_seed_42.json`, train 301,386 / val 37,673 / test 37,673
- train label SD 0.063186, so the null MSE is about 0.00396 on val

The transformer numbers come from the three checkpoints' own re-evaluation runs
under `$DATA_ROOT/wandb-experiments`, not retyped.

Generating script:
[[experiments.010-kuzmin-tmi.scripts.additive_baseline_gene_interaction]]

An important caveat on what "additive" means here. In the protein setting the
phenotype is the multimutant readout, which single-mutant effects explain
directly. Here the label is already a residual: tau subtracts the single and
double mutant expectations by construction. So an additive-in-single-gene-effects
model is a stricter null than it is in the preprint. It has one large loophole,
addressed by B4 below: the Kuzmin design crosses a query double against an
array, and the 010 split is random over triples, so a per-gene model can absorb
per-query-pair screen structure.

### Held-out results, test split (37,673 triples)

| model | features | test Pearson | test Spearman | test MSE |
|---|---|---|---|---|
| B0 train mean | none | 0.000 | 0.000 | 0.004124 |
| B4 query pair only | recurring gene pairs, third gene ignored | 0.390 | 0.362 | 0.003493 |
| B3 hierarchical mean | pair mean backing off to gene mean | 0.390 | 0.362 | 0.003493 |
| B1 additive per-gene ridge | 4,352 gene indicators | 0.400 | 0.406 | 0.003460 |
| B2 additive + gene-pair ridge | + 420 recurring pairs | 0.406 | 0.412 | 0.003442 |
| B5 nonlinear MLP | same one-hot genes as B1 | 0.426 | 0.419 | 0.003386 |
| CGT M01 lzs9pcj3 | 9 graphs, learned embeddings | 0.443 | 0.426 | 0.003400 |
| CGT M02 yv4r30bi | 9 graphs, learned embeddings | 0.438 | 0.424 | 0.003439 |
| CGT M03 c7671wgj | 9 graphs, learned embeddings | 0.455 | 0.426 | 0.003315 |

B5 is three seeds, test Pearson 0.4251 / 0.4270 / 0.4253, mean 0.4258 with SD
0.0011. The three transformer runs are 0.4379 / 0.4434 / 0.4551, mean 0.4455
with SD 0.0088, so the transformer spread across training runs is eight times
the MLP's spread across seeds.

![](assets/images/010-kuzmin-tmi/additive_baseline_test_pearson.svg)

Ridge penalty was chosen on val over a nine-point grid and then reported on
test. B1 and B2 both selected alpha 30; B4 was flat in alpha and selected 0.01.
Every gene appearing in val or test also appears in train, so no baseline is
extrapolating to unseen genes.

### What this says about the colleague's question

The DeWitt pattern is present but weaker than in their case. It is not true that
the transformer collapses to an additive model, and it is not true that the gap
is large.

- A model with no capacity to represent gene interaction at all reaches 0.400
  test Pearson against the transformer's 0.438 to 0.455. That is 88 to 91
  percent of the transformer's correlation.
- In variance explained the honest framing is smaller still. The additive model
  explains r squared 0.160 of test label variance, the best transformer 0.207.
  The increment attributable to everything the transformer adds, nine
  interaction graphs, learned embeddings, eight transformer layers, is 0.047 of
  label variance.
- On MSE the best transformer reduces error 19.6 percent below the null, and
  the additive model already delivers 16.1 percent of that.
- Nonlinearity on the identical feature space, B5, closes roughly half the
  remaining gap. B5 sees no interaction graph at all yet reaches 0.426, and its
  MSE 0.003386 beats two of the three transformer checkpoints. So most of what
  is left after additivity is capacity, not biology from the graphs.
- B4 is the sharpest result. A model that only knows which recurring gene pair
  the triple contains, and never looks at the third gene, reaches 0.390 test
  Pearson. There are 420 such pairs, the largest observed 3,308 times, which is
  the Kuzmin query-double structure. So most of the ranking signal every model
  here shows is a per-query-double offset, not trigenic biology.

The fair summary for a colleague is that the transformer beats a no-interaction
null by a real but modest margin, roughly 0.045 Pearson and 0.047 of variance
explained, and that a large majority of all models' apparent skill on this split
is screen structure that an additive model captures for free. A gene-disjoint or
query-pair-disjoint split would test the biology and has not been run.

### The larger problem found while doing this

Applying the additive model to the 010 design space, the analogue of the
preprint's Figure 1, produced a result that turned out not to be about
additivity at all.

Scripts:
[[experiments.010-kuzmin-tmi.scripts.additive_vs_transformer_inference_space]],
[[experiments.010-kuzmin-tmi.scripts.inference_space_predictor_agreement]],
[[experiments.010-kuzmin-tmi.scripts.inference_run_consistency]]

Over `inference_3`, the 465,735,532-triple genome-wide space whose predictions
the panel-12 and panel-24 selection consumed, restricted to the 325,631,845
triples whose three genes all carry an additive coefficient:

- Pearson between the additive model and checkpoint c7671wgj is 0.0018.
- Top-100 and top-1000 selection overlap between the two is 0.0000.
- Stratifying by the training support of the least-observed gene does not
  rescue it. Even for triples whose rarest gene has 800 or more training
  observations, n = 867,669, the correlation is 0.036.

Over `inference_1`, a 4,370,595-triple space on a 526-gene panel, the same
comparison behaves completely differently. Additive against the three
checkpoints gives Pearson 0.548, 0.579 and 0.511.

The two spaces cannot both be right, and the resolution is that the same
checkpoint does not agree with itself across inference runs.

| comparison | n | Pearson |
|---|---|---|
| same triple, inference_2 vs inference_3 | 2,404 | 1.0000 |
| same triple, inference_1 vs inference_3 | 330 | 0.295 |
| per-gene mean prediction, inference_2 vs inference_3 | 1,274 genes | 0.800 |
| per-gene mean prediction, inference_1 vs inference_3 | 395 genes | 0.037 |
| per-gene mean prediction, inference_1 vs inference_2 | 179 genes | 0.167 |

`inference_2` and `inference_3` are byte-identical on shared triples, mean
absolute difference 0.000000, so they are the same pipeline. `inference_1`,
from the same checkpoint, is unrelated to them.

To decide which run handles gene identity correctly, the additive ridge
coefficient is an external anchor: it is fit on measured training data, keyed on
true systematic gene name, and independently validated at 0.400 test Pearson.
Correlating each run's per-gene mean prediction against it, restricted to the
167 genes all three runs score with at least 200 observations so the gene set is
not a confound:

| run | Pearson vs additive coefficient, 167 common genes |
|---|---|
| inference_1 | 0.576 |
| inference_2 | 0.231 |
| inference_3 | 0.174 |

![](assets/images/010-kuzmin-tmi/inference_run_consistency.svg)

So on identical genes, `inference_1` tracks real gene identity more than three
times as strongly as the runs that produced the selection. Over their own full
gene sets the separation is starker, 0.579 for `inference_1` against 0.048 for
`inference_2` and -0.008 for `inference_3`.

**Hypothesis (untested):** a gene index or embedding misalignment in the
`inference_2` and `inference_3` dataset build, which share a pipeline, would
produce exactly this signature, predictions that are self-consistent and stable
but detached from gene identity. The inference builds carry no `gene_set.json`
of their own, while the training build has one of 6,579 genes, so the index used
at inference time is reconstructed rather than loaded from the build. This has
not been verified and no cause has been confirmed.

### Seed instability in the selection tail, independent of the above

On `inference_1`, where predictions do track gene identity, the three
checkpoints still disagree substantially about which triples are extreme.
Pairwise Pearson between checkpoints is 0.524 to 0.560, no higher than each
checkpoint's 0.511 to 0.579 agreement with the additive model. Top-100 overlap
between two checkpoints is 0.21 to 0.35.

![](assets/images/010-kuzmin-tmi/inference_space_predictor_agreement.png)

Two independent training runs of the same architecture on the same split
nominate largely different top-100 triples. Selection from a single checkpoint's
extreme tail is therefore not reproducible even without the inference problem.

### Follow-ups

- Verify or rule out the gene index hypothesis for `inference_2` and
  `inference_3` by re-scoring a small triple set through both build paths and
  comparing against `inference_1`.
- If confirmed, the panel-12 and panel-24 selections need regenerating, and
  `prediction_calibration_stats.csv` pins the affected parquet by sha256
  `806ef044...3550`.
- Run a query-pair-disjoint split to separate trigenic biology from screen
  structure, since B4 shows how much of the current metric is the latter.
- Report an ensemble or a seed-averaged ranking rather than one checkpoint's
  tail for any future selection.
