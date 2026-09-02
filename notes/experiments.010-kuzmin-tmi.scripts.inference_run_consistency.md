
## 2026.09.01 - The Same Checkpoint Disagrees With Itself Across Inference Runs

Checkpoint c7671wgj scored three 010 inference spaces: `inference_1` (4,370,595
triples, 526 genes), `inference_2` (479,195), and `inference_3` (465,735,532,
genome-wide, the space the panel-12 and panel-24 selection consumed). The model is a
deterministic function of the perturbed gene set, so these must agree.

Three checks: the prediction for triples appearing in two runs, the correlation of
per-gene mean predictions between runs, and the correlation of each run's per-gene
means with the additive ridge coefficient. That coefficient is fit on measured
training data, keyed on true systematic gene name, and independently validated at
0.400 test Pearson, so it anchors gene identity from outside the transformer.

`inference_2` and `inference_3` agree exactly on shared triples, Pearson 1.0000 and
mean absolute difference 0.000000 over 2,404 triples, so they share a pipeline.
`inference_1` is unrelated to them, per-gene Pearson 0.037 against `inference_3`.
Against the additive anchor on the 167 genes common to all three runs, so the gene
set is not a confound, `inference_1` scores 0.576 while `inference_2` scores 0.231
and `inference_3` scores 0.174.

![](assets/images/010-kuzmin-tmi/inference_run_consistency.svg)

No cause is confirmed. A gene index or embedding misalignment in the shared
`inference_2` and `inference_3` build would match the signature, predictions that
are self-consistent yet detached from gene identity, but that is a hypothesis and
has not been verified.

Findings: [[experiments.010-kuzmin-tmi.additive-baseline-analysis]]
