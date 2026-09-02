---
id: 6aq9a7sq8tkvznpe6du27e9
title: Score_010_checkpoints_directly
desc: ''
updated: 1788318021688
created: 1788318021688
---

## 2026.09.01 - Scoring the 010 Checkpoints Without the Loader

The stock eval cannot run on the 010 build: its LMDB stores
`perturbation_type: "deletion"` and the current pydantic schema admits only
`sga_kanmx_deletion` or `sga_natmx_deletion`, giving 204 validation errors on the
first batch.

The architecture makes a bypass possible. The encoder runs at batch size one on
the wildtype gene embedding table, so its output is the same for every strain,
and the only strain-dependent input is the perturbed gene indices. This rebuilds
the gene index from the genome, loads each checkpoint into the model class
directly, and scores index triples recovered from the build's JSON side files.

Two settings had to be right, and both fail silently by producing plausible
numbers:

- The index space is the sorted GENOME gene set, 6,607 genes, matching
  `gene_num`. Using the build's own 6,579-gene set gives val Pearson -0.001 to
  0.002.
- The readout pools by `mean`. The class now defaults to `sum`, which gives
  0.420 to 0.438, close enough to look credible.

With both correct the reproduction is exact:

| checkpoint | recomputed val Pearson | recorded |
|---|---|---|
| M01 lzs9pcj3 | 0.451970 | 0.451963 |
| M02 yv4r30bi | 0.447319 | 0.447155 |
| M03 c7671wgj | 0.461917 | 0.461881 |

That is the direct verification that the encoder is strain-independent: it was
evaluated once and the recorded per-split metrics came back.

The same run sets the graph-regularization weight from 1.0 to 0.0 and finds a
maximum absolute change in test predictions of 9.0e-7, 1.2e-6 and 1.3e-6 against
a prediction spread of about 0.036, confirming the nine graphs do not enter the
forward pass.

Outputs per-record `cgt_predictions_<ckpt>_<split>.npy` consumed by
[[experiments.010-kuzmin-tmi.scripts.paired_prediction_agreement]].

Findings: [[experiments.010-kuzmin-tmi.additive-baseline-analysis]]
