---
id: t3zbu45qsge27ox14lppfos
title: Equivariant_cell_graph_transformer
desc: ''
updated: 1788575502881
created: 1788575502881
---

## 2026.09.04 - Port of the 010 Trainer onto the 025 Build

Trains the equivariant `CellGraphTransformer` on one subset/split arm of the 025
all-solid-growth build. Ported from
`experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer.py`. Design,
arm table, and the masking-layer decision live in
[[experiments.025-solid-growth.training-plan]].

### What the arm config has to say, and why each is load bearing

010 trained on 376,732 records holding exactly the trigenic triples and exactly one
label. 025 holds 13,525,071 records over three perturbation orders and two labels, so
four things that were implicit there have to be stated here. Each produces a plausible
run rather than an error when omitted:

| config key | without it |
|---|---|
| `cell_dataset.phenotype_labels` | a triple carries fitness AND gene_interaction, so a batch of B records supplies 2B targets |
| `subset.indices` | pinning assigns but never excludes, so the 13,142,648 doubles join training |
| `subset.split_file` + `split_key` | no pinned split; R and Q store their lists under different field names |
| `transforms.fit_on_subset` | normalized by the whole column's sd 0.0444 instead of the triples' 0.0633 |

The script asserts the realized split equals the pinned artifact intersected with the
subset, per split, before training starts. That assertion is the one place a subset and a
pin can silently disagree: an index named by the pin but absent from the subset is
dropped rather than placed, which is correct behavior and also exactly how an arm could
train on fewer records than its name claims.

### Verified against the build

- Dataset opens in 10.5 s at 5.5 GB RSS; `len` 13,525,071; perturbation orders
  1: 5,694 / 2: 13,142,648 / 3: 376,732.
- The nine gene-gene relations are `physical_interaction` (144,211 edges),
  `regulatory_interaction` (44,310), `tflink` (207,250), `string12_0_coexpression`
  (1,002,806), `string12_0_experimental` (828,701), `string12_0_neighborhood` (153,320),
  `string12_0_database` (79,224), `string12_0_fusion` (18,394),
  `string12_0_cooccurence` (17,692); 6,607 gene nodes.
- Normalization on S0: mean -0.008024, sd 0.063264, min -1.0816, max 1.128043, matching
  the 010 report's label statistics.
- R split realizes 301,386 / 37,673 / 37,673; Q split realizes 301,236 / 37,705 / 37,791.
  Both equal their artifacts exactly, and every split record is a triple.
- Model instantiates at 4,774,861 parameters under both configs, the same count as 010.
- One-epoch smoke at batch 8 on a single GPU: 21 batches in 17 s under KL, 8 s under
  masking. The direction agrees with 019's 1.5-1.7x, but batch 8 is not the production
  regime and this is not a throughput measurement.

### Reference the replication arm is read against

010's three checkpoints reached validation Pearson 0.4520 (M01, `lzs9pcj3`), 0.4472
(M02, `yv4r30bi`) and 0.4619 (M03, `c7671wgj`). Their best-Pearson checkpoints sit at
epochs 24 and 25, and the cosine schedule's first cycle is 30 epochs, so the 12 h wall
clock is being spent in the range where 010 peaked rather than truncating a long climb.
