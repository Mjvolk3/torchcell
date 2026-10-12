---
id: bi8r5eggbtukj8sc5hcjw5b
title: Figure2_gate_numbers
desc: ''
updated: 1791798493917
created: 1791798493917
---

## 2026.10.12 - The numbers behind the Figure 2 gate

`notes-tex/interaction-fitness-figure-gate` is the gate document for manuscript Figure 2
(fitness and gene-gene interactions; gene essentiality moved to its own gate on the
kinetics branch), on the pattern of `notes-tex/metabolism-figure-gate`: the decision,
the mockups, the models and comparators on file, what remains, the decision rules, a
registry of eight pre-registered experiments, and every rung of the lower-order ladder
with its regime. This script writes every number the document prints as a macro
(`tables/gate_numbers.tex`, 338 of them), the ladder and comparator table rows, the
newest-render figure names, and one panel (the fitness side: a ridge over the lower-order
terms against the transformer's fitness head on the same held-out triples), from:

- `030/results/lower_order_baselines_030.csv`, `_paired.csv`, `_summary.json` (job 3902)
- `025/results/additive_baselines_025_summary.json` (the gene-identity ladder, the 010
  checkpoints, job 1598)
- `010/results/additive_baseline_gene_interaction.csv`
- `006/results/dcell_training/checkpoints.csv`, `005/results/dango_string_version_summary.csv`,
  `007/results/fba_baseline_si/stats.json` (comparators, each on another build)
- W&B (`--wandb`): `val/fitness/order3/Pearson` of the 030 arms, mean of the last ten
  synced epochs; cached in `results/figure2_gate_fitness_wandb.json`.

The regime of every rung is a column: identity (the gene-identity ridge and the
transformer as trained), lower-order terms (a lookup over measured terms of the same
strains, transductive over the closure), screen only. The document's first section states
the apples-to-apples question the user raised: both sides see the same training records
and neither sees the held-out triple's own fitness or interaction, and what differs is
the access mode; the arm that matches the regimes is E1 of the registry, the transformer
given the same terms as inputs.

Build: `make numbers && make plots && make && make check` in the document directory
(8 pages, gate clean on 2026-10-12). GilaHyper job 4111 refreshes the ladder's W&B rows
so the interaction and fitness windows share the same synced epochs.

## 2026.10.12 - The datasets and splits figure

`notes/assets/drawio/030-datasets-splits.gen.py` writes `030-datasets-splits.drawio`, the
gate's Figure 1: the three builds of the same screens (010 triples only; the 025 closure
pool with merged labels; the 030 per-entry pool), one triple's seven records under the
published identity with filled chips for training records and dashed white chips for
withheld ones on each arm, and the two arms with every count read from the split
artifacts. On arm R only the triple's own fitness and score are withheld; on arm Q the
held-out query pair's double leaves as well, and the singles, the array gene and the
array-side doubles stay. Exported headlessly (draw.io 32.4.1 under xvfb) to
`figures/030-datasets-splits.pdf`, vector with two fonts, 178 mm wide.

## 2026.10.12 - Seed macros

The transformer macros are per seed: seed 0 writes the unsuffixed `\gcgtRValWindow`
form, every seed writes `\gcgtQSeedOneValWindow` and the like, `\gcgtQNSeeds` counts
them and `\gcgtQSeedsLogged` lists each seed's synced epochs and whether it is partial;
the fitness-head macros follow the same rule. `--names-only` rewrites just the
figure-name macros. 380 macros.
