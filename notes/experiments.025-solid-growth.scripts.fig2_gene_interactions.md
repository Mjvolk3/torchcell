---
id: taow1y1tg0it8pdetsea2i1
title: Fig2_gene_interactions
desc: ''
updated: 1791912126236
created: 1791912126236
---

## 2026.10.13 - Figure 2 as one twelve-panel sheet in the gate style

Replaces the draw.io composition for review: no frames, every letter followed by the section mark (red cross, nothing reviewed yet) and the figure mark (full circle, every number on file). Primaries only: gray no graphs or a no-interaction null, red the cell graph transformer with its soft prior, orange the hard mask, purple degree-matched random graphs, blue the learned null, yellow other arms. Panels and sources: a definitions; b measured trigenic interaction by bin of the CGT M03 prediction on the 37,673 test triples (r 0.455; `cgt_predictions_M03_c7671wgj_test.npy`); c the state of the art with the SI's current numbers, FBA 0.0019 (the 0.0006 once printed does not reproduce), DCell 0.157 from three checkpoints of one run, DANGO 0.366 over five v9.1 runs, CGT 0.454 over three replicates, and the no-interaction ridge 0.400 as a line; d the additive null on both splits; e the four ways the graphs enter; f the training pool; g the weight ladder at epoch 29; h biological minus random by seed; i validation curves; j edge recall at degree; k the 60-epoch budget; l the composite embedding with and without the prior. Every number is read from a committed result file named in the script's docstring.

![](assets/images/025-solid-growth/fig2_gene_interactions.svg)
