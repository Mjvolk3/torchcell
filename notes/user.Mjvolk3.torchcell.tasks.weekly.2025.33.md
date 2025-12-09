---
id: pv3a69tpspow1fz3qxp903b
title: '33'
desc: ''
updated: 1757447566614
created: 1754939102234
---
## 2025.08.11

- [x] ![](./assets/drawio/cabbi-agent-idea.drawio.png)
- [x] We need to save best val pearson correlation model along with best mse [[Hetero_cell_bipartite_dango_gi|dendron://torchcell/experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi]]
- [x] WIP diffusion decoder → Cannot overfit batch which is concerning, but this might be expected with such small batch size for diffusion decoder. [[Hetero_cell_bipartite_dango_diff_gi|dendron://torchcell/torchcell.models.hetero_cell_bipartite_dango_diff_gi]]

## 2025.08.12

- 🔲 Launch minimal diffusion decoder for proof of concept, we are following the ideas from [CausCell](https://github.com/bm2-lab/CausCell) → Got blocked because could not overfit a single batch. It is unclear why this is happening to me due to my lack of familiarity with diffusion models. We mainly took from CausCell. I even tried a linear probe on `z_c` but only on `physical` and `regulatory` and this did not work. → Running small run on 5000 data overnight to see what happens. → [[2025.08.12 - GeneInteractionDiff Model - Diffusion-based Gene Interaction Prediction|dendron://torchcell/torchcell.models.hetero_cell_bipartite_dango_diff_gi#20250812---geneinteractiondiff-model---diffusion-based-gene-interaction-prediction]]
- Plots from training showing little correlation.

## 2025.08.13

1. Triple mutant correlation and query new dataset
2. Morphology dataset
3. Expression dataset
4. Database build

- [x] Check diffusion experiment. → Small diffusion experiment is showing distribution matching in validation, but no correlation. There is degenerate output in train, but this might be do to dummy output on diffusion decoder. Looks promising for distribution matching problem.
- [x] Dataset with fitness and gene interaction on triples. Plot correlation. Slide on how this justifies the inclusion of fitness data for improved prediction performance. → plots [[2025.08.13 - Low Correlation Between Tmi and Tmf|dendron://torchcell/experiments.007-kuzmin-tm.scripts.tmi_tmf_correlation#20250813---low-correlation-between-tmi-and-tmf]]

## 2025.08.14

- [x] Morphology Calmorph label id and description generation as python scripts. → [[2025.08.14 - First Attempt With Open Questions|dendron://torchcell/scripts.generate_calmorph_labels#20250814---first-attempt-with-open-questions]]

- [x] Split between labels and statistics. This changes the original representation but matches torchcell data schema better [[Calmorph_labels|dendron://torchcell/torchcell.datamodels.calmorph_labels]].
