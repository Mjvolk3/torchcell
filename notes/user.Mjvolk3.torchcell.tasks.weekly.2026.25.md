---
id: jm7bud7nl1qyuqev3gcmub8
title: '25'
desc: ''
updated: 1781801413593
created: 1781801413593
---

## 2026.06.18

- [x] Stood up the Nature Biotech figure pipeline and added Fig 1 (TorchCell overview), with workflow docs and every figure asset linked so editing can continue on the M1 [[paper.nature-biotech.figures]]
- [x] Added a figure gate (size <=180x170 mm and true-size only) that blocks the paper build when a figure is out of spec [[paper.nature-biotech.check-figures]]
- [x] Created a draw.io-aligned color palette and recolored the Cell Graph Transformer Type I/II mermaid diagram [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]

## 2026.06.21

- [x] Re-rendered the Type I/II mermaid figure with proper typeset KaTeX math and multi-line boxes, using the triple-backslash workaround for mermaid's line-break bug so labels wrap instead of running too wide [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]
- [x] Added a reusable standalone mermaid-to-PDF/SVG/PNG renderer (md-matched names, draw.io-ready outlined SVG) so single diagrams become figure assets without the flaky pandoc mermaid-filter path [[notes.assets.publish.scripts.mermaid_pdf]]
- [x] Added a beige Background swatch to the figure color palette to match the mermaid diagram canvas [[torchcell.models.equivariant_cell_graph_transformer.mermaid.colors]]
