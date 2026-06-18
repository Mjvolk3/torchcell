---
id: jjdw91ahk1rouk14rpq8vr2
title: Figures
desc: ''
updated: 1781801256820
created: 1781801256820
---

## 2026.06.18 - Paper figure workflow + asset catalog

Catalog and workflow for the Nature Biotechnology manuscript figures (`paper/nature-biotech/`).
Paper editing is now done **locally on the M1 Mac**; experiments run on gilahyper. This note
documents how figures are produced and links every figure asset so the work can continue on M1.
Related: [[paper.overleaf-workflow]].

### Workflow

draw.io source (`notes/assets/drawio/NAME.drawio.png`) -> `make fig` exports
`paper/nature-biotech/figures/NAME.pdf` -> `checkfigs` verifies it fits the print box and is
unscaled -> `\tcfig{figures/NAME.pdf}` places it true-size -> `make paper` builds the three views
(gated on `checkfigs`) -> `make paper-sync` publishes the curated set to Overleaf.

Commands (from repo root):

- `make paper-fig` -- force re-export every figure referenced in the `.tex` + size/scale check.
- `make paper` -- build submission/editing/twocolumn (refuses to build if any figure is out of spec).
- `make paper-sync` / `make paper-pull` -- push to / merge from Overleaf.

Nature constraints: figure width <=180 mm (column 88 mm), height <=170 mm; in-figure text 5-7 pt;
panel labels **8 pt bold lowercase Arial**. Figures are placed **true-size** (no scaling) via `\tcfig`,
so the size/font drawn in draw.io is what prints. The gate is [[paper.nature-biotech.check-figures]].

### Continuing on M1

- Edit `.drawio.png` sources with the VSCode draw.io extension (or draw.io desktop on Mac).
- Tooling: draw.io (set `DRAWIO` to the desktop/AppImage binary), Tectonic, poppler (`pdfinfo`),
  and `mermaid-cli` (`mmdc`) for the CGT mermaid diagrams.
- After editing a source: `make paper-fig` (re-export + check) then `make paper-sync`.

### Color palette

[[torchcell.models.equivariant_cell_graph_transformer.mermaid.colors]] -- the draw.io-aligned palette
(base primary/secondary + alternates). Swatch: ![palette](assets/images/color-palette.svg)
Source: `notes/assets/drawio/color-palette.drawio`.

### Main-text figures (`paper/nature-biotech/figures/`)

- **Fig 1** -- TorchCell overview (database + software). Source: `notes/assets/drawio/Fig1-torchcell-overview.drawio.png` (panels a-f, assembled from the source images below).
- **Fig 2** -- Trigenic gene-gene interactions (placeholder).
- **Fig 3** -- Expression, morphology, fitness (placeholder).
- **Fig 4** -- Natural variation vs DL design (placeholder; panels to confirm from handwritten plan).
- **Fig 5** -- Metabolism (placeholder; panels to confirm from handwritten plan).

### Fig 1 panel source images (`notes/assets/images/`)

![Neo4j KG browser at NCSA -- panel a](assets/images/torchcell-database-ncsa-illinois-neo4j-query-across-datasets.png)
![Kemmeren knockout expression array](assets/images/kemmeren-expression-array-colored-matrix.png)
![Ohya CalMorph cell-morphology overlay](assets/images/ohya-cell-morphology-software-overlay.png)
![MAGIC CRISPRa/i/d system](assets/images/magic-crispr-aid.png)
![97 kinase KOs / metabolic network (Zelezniak)](assets/images/zelezniak-97-kinase-knockouts.png)
![Costanzo digenic interaction network](assets/images/costanzo-digenic-interaction-network-white-on-black.png)

### draw.io sources (`notes/assets/drawio/`)

- `Fig1-torchcell-overview.drawio.png` -- Fig 1 (assembled overview).
- `src.drawio.png` -- master multi-panel source (reference/experiment data, nested-set cell representation).
- `color-palette.drawio` -- palette swatch (draw.io form).
- `figure-limits.drawio` -- Nature print-box reference card (exports `figure-limits.pdf`, shared to Overleaf).
- `nature-figure-templates.drawio` -- figure size/boundary templates.
- SI figure sources: `TorchCell-Supervised-Learning-and-Teacher-Forcing-Generic-Phenotypes.drawio.png`,
  `fungal_up_down_transformer_upstream_1003bp_pad_for_undersized.drawio.png`,
  `neo4j_cell-conversion-deduplication-aggregation.drawio.png`,
  `ontology_pydantic_hourglass_data_model.drawio.png`,
  `s288c_selecting_gene_sequence.drawio.png`,
  `uncharacterized-genes-profile-enricment.drawio.png`.

### CGT mermaid diagrams

- [[torchcell.models.equivariant_cell_graph_transformer.mermaid]] -- source diagrams.
- [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]] -- recolored Type I/II diagram; vector PDF export `notes/assets/pdf-output/equivariant-cell-graph-transformer-type-i-ii.pdf`.
