---
id: x1uxj1p9firsjbeakzzhgxc
title: Mermaid_pdf
desc: ''
updated: 1782095909531
created: 1782095909531
---

## 2026.06.21 - Standalone mermaid -> figure-asset renderer

Exists so a single mermaid diagram in a markdown file becomes a paper-ready figure
asset without going through the pandoc `mermaid-filter` path, whose bundled (older)
puppeteer times out launching Chromium. It calls the global `mmdc` (mermaid-cli
v11+, which renders `$$...$$` via KaTeX) and emits a vector PDF plus an outlined SVG
(for embedding in draw.io) and a high-DPI PNG fallback, all named to match the
source md (Dendron fname) so assets stay traceable to their diagram.

- Usage: `bash notes/assets/publish/scripts/mermaid_pdf.sh <file.md> [bg]`.
- `--pdfFit` trims the page to the diagram; `pdftocairo -svg` outlines glyphs to
  paths (zero `<text>`) so the SVG embeds in draw.io without font dependencies.
- Authoring rule for the diagrams it renders (multi-line math needs `\\\`, no
  text/math mixing) lives in [[paper.nature-biotech.figures]]; worked example is
  [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]].
