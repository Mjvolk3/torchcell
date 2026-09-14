---
id: ailfbi13qatzl3of17ex12m
title: Perspective Epistasis in Metabolic Engineering
desc: ''
updated: 1788413069715
created: 1788413069715
---

## 2026.09.03 - Draft of a standalone perspective

The typeset draft is `notes-tex/008-xue-ffa-epistasis/`, built as `editing.pdf`. It is
**intended for separate publication** after the Xue free fatty acid study it re-analyzes is
published, which is why it uses the Nature Biotechnology manuscript's house style (sn-jnl
class, sn-nature bibliography style, print-approximate editing geometry, section status
chips) rather than the tcdoc notes style every other `notes-tex/` document uses. When a
convention changes in `paper/nature-biotech`, change it here too.

```bash
cd notes-tex/008-xue-ffa-epistasis
make plots   # true-size panel SVGs -> figures/*.pdf
make         # editing.pdf
```

### The argument

Stepwise strain engineering keeps what improves and discards what does not, which assumes an
improving combination can be reached through improving intermediates. The Xue design tests
that assumption directly, because it is complete: ten transcription factors in every single,
double and triple combination on the pox1/faa1/faa4 FFA chassis, so all 6 x 120 = 720
construction routes are measured rather than inferred.

Measured, on total FFA titer, all from the scripts named below:

| quantity | value |
|---|---|
| single deletions beating the base strain | 1 of 10 (GCN5, 1.058x) |
| double deletions beating the base strain | 29 of 45 (best 1.801x) |
| triple deletions beating the base strain | 51 of 120 (best 2.045x) |
| strictly monotone construction routes | 2 of 720 |
| monotone within 1 SE | 11 routes across 9 triples |
| improving triples at the end of a monotone route | 2 of 51 |
| median valley depth of the shallowest route | 0.314 (31%) |
| greedy campaign endpoint | 1.801x (GCN5 + TFC7), stops at 2 deletions |
| global optimum | 2.045x (RFX1, RPD3, YAP6); every route dips to <= 0.914x |
| trigenic FDR < 0.05 within total titer (multiplicative) | 86 of 119, 11 positive |
| consensus across all four models | 73 of 119, 0 positive |
| graph-by-model enrichment tests at P < 0.05 | 5 of 28, every one a depletion |

The greedy trap has two distinct forms and the note should keep them distinct. Most
improvements are **unreachable**: every route to them passes through a loss. One improvement
is reachable and still **not found**: the monotone route to FKH1/GCN5/MED4 (1.729x) exists,
but at round two TFC7 (1.801x) beat FKH1 (1.529x) and the better local step led to a dead
end.

### Figures

Panels are individual true-size SVGs, one per file, no panel letters, so the arrangement
stays a draw.io decision. Sources:

- `experiments/008-xue-ffa/scripts/perspective_figure_panels.py` -- ten panels
- `experiments/008-xue-ffa/scripts/notable_interaction_selection.py` -- three consensus panels
- `experiments/008-xue-ffa/scripts/ffa_epistatic_path_panels.py` -- the trajectory grids
- `experiments/008-xue-ffa/scripts/create_ffa_multigraph_overlays.py` -- the network overlay
- `experiments/008-xue-ffa/scripts/build_perspective_drawio.py` -- assembles the above into
  `notes/assets/drawio/ffa-epistasis-fig*.drawio`

The document composes the panels itself with `\panelrow` so it builds before the draw.io
arrangements are settled. Once a figure is arranged by hand in draw.io, replace its
`\panelrow` block with a single `\tcfig` on the exported PDF and use `make figures`.

### Open items

- **Authorship is a placeholder.** Any submission must include the source study's authors.
- **Citations are not wired.** No Zotero collection exists for this document, so prior work
  is named in prose. `make check` would fail on any `\cite` key absent from a generated
  `references.bib`, and that file is never hand-edited, so the collection has to come first.
- **Headless draw.io export does not work on GilaHyper right now.** `/tmp/drawio.AppImage`
  extracts, exits 0, and writes nothing, for a known-good source as well as a generated one.
  The five `.drawio` files were verified structurally instead: the XML parses and every
  embedded payload decodes back to the panel SVG it came from. Open them in the VS Code
  draw.io extension to view.
- **The network overlay needs a layout pass, and it is the only figure that does.** Three
  separate problems, all in the same place and all pre-existing:
  1. It is 131.6 x 170 mm, capped by the print box, which puts its labels near 4.4 pt
     against Nature's 5 pt floor. Reaching 5 pt needs about 190 mm of height, which the
     page does not have.
  2. The metabolite labels are right-aligned at their node, so they run leftward **across
     the reaction column**. Raising `MET_X_SHIFT` from 0.8 to 2.6 cleared them off the gene
     labels; it did not clear them off the reactions, because the reactions sit between the
     two columns.
  3. At 131.6 mm in a 182 mm text block it is neither full width nor column width, so it
     leaves a ragged margin.

  The ways out are a landscape re-layout, dropping the `[peroxisome]` / `[ER membrane]` /
  `[lipid particle]` suffixes from the ~33 metabolite labels (which is what drives both the
  width and the height), or accepting it as a full-page supplementary figure. Deciding
  between them is a design call, not a bug fix.
- **`notes-tex/common/Makefile.common`'s `plots` rule oversizes every panel by 1.389.** It
  calls `rsvg-convert` directly, and rsvg reads a unitless SVG length as a pixel and writes
  one PDF point per pixel, so a panel written in draw.io's 100-units-per-inch canvas arrives
  at 100/72 of its intended size with its 6 pt type near 8 pt. This document uses
  `notes-tex/common/svg_true_size_pdf.py` instead. Whether to switch the shared rule over is
  a call for the other documents' owners.

Related: [[experiments.008-xue-ffa.scripts.perspective_figure_panels]],
[[experiments.008-xue-ffa.scripts.build_perspective_drawio]],
[[experiments.008-xue-ffa.figure-candidates]], [[paper.nature-biotech.style-guide]]

## 2026.09.14 - Network overlay rebuilt as a panel

The layout pass listed above is done: Fig. 4 now comes from
`experiments/008-xue-ffa/scripts/ffa_network_overlay_panel.py`
([[experiments.008-xue-ffa.scripts.ffa_network_overlay_panel]]) as
`panel_network_overlay.svg`, 179 x 118 mm at 6 pt, and the Makefile's special overlay rule
is gone because the generic `plots` rule picks the panel up like any other. The draw.io
source `ffa-epistasis-fig4-network-overlay.drawio` embeds the new panel.

One change of content alongside the change of layout: the panel draws every triple
significant on total titer (86: 75 negative, 11 positive) rather than the 47 connected in
the genetic interaction graph, which was a leftover of the enrichment sweep's file naming.
`--graph genetic` restores the restriction. The caption in `sections/results.tex` was
rewritten to match.

The other three figures were checked page by page at this pass and match the standard
(palette, 6 pt Arial, boxed axes, standard widths). Still open: authorship, citations, and
headless draw.io export.
