---
id: 4myj9i6uwuljc1w4iym6umj
title: Essentiality_gate_sheet
desc: ''
updated: 1791804118384
created: 1791804118384
---

## 2026.10.12 - The gate's Figure 1, re-laid out from the summary readers

The essentiality gate's proposed main-text sheet, built on `summary_figure.py`'s readers
(imported, not copied, so every value is the same committed result file) with the layout
and the words changed after the review of 2026-10-12: the repeated-splits panel b at
52 mm instead of 72, the per-gene panel e at half width, a legend above every panel and
nothing inside an axes but data, one name per family everywhere (CGT, GCN on GO, CGT +
GCN average) in place of "transformer" and "family 1", and the metabolism gate's status
marks beside each panel letter (red X: section not reviewed; filled circle: every value on
file) with the key at the sheet's foot. Written to
`notes/assets/images/028-gene-essentiality/essentiality_gate_sheet_<ts>.{svg,png}`; the
gate converts the newest render by stem.

## 2026.10.12 - Graph and input named on every bar

Review of the first render: "GCN on GO" read as a graph convolution over the GO graph.
It is not. Every GCN of 028 runs over the union of the nine gene graphs (physical,
regulatory, TFLink and the six STRING channels, `graph_baselines.py`), and "GO" names its
node input, the GO multi-hot; "three inputs" names ProtT5 + pathway multi-hot + FBA
ratios. The sheet now labels every learned bar as family (graph), input, through a
`LABEL` map keyed by the `summary_figure.py` labels that refuses an unmapped label. Panel
b says three seeds per point (a point is a family score, not one run); panel e's axis
reads "share of nonessential genes scored above the gene" instead of "misrank", with the
bold-name rule in its legend; panel c's title and axis name the 4,674 genome-wide genes
and chance. Render: `essentiality_gate_sheet_2026-10-12-07-18-07.svg`.

## 2026.10.12 - Two rows of three, the label as panel a, the status key on top

Review: the status key belongs at the top of the sheet, not its foot; panel a should
introduce the experiment rather than open on the leaderboard; "released 195" is the FCL
held-out set (the 223 genes Merzbacher 2025 held out of its own training, 195 of them in
the cell graph); and the genome-wide arms were trained on the 787 FCL-labeled genes
outside that set (629 train, 158 validation, split `ess_genome_heldout_val0.2_s0`), not
on the 195. The sheet is now 2 x 3: a the label (a genome with one ORF deleted, the
deletion strain on YPD, a budding cell for nonessential and a cell with a skull for
essential; drawn in matplotlib by `panel_concept`, no data), b leaderboard, c resplits,
d genome-wide read with the training counts read from the split file, e misrank, f
enrichment with its class names moved inside the axes. Every panel legend uses "GCN, GO
input"; the top color key names the nine graphs once. Render:
`essentiality_gate_sheet_2026-10-12-07-53-32.svg`.

## 2026.10.12 - Figure 1's yeast cells, real arrows, the key moved into the document

Review of the 2 x 3 sheet: the status key belongs in the document above the figure, not
inside it; panel a should use the yeast cell drawing of manuscript Figure 1 rather than a
drawn oval, three of them; the arrows must be real arrows; "CGT layer-free" is opaque
(it is CGT reading the metabolic solve in its kept, layer-free form); the published
comparators carry error bars because their released per-gene predictions were rescored
on the 195 with the same gene bootstrap, which the labels now say. The three cells are
written out of `Fig1-torchcell-overview.drawio.svg` by `fig1_yeast_cells.py` (orange,
rose, sand) and placed with `imshow`; they are raster in Figure 1 too, which is the one
`make check` warning. Legends name the families by what they read: "CGT + metabolic
solve" and "GCN, GO input"; the top color key names the nine graphs once. Columns are
three of 58 mm, all titles on the panel line, 5 pt tick labels on the bar panels. The
`\statuskey` macro of the root tex prints the key above the sheet. Render:
`essentiality_gate_sheet_2026-10-12-08-12-52.svg`.
