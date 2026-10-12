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
