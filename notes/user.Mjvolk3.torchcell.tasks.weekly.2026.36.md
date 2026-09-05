---
id: wq2nb2dkb5khfx6w5hvxn00
title: '36'
desc: ''
updated: 1788221146558
created: 1788221146558
---

## 2026.08.31

- [x] **eQTL data-model doc revised against all 25 written review comments and republished to Zotero as v2** (`notes-tex/eqtl-data-model`). Full comment-by-key ledger in [[torchcell.datamodels.eqtl-data-model]]. Headlines: Sec 2.6 rewritten, LOD/LD/centimorgan/gene-conversion defined at first use, the mangled schema regex now renders verbatim, and the new interval-keyed perturbation type is recorded as an accepted decision
- [x] The doc now cites through the standard two-tier bib: eight items added by DOI to the personal `eqtl-data-model` collection with pinned citekeys (group twin created), `references.bib` emitted via `zotero_add_ref.py --emit-bibtex` with the `zotero-pending` marker since Better BibTeX is unreachable on GilaHyper; the reviewer's "new eLife paper" is Boocock 2025 itself, and Bloom 2013 stays as the panel Boocock's 393 segregants came from
- [x] Figure re-composed per comments: the gene-conversion tract sat exactly ON marker bm7 while the caption said BETWEEN markers, moved into the inter-marker gap; panel d redrawn as a square gridded matrix with "each dot is one significant (gene, marker) pair: a row of the QTL table" and tied to Sec 1.1 in caption and prose
- [x] Fixed `zotero_comments.py`, which broke when `zotero_publish.py` generalized past notes-tex (`DOCS_COLLECTION` gone); it now walks the repo-path collection route, so it also works for `paper/nature-biotech` [[notes-tex.common.zotero_comments]]
- [x] Checked the mirrored Boocock text for an aneuploidy/CNV screen before answering the reviewer's "do the papers estimate these": no hits for aneuploid/ploid/copy number/disom, so the doc says unmodeled error term for that design and makes no claim about unmirrored Bloom 2013

## 2026.09.03

- [x] SI section on the nine graphs that regularize CGT attention (experiment 010) + STRING-release drift + DANGO replication by release: new Supplementary Note `note:graphs`, two composed SI figures, three script-generated tables. Scripts and notes: [[experiments.010-kuzmin-tmi.scripts.graph_statistics]], [[experiments.010-kuzmin-tmi.scripts.compose_graph_si_figures]], [[experiments.005-kuzmin2018-tmi.scripts.dango_string_version_sweep]]
- [x] Found the SI graphs table was stale (regulatory 3,632/9,753 in the scratch-note transcription vs 6,582/39,636 in the build every 010 run logs at init); the table is now emitted by `graph_statistics.py` and the inline copy is gone
- [x] DANGO STRING 9.1/11.0/12.0 sweep now pulled from wandb (19 runs, best val Pearson 0.415 to 0.427, no release or schedule separates) instead of chart-read values
- [ ] References to add to the paper library when curating: STRING (Szklarczyk), TFLink (Liska 2022), SGD, DANGO (Zhang 2021); the Note currently cites only Kuzmin 2018
- [x] Reconciled the five new Supplementary Notes (7-11: graphs, DANGO reproduction, DANGO full dataset, DCell model, DCell training): prose cut from 4,697 to about 2,700 words (SI pages 36-49 -> 36-47), every measured number kept in prose, a caption, a generated table, or the script's Dendron note; one canonical statement of the experiment-006 vs experiment-010 build mismatch in `note:dango-full` with the other notes pointing to it; the lambda fall-through confirmed from code for the full-dataset v11.0/v12.0 runs (labeled read-from-code, not measured); uniform terminology and `%% REF-TO-ADD` markers for the six missing references; Contents block and outline board updated; stale `fig:dango-string` label in the outline fixed
- [x] `graph_statistics.py` made deterministic (tie-breaks by systematic name, sorted regulator set, pinned float format) after a rerun churned three result CSVs [[experiments.010-kuzmin-tmi.scripts.graph_statistics]]; every SI script's offline path rerun, tables regenerated (caption-only diffs), figures pass `check-figures.sh` and `drawio_font_band.py --check`
- [ ] Author's call: add to the Fig. 2 caption that DCell and DANGO were trained on the experiment-006 build and CGT on the experiment-010 build, and that the DCell error bar is checkpoint spread from one run (sentence proposed in the reconciliation report)

## 2026.09.05

- [x] Second author pass on SI Figs 6 to 11: graphs regrouped by theme (each graph alone vs how they relate), union row shows 45 genes in no graph, STRING-release drift moved to open the DANGO figure, MathJax equations in both schematics, real Kuzmin 2018 triple traced through the DCell ontology, DCell stage table, CPU per-op profile of a DCell step, data-effect panels for DANGO (0.42 on 91k vs 0.36 on 332k) and DCell (0.259 vs 0.173, one run each)
- [x] Counting correction: the cached regulatory graph carried 435 non-vocabulary nodes that `to_cell_data` drops; the prior training used is 6,147 nodes / 37,767 edges, and the union of the nine graphs is 1,514,071 pairs ([[experiments.010-kuzmin-tmi.scripts.graph_statistics]])
