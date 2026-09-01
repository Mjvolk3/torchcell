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
