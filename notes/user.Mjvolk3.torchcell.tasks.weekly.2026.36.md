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

## 2026.09.02

- [x] **tc-lit `/bib` endpoint built**: a bibliography store in the mirror (`_bib/`) exported nightly over the Zotero Web API and served with sha256 manifests, so a document's `references.bib` is pulled (`make bib-pull`) rather than regenerated per machine through Better BibTeX [[torchcell.literature.bib_store]] [[scripts.lit_bib_store]] [[scripts.lit_bib_pull]]; the flows table and the closed expectation gap in [[tectonic-builds-and-bibliographies]]
- [ ] **Curation pass before any `bib-pull`**: every committed `references.bib` is behind Zotero and each build cites keys its collection no longer holds (paper 4 of 17, 024 costing 13 of 51, eqtl 7 of 8; the eqtl group collection is empty and its Makefile's personal collection is the publish one). Key-by-key table in [[tectonic-builds-and-bibliographies]]; nothing was pulled or written to Zotero
- [x] **Read the banked 026 flux-layer runs and found the comparison is not yet measuring anything.** Over all 18 completed (arm, seed) runs at 20 epochs, validation Pearson peaks at a median epoch of 4.5 and the mean of the last five epochs is at or below zero in 13 of them. The reported per-arm score is a maximum over epochs, an upward-biased order statistic, and the new `n_val_betaxanthin` counter measures 353 validation observations, so a single Pearson there has null width 0.0535 against arms separated by 0.004 to 0.08
- [x] Overnight sweep launched on all four GilaHyper GPUs, jobs 1568-1571, 12 h wall with an 11 h internal budget so each job ends on a complete run: `null` calibration by target permutation, a regularization grid on `flux_anchored`, and the five arms at ten fresh seeds [[experiments.026-metabolism-flux.scripts.sweep_flux]]
- [x] Scoring script written ahead of the results so the sweep is readable on arrival: peak, peak epoch and last-five mean side by side, an empirical p-value against the permuted null, and never a bare maximum [[experiments.026-metabolism-flux.scripts.analyze_sweep]]

## 2026.09.03

- [x] **Experiment 025 dataset note written**: the full record of the uncapped KG rebuild (job 1558, 83.0M nodes / 310.9M rels), storage tiers, local serving, the 13,525,071-group all-solid-growth build, and the exact 010 split transfer (376,732/376,732) in [[experiments.025-solid-growth]]. 025 is the named progression of 010-kuzmin-tmi: same triples and splits, plus all fitness and all interactions around them. Next: train over all fitness + interactions with the pinned splits
- [x] **025 training-campaign plan revised and written** after reading the 010-positive-panel report, the additive-baseline analysis (B1 ridge 0.400 random-split vs 0.127 query-pair-disjoint, CGT never refit there), and the hard-mask/virtual-instrument CGT code: subset ladder S0-S6 with fixed tmi evaluation, both split regimes, closure-based double selection with the leakage rule, one sum-pooled PerturbationHead across orders as the general-equation claim, and the S6 zero-shot order-transfer arm [[experiments.025-solid-growth.training-plan]]
- [x] **Recapitulation gate implemented and run**: recompute tau from 025's own smf/dmf/tmf against stored tmi (aggregate vs Kuzmin-sourced variants, digenic check, closure coverage, positive-call confusion) [[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]]; subsets + query-pair-disjoint split materialized by `subset_definitions.py`
- [x] **Gate result: 025's fitness cannot numerically reconstruct its own tmi.** r = 0.230 aggregate / 0.354 Kuzmin-sourced / 0.213-0.272 stored-dmi, at 99.99% closure coverage, digenic slope 0.99, essentiality contamination ruled out (<= 0.06); the additive null fit to labels (0.400) beats the physical equation on stored values. Thesis mechanism revised to representation transfer; S6 zero-shot promoted. Label parity 010 vs 025 exact (376,732/376,732, gi identical to float precision; 010 has no fitness labels)
