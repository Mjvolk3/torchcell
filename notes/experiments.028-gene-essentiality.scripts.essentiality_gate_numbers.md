---
id: i08gs8agqmrb08xogkq38fe
title: Essentiality_gate_numbers
desc: ''
updated: 1791798413293
created: 1791798413293
---

## 2026.10.12 - The numbers behind the essentiality figure gate

`notes-tex/essentiality-figure-gate` is the gate document for the manuscript's
gene-essentiality figure, built on the pattern of `notes-tex/metabolism-figure-gate`:
the decision, the mockups (the two 028 summary sheets, with the proposed main-text
composition listed panel by panel), the models and comparators, what remains, the
decision rules, a registry of pre-registered items, and every ensemble and genome-wide
arm as tables. This script writes every number the document prints as a macro
(`tables/gate_numbers.tex`, 109 of them), the two long tables (`gate_family_rows.tex`,
`gate_genome_rows.tex`) and the newest-render figure names, from the same six result
files `summary_figure.py` draws: `fcl_comparison.json`, `family_ensembles_folds.json`,
`resplit_readout.json`, `simplification_readout.json`, `genome_heldout_readout.json`
and `dcell_style_folds.json`. Nothing in the document is typed by hand.

One discrepancy it surfaced and the document records: the 028 summary quotes the best
single model at 0.934, which is the one-seed fold ensemble of the four-media form in the
simplification ladder; the fold-ensemble table's best single family is the rule-count
form at 0.930. Both are on file, and the gate's captions say which they use.

Build: `make numbers && make plots && make && make check` in the document directory
(9 pages, gate clean on 2026-10-12). Branch `docs/essentiality-figure-gate`, cut from
`feat/kinetics-equilibrator-datasets` at 6cf91dc48 because the 028 result files live
there; it lands through that branch.
