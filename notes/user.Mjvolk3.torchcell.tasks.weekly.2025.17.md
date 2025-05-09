---
id: 1n3kxwub86pbf5c2x5psl0k
title: '17'
desc: ''
updated: 1745956934456
created: 1745258792127
---
## 2025.04.21

- [x] Fix nan coming from `ICLoss` → This loss is highly specialized indexing into `fitness` and `gene_interaction` labels. I instead just used LogCosh
- [x] Run gene_interaction only with logcosh → `Delta`. → This fails. After 107 epochs we don't have any correlation in interaction.
- [x] Annotate fitness reconstruction comparison [[experiments.003-fit-int.scripts.hetero_cell_bipartite_bad_gi_analytic_v_direct]]

## 2025.04.23

- [x] Commit images and summarize finding from STRING. → [[Dango_ppi_vs_sgd_ppi|dendron://torchcell/experiments.004-dmi-tmi.scripts.dango_ppi_vs_sgd_ppi]]

## 2025.04.24

- [x] Added string data to graph → Things are getting a bit messy here and it will be important to get this right.

## 2025.04.25

- [x] Remove gene interactions if they are not in gene set. We show that for `9.1` some of the genes are not in genome. Strangely this is no issue for `12.0` this means that genes must have been revised in new version.
- [x] Add `genome.alias_to_systematic` to `genome`
- [x] Add `TFlink` data

## 2025.04.27

- [x] Query `005-kuzmin2018-tmi` - This dataset will allow full reproduction of Dango. → query issue resolved with more flexible query. [[Issue|dendron://torchcell/experiments.005-kuzmin2018-tmi.query.issue]] → It would be interesting to look to see if predictions differ on perturbations that are `deletion` vs `allele`.

