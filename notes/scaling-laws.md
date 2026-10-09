---
id: 71ndi41isvdo116g1v9gorl
title: Scaling Laws
desc: ''
updated: 1791533054495
created: 1791533054495
---


## 2026.10.09 - Scaling laws for the paper and the thesis

Working note paired with `notes-tex/modeling/scaling-laws/` (new `modeling/` group). Figure, eight GIFs and the constants table come from [[experiments.041-scaling-laws.scripts.scaling_law_forms]]. The document covers: the Kaplan and Chinchilla forms and the compute-optimal allocation; a 3 x 3 figure that walks through both; how to count `N`, `D` and `C` for genotype-to-phenotype records (instances consumed, not dimensions; epochs are not a multiplier on `D`; subsample by the leaking unit); the minimal ablation and what makes a curve defensible; and a plan for the trigenic build, which has not been run.

Status: every section `todo`. No torchcell number anywhere in it.

### Open items

- [ ] Decide whether Kaplan 2020, Hoffmann 2022, Besiroglu 2024, Yang 2022 (muP) and Muennighoff 2023 go into the library; until then they are named by arXiv id and flagged `\external`.
- [ ] Run the `D`-sweep on the 025 disjoint split (1/64, 1/16, 1/4, 1 by query pair) before anything else; it is the sweep that says whether a new screen or a larger model pays more.
- [ ] Compute the replicate-noise bound on `E` from the Kuzmin/Costanzo SD columns in the loss's own units before fitting anything.
- [ ] Where it lands in the manuscript: a Methods subsection plus one three-panel figure (`L` vs `D`, `N`, `C`); the paper's status chips decide when.
- [ ] A `\cite`-able version needs a Zotero collection named `scaling-laws` and `make bib-pull`.
