---
id: cg3ffquoe3icdundnnjbh9m
title: next_doubles_selection
desc: ''
updated: 1787527424420
created: 1787527424420
---

## 2026.08.23 - SUPERSEDED

Ran on the **10-gene / 31-target** basis. YLR104W (LCL2) is both built and a panel-12
prediction node, so the correct basis is 11 genes / 39 targets. The round actually being
built comes from [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]].

Its outputs still on disk are on the superseded basis and must not be used to plan a round:
`results/next_doubles_strategies.csv`, `results/next_doubles_picks.csv`,
`results/next_doubles_gene_coverage.csv`, `results/next_doubles_milestones.csv`, and
`notes/assets/images/W019-echo-crispr-array/next_doubles_selection.{png,svg}`.

Kept for history. The four-objective comparison it introduced (rank / count / balanced /
no_ylr) carried forward into the live script.
