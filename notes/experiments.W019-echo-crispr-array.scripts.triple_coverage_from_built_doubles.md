---
id: tr1lpphigbyi65px9wthtgk
title: triple_coverage_from_built_doubles
desc: ''
updated: 1787527431638
created: 1787527431638
---

## 2026.08.23 - SUPERSEDED

Ran on the **10-gene / 31-target** basis. YLR104W (LCL2) is both built and a panel-12
prediction node, so the correct basis is 11 genes / 39 targets. The round actually being
built comes from [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]].

Its outputs still on disk are on the superseded basis and must not be used to plan a round:
`results/triple_coverage_by_rank.csv`, `results/triple_coverage_double_ranking.csv`,
`results/triple_coverage_greedy_path.csv`, `results/triple_scorability_by_double.csv`,
`results/triple_scorability_greedy.csv`, and the matching images.

### What survives

The BUILDABLE vs SCORABLE distinction it drew is still the right one and still holds:

- **BUILDABLE** needs >= 1 of the triple's three doubles, since a triple is made by
  crossing an existing double with the third single. This is what the original set-cover
  covered, and it succeeded.
- **SCORABLE** needs ALL THREE doubles, because tau subtracts a digenic term per pair. That
  was never the cover's objective, and no target satisfied it from the built panel alone.

That is exactly the gap the 25 new doubles in
[[experiments.W019-echo-crispr-array.build-list]] close, and it is now asserted by
[[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]].
