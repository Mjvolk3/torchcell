---
id: h4eg97spis0cmjsilm1tzrf
title: Interaction_graph_enrichment_analysis
desc: ''
updated: 1786496527181
created: 1786496527181
---

## 2026.08.11 - Make Enrichment Analysis Runnable Off the Original Mac

Tests whether the significant FFA epistatic interactions are enriched on known biological
graphs (STRING and friends). Two absolute Mac paths blocked it elsewhere: the results
directory and the matplotlib style file. The results path now resolves from
`EXPERIMENT_ROOT`, and the style file is located relative to the installed `torchcell`
package (`osp.dirname(torchcell.__file__)`) rather than a checkout-specific literal, so it
follows whichever checkout is on `PYTHONPATH` -- which matters because this work happens in
a worktree, not the primary tree.

Part of making the whole 008 script set machine-portable so the FFA analysis can continue
on GilaHyper -- see [[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]].
