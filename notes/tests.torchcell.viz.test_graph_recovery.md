---
id: jjvcshg93u7oufjfpv9zo0e
title: Test_graph_recovery
desc: ''
updated: 1790648881456
created: 1790648881456
---

## 2026.09.28 - Graph recovery plots on hand-built metric dicts (Phase 9)

17 tests on the Agg backend with an autouse figure close and a wandb recorder. The default palette is the `torchcell.mplstyle` prop cycle, asserted entry for entry against line 39 of the style file; an unreadable style or one without a prop cycle falls back to the hard-coded 22-color list, which the comment at graph_recovery.py line 54 calls the "full torchcell.mplstyle palette" although it differs from the style at 8 of 22 indices (1, 6, 9, 10, 11, 12, 13, 15; index 1 is `#CC8250` in the fallback and `#D86E2F` in the style). Each plot function is checked on its data: bar heights, sorted recall order, precision line styles and the two legends, per-graph figures one per graph, the summary table's cell strings (`"[0, 1]"`, `"2"`, `"N/A"`, `"10.00"`) and row shading, the `ylim` of `min(0.4 * 1.15, 1)`, the saved file's exact name, and the wandb call with and without a run. A metric key that does not parse (`weird_Lx_H0`) lands in layer 0. Coverage 4.8% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
