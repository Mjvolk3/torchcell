---
id: 9koqa8gcdztfdfb9jlxace4
title: Test_viz
desc: ''
updated: 1790649910849
created: 1790649910849
---

## 2026.09.28 - Plate heatmaps, histograms and overlays on the Agg backend (Phase 9)

11 tests (14 cases). Grid geometry, the magma colormap, heatmap structure (x line at 0.5, labels at 0.25 and 1.25, 4.0 by 2.5 inches), options (20 * 0.3 = 6.0), layout codes, a 30-bar histogram summing to 4, shape panel offsets, fitness bars with colors and alpha, plate labels, the DejaVu font, and `label_plate_overlay` (font size `max(30, int(min(5.6, 43, 29))) = 30`, pad 72, tick 18, canvas 294 by 344). `boxplot(labels=...)` at line 179 is deprecated in matplotlib 3.9 and warns under 3.10. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
