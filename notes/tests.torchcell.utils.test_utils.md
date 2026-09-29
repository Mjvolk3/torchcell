---
id: etxrk8io6blto1dsmxuk3lz
title: Test_utils
desc: ''
updated: 1790649322025
created: 1790649322025
---

## 2026.09.28 - Figure constants, palette, true-size SVG and number formatting (Phase 9)

11 tests: the Nature panel widths and height cap; the eighteen-color palette with the six locked primaries and the 0.73 dark tier recomputed channel by channel (`#D79B00` to `#9D7100`, and so on, equal to `PLOT_PALETTE[6:12]`); `display_label` maps exactly the two species language-model keys; the paper rc style; `savefig_true_size_svg` rescales the root to 100 units per inch (144 by 72 pt becomes 200 by 100, scale `1.3888888889`), is byte-stable, honors a custom dpi and refuses non-matplotlib output; the panel label sits flush left of the tight bbox and 12 pt up, and refuses a detached axes. Finding: `format_scientific_notation` (lines 327 to 353) promises "preserving all significant digits" but compares the rounded reconstruction to the rounded input, so `1234.5` becomes `1.234e03` and `99.9` becomes `1e01`. Coverage 39.8% to 99%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
