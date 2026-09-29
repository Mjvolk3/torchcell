---
id: e4xeabiyabvhn7qkxprb2b3
title: Test_transformer_diagnostics
desc: ''
updated: 1790648889142
created: 1790648889142
---

## 2026.09.28 - Transformer diagnostics panels and the zero-ratio decompression bomb (Phase 9)

8 tests. The six-panel figure is checked series by series: plotted data, colors, labels, text annotations, log scales and x ticks. Finding, reproduced end to end by the audit: with `residual_ratios=None` (the default, and what `int_transformer_cell.py` line 573 passes whenever its accumulator is empty) every ratio is 0.0, drawn on a log axis and labeled with `ax.text(layer, 0.0, ...)`; matplotlib clips log10(0) to -1000 decades, the tight bounding box becomes about 3100 figure inches tall (measured 16.1 by 3098 inches), `savefig(bbox_inches="tight", dpi=300)` succeeds with a 4842 by 929,480 pixel PNG (4.55 gigapixels, 20 MB) and `PIL.Image.open` raises `DecompressionBombError`. The test asserts the bounding-box geometry and bypasses the save. The palette comments at lines 296 ("Teal" for `#775A9F`, purple) and 340 ("Purple" for `#A05B2C`, brown) are wrong; line 307 is right. Coverage 9.0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
