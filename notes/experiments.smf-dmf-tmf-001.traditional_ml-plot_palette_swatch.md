---
id: 37y5n08zc2xyp182gnke5vi
title: Traditional_ml Plot_palette_swatch
desc: ''
updated: 1783996986931
created: 1783996986931
---

## 2026.07.13 - Qualifying the 17-encoding color set before committing panels to it

The classical-ML panels carry 17 gene encodings, each drawn twice: validation as the solid base color, test as that same color at 45% opacity over white. That is effectively 34 colors on one panel, and it creates a real failure mode where one encoding's light test bar reads as another encoding's solid validation bar. This script renders the whole set and measures that risk, so the Extended 17-series was accepted on evidence rather than by eye.

- **No data input.** It *is* the reference: the 17 hex values in canonical size order (`random_1` -> `one_hot_gene`) plus their derived test tints $t(c) = 0.45\,c + 0.55\,w$ (with $w$ = white) -- the exact appearance of a test bar drawn at `alpha=0.45`.
- **Confusability check (the point of the script):** for each test tint it prints the RGB Euclidean distance to the nearest *other* encoding's base color and flags anything under 40.
- **Writes** `notes/assets/images/tradml-plot-palette.svg` (`ASSET_IMAGES_DIR`): one row per encoding, base swatch next to test swatch, both hex-labelled. A matplotlib PNG preview also gets written, but to a hardcoded scratchpad path -- only the SVG is durable.
- A reference/QA artifact, not a result plot. The series it renders is the green-free Extended 17-series of `notes/assets/images/color-palette.svg`, consumed by the RF, SVR, and progression `_palette` scripts.
