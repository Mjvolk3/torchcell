---
id: 182ydrs9451th7xqmbrspi2
title: Utils
desc: ''
updated: 1783996994333
created: 1783996994333
---

## 2026.07.13 - Make a panel's authored width survive the trip into draw.io

Every paper figure is composed in draw.io from matplotlib panels, and panel **width** is the
one dimension that must be exact -- it is what makes panels tile across Nature's 180 mm page.
That contract was silently broken: a panel authored at 57.8 mm was landing in draw.io at
41.4 mm. This module now owns both halves of the fix, the width vocabulary and the export, so
no plot script has to rediscover either. It is the code behind the "Figure & Plotting
Standards" section of `CLAUDE.md`; the palette half of that standard lives in
`notes/assets/images/color-palette.svg`.

### The bug `savefig_true_size_svg` exists to kill

matplotlib writes SVG coordinates in PostScript points and tags the root `width`/`height` with
the `pt` unit. draw.io ignores the CSS meaning of `pt` and reads the bare numbers in its own
native unit of 100 per inch. A figure authored at `W` inches therefore imports at
`W * 72/100` -- **72% of intended, fonts and all**. That is the whole 57.7 mm-on-disk /
41.4 mm-in-draw.io discrepancy, and nothing about the rendered SVG looks wrong, which is why
it went unnoticed.

The fix rewrites only the *declared* coordinate system: rescale the root
`width`/`height`/`viewBox` to 100-per-inch and wrap the drawn content in a matching
`scale()` group. The picture is byte-for-byte visually identical; only the units it claims
change. So draw.io lands on true mm, and 6 pt text stays 6 pt rather than becoming 4.3 pt.

- **Do not pass `bbox_inches="tight"`** on a fixed-`figsize` panel. It recrops the canvas and
  defeats the width template. It is fine for a standalone tight crop (a legend-only image),
  where the rescale still yields a correct physical size for whatever box matplotlib emits.

### The width vocabulary

`PANEL_WIDTHS_MM` is the enumerated set of widths that tile the page -- `full` 179,
`wide` 118.9, `half_plus` 88.5, `half` 88 (= Nature's single-column width), `third` 57.8,
`sixth` 28.3 -- with `MAX_HEIGHT_MM = 170` as a loose cap on height. Width is strict because
alignment depends on it; height follows the content. Use `mm_to_in()` for `figsize`; do not
hardcode or eyeball inches. `format_scientific_notation` is unrelated and predates this work.

Consumers: the `_palette` plot scripts in
[[experiments.smf-dmf-tmf-001.traditional_ml-plot_svr_palette]],
[[experiments.smf-dmf-tmf-001.traditional_ml-plot_random_forest_palette]] and their
`002-dmi-tmi` twins, which author panels at `third` (57.8 mm) so three stack across a row.

### Figure/plotting standards moved into utils

Added the repo-wide plotting infrastructure so every plot pulls from one source: `savefig_true_size_svg` (rescales matplotlib's 72-dpi points so draw.io imports at true mm), `PANEL_WIDTHS_MM` / `MAX_HEIGHT_MM` / `mm_to_in` (Nature panel-width templates -- strict width, loose height), and `PLOT_PALETTE` / `PLOT_PALETTE_FILL` (one ordered green-free draw.io `(line, fill)` palette matching Fig 1). Rationale and usage rules live in CLAUDE.md "Figure & Plotting Standards".
