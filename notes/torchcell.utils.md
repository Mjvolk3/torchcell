---
id: eixo5qidh0oar6xakj5tnoo
title: Utils
desc: ''
updated: 1784001492287
created: 1784001492287
---

## 2026.07.13 - Utils package exports

`torchcell/utils/__init__.py` re-exports the utilities used across the repo: `FileLockHelper`, `format_scientific_notation`, and the figure/plotting infrastructure from [[torchcell.utils.utils]] -- `savefig_true_size_svg`, `mm_to_in`, `PANEL_WIDTHS_MM`, `MAX_HEIGHT_MM`, `PLOT_PALETTE`, `PLOT_PALETTE_FILL`. Plotting scripts import these from `torchcell.utils` so the palette and panel-sizing standards have a single source of truth. See CLAUDE.md "Figure & Plotting Standards".
