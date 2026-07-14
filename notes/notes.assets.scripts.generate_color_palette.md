---
id: agl8tojtj8lzj0il67soonj
title: Generate_color_palette
desc: ''
updated: 1784001493946
created: 1784001493946
---

## 2026.07.13 - Generate the one palette reference

Generates `notes/assets/images/color-palette.svg` from `torchcell.utils.PLOT_PALETTE` / `PLOT_PALETTE_FILL`, so the visual palette reference can never drift from the code that plots use. The palette is one ordered, green-free qualitative series of draw.io `(line, fill)` pairs matching Fig 1 (primaries orange/red/purple/yellow, then blue/gray, then darker variants). Re-run `python notes/assets/scripts/generate_color_palette.py` after any palette change. See CLAUDE.md "Figure & Plotting Standards".
