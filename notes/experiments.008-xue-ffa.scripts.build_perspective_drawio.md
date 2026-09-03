---
id: k79p01gu6w5p1qfnx1khctf
title: Build_perspective_drawio
desc: ''
updated: 1788413084304
created: 1788413084304
---

## 2026.09.03 - Assembling the perspective figures for draw.io

Embeds the true-size panel SVGs into `.drawio` sources under `notes/assets/drawio/`, so the
panels can be rearranged, relettered and annotated by hand.

```bash
PYTHONPATH=$PWD python experiments/008-xue-ffa/scripts/build_perspective_drawio.py
```

| file | size | panels |
|---|---|---|
| `ffa-epistasis-fig1-interaction-landscape.drawio` | 179.0 x 66.1 mm | a-c |
| `ffa-epistasis-fig2-stepwise-inaccessibility.drawio` | 179.0 x 168.2 mm | a-d |
| `ffa-epistasis-fig3-model-and-scale.drawio` | 179.3 x 140.2 mm | a-e |
| `ffa-epistasis-fig4-network-overlay.drawio` | 131.6 x 170.0 mm | 1, unlettered |
| `ffa-epistasis-figS1-supporting.drawio` | 179.0 x 168.2 mm | a-c |

Every one fits Nature's 180 x 170 mm print box. The script prints the assembled size and
flags anything over.

### Why a generator instead of five hand-built files

Panels are regenerated whenever a plot script runs, and a hand-placed panel would then
silently show the previous render. Re-running this re-embeds the current SVGs at the current
sizes. **Re-running also discards hand edits**, so once a figure's arrangement is settled,
stop re-running it for that figure and treat the `.drawio` as the source.

### Two things that are easy to get wrong

- **Units.** draw.io's canvas is 100 units per inch, so 1 mm = 3.937 units and the 180 mm
  box is ~709 units. `savefig_true_size_svg` already writes width and height in those units,
  which is why a panel's SVG header goes straight into an mxCell geometry. The network
  overlay is the exception: `_rescale_svg_to_mm` writes millimetres, and the script reads the
  unit rather than assuming it.
- **Font size.** draw.io's font-size field is in canvas units, not points, so an 8 pt panel
  letter is typed as `11.1`. Audit hand edits with
  `paper/nature-biotech/scripts/drawio_font_band.py --check`.

### Verification status

Headless draw.io export produced nothing on GilaHyper when this was written, for a
known-good source as well as a generated one, so the files were checked structurally
instead: the XML parses and every embedded payload base64-decodes back to the SVG it came
from. They have not been rendered by draw.io itself.

Related: [[experiments.008-xue-ffa.perspective-epistasis-in-metabolic-engineering]]
