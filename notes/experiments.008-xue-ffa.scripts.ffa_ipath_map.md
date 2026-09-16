---
id: glr127c3zgc7livra8o9ox5
title: Ffa_ipath_map
desc: ''
updated: 1789598722453
created: 1789598722453
---

## 2026.09.16 - The iPath3 panel, and the three things it cost

Panel a of the network figure (Fig. 5a in the 008 perspective): the measured chemistry
drawn on iPath3's reference map of KEGG's global metabolic pathways, cropped to the route
from central carbon out to the fatty acids.

**There is no iPath python package.** iPath3 is a web service at
`https://pathways.embl.de/mapping.cgi`. The client is the script. It takes a
newline-separated selection of KEGG identifiers, each with a color and a width or radius,
and returns the reference map as SVG with those entries restyled.

**Everything selected comes from KEGG REST**, not from a hand-written list of KO numbers:
`link/sce/<pathway>` for the yeast genes of each pathway, `link/ko/sce` for their orthology
ids. Both responses, the exact selection posted, the returned map and the per-compound
probes are stored under `results/ipath/` with sha256, so the panel rebuilds with no network.
`--refresh` re-fetches and RAISES if a stored response has changed, rather than following
the drift.

### Three findings worth keeping

**The per-entry opacity flag is ignored.** The selection syntax accepts `O<value>` per
entry, and the service does not honor it: one `default_opacity` is applied to every entry
including the highlighted ones. So the background is requested faint and the route is set
opaque afterwards, in the returned SVG, by the rule that an element carrying a highlight
color is part of the route.

**The crop has to be anchored on the COMPOUNDS, not on the highlighted reactions.** KEGG
draws one orthology group wherever it occurs, so highlighting glycolysis puts ink in every
corner of the 3774 x 2250 map: the bounding box of the highlighted reactions measured
3483 x 2008, which is the whole map. The compounds are in one place each, so their box is
the region the panel is about.

**Which compounds the map draws cannot be read off the finished map**, because several
share a highlight color. Each is posted on its own in a color nothing else uses and only
the coordinates are kept (`compound_positions.json`). That file is also what places the
labels: the script writes each compound's position in millimetres inside the cropped panel,
so re-cropping moves the labels with it.

### Two of the five measured species have a node on this map

| species | KEGG | on the reference map |
|---|---|---|
| C14:0 myristate | C06424 | yes |
| C16:0 palmitate | C00249 | yes |
| C16:1 palmitoleate | C08362 | no |
| C18:0 stearate | C01530 | no |
| C18:1 oleate | C00712 | no |

The caption says so and nothing is substituted for the three the map does not carry.

### The map's own labels are dropped

At 100 mm the reference map's largest label prints at about 2.5 pt, under Nature's 5 pt
floor, so none of it is type this figure is allowed to set. All text goes, with the colored
pill each label sat on (the pills are the only fully opaque rects in the drawing, which is
what identifies them once their text is gone). The key beside the panel names the modules
at 5.98 pt instead.

### The panel is assembled as SVG, not in draw.io

The cropped map is ~230 kB, which becomes a ~300 kB base64 string in one draw.io style
attribute. The headless exporter fails on that and prints only `Export failed`. Writing the
panel directly, map plus leader lines plus key, keeps it vector and removes the exporter
from this figure's path. Two details of that hand-written SVG:

- `paint-order` is ignored by rsvg, so a white halo written as a stroke with
  `paint-order="stroke fill"` paints OVER the glyphs and the label vanishes. The halo is a
  second copy of the text drawn underneath instead.
- The header must carry `width` before `height` and no units, which is the form
  `notes-tex/common/svg_true_size_pdf.py` reads as the 100-units-per-inch canvas.

See [[experiments.008-xue-ffa.scripts.drawio_doc]] for the export check that made the
draw.io failure visible.
