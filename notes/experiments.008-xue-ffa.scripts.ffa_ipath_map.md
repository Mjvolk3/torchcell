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

## 2026.09.17 - Species parity with panel b, and the regulator check

Review asked three things of this panel: why panel b shows five species and this one two,
that labels stop clashing with route lines, and that the claim behind the figure be
verified: the ten deleted genes are regulators and should not appear on a metabolic map.

### The inspiration is iPath3, not the Yeast9 map

Fig. 2d of Wu et al. (Yeast-MetaTwin) is drawn with iPath3, the paper says so in its
Results ("We then used iPath3 to visualize ..."), so the reference map behind that figure
is KEGG's global map as iPath3 draws it, not the genome-scale model's own map. Panel b's
species come from the yeast GEM, which is why b had all five and a had two.

### Where the three missing species went, measured

Probed one identifier at a time on iPath3 (`ipath_probes.json`):

| species | free acid | acyl-CoA | on iPath3 |
|---|---|---|---|
| C14:0 myristate | C06424 | C02593 | acid yes, CoA yes |
| C16:0 palmitate | C00249 | C00154 | acid yes, CoA yes |
| C16:1 palmitoleate | C08362 | C21072 | neither |
| C18:0 stearate | C01530 | C00412 | neither |
| C18:1 oleate | C00712 | C00510 | neither |

KEGG's own `link/pathway` puts C08362 and C01530 on map01100, and puts the elongase and
desaturase orthology groups (K10245, K10246, K00507) there too, yet iPath3 draws none of
them. The earlier note said the absence was "the KEGG global map's layout"; it is iPath3's
drawing specifically, and the same drawing lacks ELO1/2/3 and OLE1 (0 elements each in the
per-KO probes) while the other nine pathway genes draw 1 to 21 elements.

### The three are attached from the GEM, by shortest path

So that a and b show the same five species, the script collapses the pathway subgraph panel
b reads (`ffa_bipartite_network.graphml`) to metabolites by name, currency metabolites
dropped, substrates joined to products of each reaction (not substrate to substrate: that
put ELO2/3 on the FAS step once), and takes for each missing species the shortest path from
any compound the map draws or any species already attached:

- palmitoleate from palmitoyl-CoA via palmitoleoyl-CoA: OLE1, FAA1/2/4
- stearate from acetyl-CoA or malonyl-CoA via stearoyl-CoA (equal length, the panel takes the
  nearer copy): FAS1/2, FAA1/2/4
- oleate from stearate via stearoyl-CoA and oleoyl-CoA: FAA1/2/4, OLE1

They are drawn as open rings so they cannot be read as nodes of the map, with the genes of
the path on a second 5 pt line under the name, and a dashed link to the anchor.

### The regulator check

For each of the ten deleted genes: KEGG `find/sce/<name>` gives the systematic name (the
entry whose FIRST symbol is the name; TFC7 is the standard name of YOR110W and an alias of
YNL039W), `link/ko/sce` its orthology group, and a single-identifier iPath3 probe counts the
elements that group draws. All ten draw 0. This is what the key and the caption state, from
`provenance.json["genes"]`, not from a hand-written sentence.

### Label placement is now a search, and every rule it needed

Every drawn element is binned into a 0.5 mm ink grid, the route weighted 1 and the faint
background 0.08 (a label over gray lines costs nothing; a label over the route hides the
panel's content). A label tries 16 directions at 4 leader lengths and takes the least-inked
candidate whose plate stays on the map, overlaps no plate, covers no node and no drawn
line, and whose leader crosses no plate and no other leader or link. Labels of a crowded
group (the five intermediates) are placed in the best of all orders rather than greedily.
The attachment column tries both sides (labels left or right of the rings) at every 1 mm
window position and is scored by the ink under it plus the ink each link would cross, with
any window whose link would pass through its own labels rejected. Each rule above was added
after a render showed the failure it prevents (a leader through "palmitate", a plate on the
myristate node, two leaders crossing in an X, an unlabeled second copy of malonyl-CoA at the
start of a link). A compound the map draws twice keeps its large node only at the labeled
copy; the other copy is returned to the map's own faint style.

The map is 89 mm wide (53.2 mm tall at its own aspect ratio) so that with panel b under it
the figure stays inside the 170 mm cap; the key beside it holds four note lines at 3.0 mm.

## 2026.09.17 - Superseded by the KEGG redraw; placement code moved out

Fig 5a is now [[experiments.008-xue-ffa.scripts.ffa_kegg_map]]: KEGG's yeast global map
redrawn from KGML coordinates with Yeast9 overlaid, which has the nodes and genes iPath3's
older drawing lacked. This script stays as the record of the iPath3 panel and still runs;
its label placement now lives in [[experiments.008-xue-ffa.scripts.map_labels]], and the
panel regenerated pixel-identical after the move.
