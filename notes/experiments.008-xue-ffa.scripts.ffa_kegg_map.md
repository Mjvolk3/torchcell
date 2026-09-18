---
id: xy5etpurhz2erqfhnrpmi1g
title: Ffa_kegg_map
desc: ''
updated: 1789696982613
created: 1789696982613
---

## 2026.09.17 - KEGG's yeast global map, redrawn from KGML with Yeast9 overlaid

Fig 5a of `notes-tex/008-xue-ffa-epistasis`. Replaces the iPath3 panel
([[experiments.008-xue-ffa.scripts.ffa_ipath_map]]) after review asked whether the whole of
Yeast9 could be squeezed into a coherent map.

![](./assets/images/008-xue-ffa/ffa_kegg_map.svg)

### Why KEGG's coordinates, not a layout of Yeast9

A coherent map (circular TCA, glycolysis as a spine, the fatty acid comb) is a hand-drawn
artifact. No layout algorithm produces one from a stoichiometric model, Yeast9 ships no map,
and Escher has one yeast map (iMM904 central carbon, BiGG ids) that stops before fatty acid
synthesis. KEGG publishes its yeast global map (`sce01100`) as KGML: 3,878 reaction
polylines and 3,562 compound circles with coordinates, every line carrying the yeast ORFs
and KEGG reaction ids it stands for. This script redraws that as vector in the document's
palette and overlays Yeast9 membership.

### What Yeast9 puts on it, measured against yeast-GEM 9.0.2

| Yeast9 content | on sce01100 |
|---|---|
| genes | 745 of 1,161 drawn |
| reaction lines of the map in Yeast9 (by gene or KEGG reaction id) | 889 of 3,878 |
| compound circles of the map in Yeast9 (by `kegg.compound`) | 684 of 3,562 |

Never on any such map: transport (1,468 reactions), exchange (274), SLIME pseudo-reactions
(188), and most per-chain lipid chemistry, which KEGG draws generically. Panel b carries that
compartment and acyl-chain detail.

### What KEGG's current drawing has that iPath3's did not

Palmitoleate (C08362), stearate (C01530), palmitoyl-CoA (C00154) and all four of ELO1, ELO2,
ELO3 and OLE1 are drawn, so four of the five species are nodes and only oleate (C00712, no
node, nor oleoyl-CoA C00412/C00510) is attached from the GEM (from stearate via stearoyl-CoA
and oleoyl-CoA: FAA1/2/4, OLE1). All 13 pathway genes are on the map; none of the 10
regulators is, read directly from the KGML gene entries rather than probed.

### Drawing

Three tiers: what Yeast9 contains (`#666666`, opacity 0.75, 0.15 mm), what KEGG draws for
yeast or any organism that Yeast9 lacks (`#BBBBBB`, 0.45, 0.09 mm), the route in the module
colors at 0.30 mm. Species nodes are 0.62 mm because KEGG places the measured acids a few
units apart at the end of the fatty acid comb. The map is 83 mm wide (aspect 1.55, so 53.6 mm
tall) to keep Fig 5 inside the 170 mm cap with panel b at 107.9 mm. Labels are placed by
[[experiments.008-xue-ffa.scripts.map_labels]].

### Provenance

`results/kegg_map/`: `sce01100.kgml` (KEGG REST `get/sce01100/kgml`), the seven
`kegg_link_sce_<pathway>.tsv` gene lists, `yeast9_ids.json` keyed by the SBML file's sha256,
`label_anchors.json`, and `provenance.json` with every file's sha256, the counts the key
prints, and the gene rows. `--refresh` refetches and raises on drift. KEGG's KGML is served
under KEGG's academic-use terms; a published redraw cites KEGG, which this document does not
yet do.

## 2026.09.18 - Review round 6: colors, label plates, and a left gutter

### The route colors changed, and the module order is now load bearing

Fatty acid degradation was `#666666`, the same gray the Yeast9 tier is drawn in, so the
arm this chassis engineered away was invisible against the background. It is now brick
`#B85450`, the palette color that had been on the citrate cycle. The citrate cycle went to
terracotta `#C48161` first and that failed a render: dE 22 (CIE76) separates two swatches
and does not separate two 0.3 mm lines running near each other, and the TCA circle read as
more degradation. It is now dark blue `#4F688B`, dE 60 from brick and 25 from the purple
beside it, at the cost of dE 17 from the blue of the measured species, which are five
filled circles rather than lines. The tiers lightened with it: Yeast9 `#8C8C8C` at 0.75,
everything else `#CCCCCC` at 0.5, so no route color is a gray now.

`MODULES` in [[experiments.008-xue-ffa.scripts.ffa_ipath_map]] is a PRIORITY order, first
list wins, and both ends of it matter. Degradation has to outrank the three anabolic
lists: KEGG puts POX1 on `sce01040` as well, and ordered last the whole beta-oxidation
comb took the anabolic amber (measured: 1 red line instead of 21). It must not outrank
central carbon: KEGG's degradation list also carries the alcohol and aldehyde
dehydrogenases, and above `sce00010` it took 20 of glycolysis's 61 lines with it. The
order that holds is central carbon, degradation, biosynthesis.

### Label plates are drawn boxes now

Rounded rectangles, `rx` 0.55 mm, stroked `#666666` at 0.12 mm over white at 0.94 opacity.
The padding is even because the plate is built around the text's INK (`INK_H` 0.74 of the
em, shifted up by `INK_DY` 0.07 for the caps that reach above the x-height that an SVG
`dominant-baseline="middle"` centers on), not around the em box. Around the em box the
plate carried a visible band under the text and nothing beside it.

A plate is also sized on its widest LINE, not on its name. The oleate plate holds
`FAA1/2/4, OLE1` at 5.04 pt under a 6 pt name, and the gene line is the wider of the two:
with no stroke the overflow was invisible, and the first stroked render showed `OLE1`
printing outside its own box.

### A 5 mm gutter, and one narrowed search

Label bounds now span a 5 mm gutter to the left of the drawing that the drawing itself may
not use, because the compounds this panel names sit at the left end of the fatty acid comb
and their labels were being pushed back over it. The map narrowed 83 to 80 mm to pay for
it, which keeps the key's longest line inside 179 mm; the panel is still 54.1 mm tall
because the key, not the map, sets that.

`LABEL_PREF` narrows ONE label's search: pyruvate to upward directions and the two longest
leaders. The ink search still chooses among what is left, so this is a smaller candidate
set rather than a hand-placed label, and it is in the script rather than in a render.
