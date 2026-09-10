---
id: lvb7utk2k9y6i888vfcykhn
title: Media
desc: ''
updated: 1788314530681
created: 1788314530681
---

## 2026.09.01 - Media ontology to exchange bounds

Maps a torchcell `Media` object onto a `MediaBounds` for any cobra model, resolving each
component to an exchange reaction through the model's own annotations and reporting which
components did NOT resolve rather than dropping them.

**The four recipes work; the ontology objects do not reach them.** SM, SC, SC-URA and
YPD-approx each resolve every component and support growth (0.314, 0.543, 0.539, 0.535
h^-1). But all four datasets emit a name-only `Media` with zero components, so the join from
a dataset to a medium is currently a name string. See [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]] for the seven missing schema
fields.

**The supplement rate follows the source, not our older code.** Suthers et al. set a fixed
0.165 mmol/gDW/h, which is 5 % of their DEFAULT 3.3 glucose uptake; our older code computes
5 % of whatever glucose bound is set, giving 3.03x the sourced value at glucose 10.0.

`YPD_APPROX_FBA` is deliberately not called YPD. Peptone is never modeled, by us or by
Suthers, and the name has to record what it asserts.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]

## 2026.09.10 - The SGA selection media as bounds, and an amino-acid nitrogen source

`SGA_DM_SELECTION_FBA` and `SGA_TM_SELECTION_FBA` expand the ontology's two SGA selection
media (`torchcell.datamodels.media.SGA_DM_SELECTION` / `SGA_TM_SELECTION`, Tong & Boone 2006
recipe #16) into resolvable recipes: the `yeast nitrogen base` line becomes the YNB vitamin
list, the `SC amino-acid supplement powder` line becomes the SC amino acids plus adenine and
uracil minus the medium's dropouts (His/Arg/Lys for DM, plus Ura for TM), the mineral base is
added, and glucose, monosodium glutamate, agar and the four selection agents pass through
unchanged, so agar and the agents come back `excluded_by_role`. Both are in `FBA_MEDIA` under
`SGA_DM` and `SGA_TM`. Tests: `tests/torchcell/metabolism/test_media.py` (a toy model with one
exchange per species; no yeast-GEM download).

**Monosodium glutamate cannot be opened like ammonium.** Measured on yeast-GEM 9.0.2
(`fba_screen_medium.py` probe, 2026-09-10): with the TM medium and glutamate at the
`nitrogen_source` role's unlimited 1000, the optimum is 9.81 h^-1 on 299 mmol/gDW/h of
glutamate uptake, glucose still at 3.3, oxygen 255, ammonium excreted at 103; the model has
switched its carbon source. Capped at 0.165 (the Suthers supplement rate), growth is 0.4966
h^-1 with glucose 3.3 as the carbon source and ammonium excreted at 0.37; at 0.5, 1.0 and 3.3
it is 0.523, 0.560 and 0.727. SM (ammonium, glucose 3.3) gives 0.314. `UptakePolicy` therefore
gains `organic_nitrogen_uptake = 0.165`, applied to a `nitrogen_source` component whose
normalized name is in `_ORGANIC_NITROGEN_SOURCES` (the glutamate and glutamine salts,
proline, urea); `bound_for` and `source_for` take the component name for that reason. The
0.165 is a modeling choice, stated in the bound's source string, not a sourced value; it
coincides with the medium's MSG-to-glucose molar ratio (1 g/L over 20 g/L, 0.053) times the
3.3 glucose uptake (0.176). Leucine at 10 g per 55.2 g of powder against 2 g for the other
amino acids is also flattened to the single supplement rate.

Used by [[experiments.007-kuzmin-tm.scripts.fba_screen_medium]] to rerun the Yeast9 FBA
baseline of Fig. 2d on the medium the trigenic screens were scored on.
