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
