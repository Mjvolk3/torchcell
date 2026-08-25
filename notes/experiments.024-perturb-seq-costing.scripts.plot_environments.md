---
id: vfxl4578uph81cyzwhsko76
title: Plot_environments
desc: ''
updated: 1787676660678
created: 1787676660678
---

## 2026.08.25 - Fig. 17, environments as the other scaling axis

Draws Sec. 7.3 of [[microbe-perturb-seq.method-review-and-costing]]. One row,
three panels, full Nature width.

- **(a) Environments are the linear axis.** Log axes, so the exponent is the
  slope. Environments cost 600,000 cells each and rise with slope one: 4 need
  2.4 million cells, 12 need 7.2 million. Covering every named pair of a
  T-target panel rises with slope two, so a 200-target panel already costs about
  what 12 environments cost and every further target costs more than the last.
- **(b) The round-1 plate is idle in a single-condition screen.** Split-pool
  spends round 1 on sample identity, so a pooled screen in one condition uses 1
  of 96 wells and the other 95 are bought and unused. Drawn as two 96-well grids
  rather than a bar, which has no label-collision surface.
- **(c) Droplet pays per condition.** Recurring cost against environments on the
  Sec. 5 cost model, at 600,000 usable cells per condition.

**Panel (c) reuses `cost_model.budget_for` rather than introducing a second cost
model.** `environment_cost` in
[[experiments.024-perturb-seq-costing.scripts.scaling_analysis]] composes it
along the environment axis, and the composition is where the platforms differ: a
droplet channel holds one condition and cannot be pooled across conditions, so
channel count is rounded up per condition and summed, while split-pool pools
every condition into the same runs and sublibraries and rounds up once over the
pooled total. Indexed libraries still share lanes in both.

**The check that the composition is right** is that at one environment it
returns the numbers Sec. 5 already publishes: $25,190 for depleted split-pool
and $45,764 for preindexed droplet.

**Preindexed droplet is a third series, added after review.** The first version
drew only un-preindexed 10x at $144,979 per added environment against
split-pool's $22,341, which overstates the case: Sec. 5.7 promotes preindexed
droplet to a co-equal candidate and it comes in at $44,258. Preindexing does not
escape the per-condition floor, since a channel still holds one condition, so it
lowers the slope without changing the shape.

![](./assets/images/024-perturb-seq-costing/environments.svg)
