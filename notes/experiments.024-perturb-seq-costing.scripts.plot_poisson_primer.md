---
id: wixyiagiexrh0e72hmi60bq
title: Plot_poisson_primer
desc: ''
updated: 1787625658547
created: 1787625658547
---

## 2026.08.24 - A Poisson primer, on the problem it is used for

`experiments/024-perturb-seq-costing/scripts/plot_poisson_primer.py`

Sec. 7.1 goes straight into a zero-truncated Poisson, which is a big step for a
reader who has not met the distribution. This is the step before it, and it sits
BEFORE the arithmetic rather than after.

The panel order is pedagogical, not the order a statistics course would use:

- **(a)** You cannot hand a cell two plasmids. Setting a concentration sets an
  AVERAGE, and cells sort themselves around it.
- **(b)** Selection deletes the zero class, and deleting the smallest value
  RAISES the mean of what survives, 1.59 to 2.0. This is the counterintuitive
  part, and it is why every lambda in table 17 is lower than the
  plasmids-per-cell figure beside it.
- **(c)** Therefore `k` is a spread to steer, not a number to set.

Panels are deliberately bare. At roughly 40 mm per panel every attempt at
in-panel prose collided with something, so the caption carries the teaching. Two
labels still ran off the right edge in review and are now right-anchored.

Calls `lam_for_target_mean` from `scaling_analysis.py`, so the figure and table
17 agree by construction rather than by coincidence. Renders Fig. 14.

## 2026.08.24 - Moved to Sec. 4.1, and rebuilt to the house bar style

Review round 2 moved the figure from the end of Sec. 7.1 to a new Sec. 4.1,
"Counting Is Poisson, and It Appears Three Times". The distribution is used in
three places and the first of them, droplet loading in Sec. 2.2, is thirty pages
ahead of where the primer used to sit. Sec. 7.1 is now a back-reference. It
renders Fig. 7 rather than Fig. 14, and every figure from 7 onward shifted by one.

Three defects fixed, and the first was a house-style violation rather than a
taste call.

- **Bars now use `PLOT_PALETTE` line colors as FACES with black edges**, matching
  `plot_compression.py` and `plot_economics.py`. This figure had been filling
  faces with the pale `PLOT_PALETTE_FILL` companions and drawing borders in the
  line color, which is backwards: the pale fills are the draw.io object palette
  and are never a bar face. Flagged in review as reading like a different figure
  while using the same palette, which is exactly what that inversion does.
- **Widened from `wide` (118.9 mm) to `full` (179 mm).** At 118.9 mm each panel
  was 33 mm across, which is what put the panel (b) legend on top of its own bars
  and squeezed panel (c)'s in-plot labels against the frame. It is also what
  cropped the (c) panel letter: `place_panel_letters` clamps to y=0.985 and sets
  the glyph `va="bottom"`, so at 8 pt it ran off the canvas. The clamp is a
  backstop, and the layout has to reserve the room, so `top` went 0.86 to 0.84.
- **The panel (b) mean annotation was rendering as a stray tick.** The arrow
  spanned only 0.41 x-units on an axis running to 7.5, so at print size it read
  as a mark rather than a direction, and building the label as
  text-arrow-text put the arrow glyph hard against the following digit, which
  looked like a strikethrough over "2.00". It is now one mathtext run,
  `$\mathrm{mean}\;1.59\,\to\,2.00$`, with both means labeled on their own rules.

The panel (a) label placement is worth recording because three attempts failed
before one worked. "Take up nothing, die on selection" is about 3.4 x-units wide
at this panel width, so over its own bar it runs under the lambda rule; above the
bar it lands on bars 1 and 2; and any STRAIGHT leader from the empty tail back to
a bar top of 0.204 passes below bar 1's top of 0.325. The path that touches
nothing is an elbow, across at y=0.455 and then straight down the x=0 column.
