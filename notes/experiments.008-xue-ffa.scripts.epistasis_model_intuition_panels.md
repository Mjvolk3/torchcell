---
id: zaudq24mn5lf82591hnto85
title: Epistasis_model_intuition_panels
desc: ''
updated: 1789598707632
created: 1789598707632
---

## 2026.09.16 - The three measured panels of the epistasis-model figure

Fig. 3 of the 008 perspective exists because the four nulls were four lines of algebra
behind the main text, which reads as one default plus three robustness checks. They are
four different statements about how two deletions combine. The schematic and the table are
native draw.io cells ([[experiments.008-xue-ffa.scripts.epistasis_model_intuition_drawio]]);
these three panels are the measured part.

| panel | what it answers | measured |
|---|---|---|
| a | what does each null expect | MED4-RPD3, the pair whose two expectations are furthest apart; the gap is 0.149 |
| b | which null do the data follow | 43 of 45 doubles sit above both, median residual +0.40 multiplicative and +0.45 additive |
| c | what noise model does the titer justify | replicate SD against strain mean over 165 strains, fitted log-log slope 1.10 |

Panel c is the load-bearing one: slope 0 is a constant spread and slope 1 is a spread
proportional to the mean, which is the Gamma family and is what makes a log link the
natural choice. At 1.10 the data say so rather than the Methods asserting it.

Panel a carries an identity, not a property of that pair: the additive and the
multiplicative expectation differ by exactly $(1-f_i)(1-f_j)$, the part of the first
deletion's loss that the second would have taken again.

### Two rendering notes

- The proportional-to glyph survives matplotlib but not the draw.io SVG embed, where it
  renders as a slash. Words instead.
- `apply_paper_style` sets `mathtext.fontset=custom`, which leaves `mathtext.cal` at
  cursive and prints a findfont warning on every run. Harmless here, no calligraphic
  symbol is used.

## 2026.09.18 - The surface panel became four level-set panels

The 3D surface pair was hard to read and carried a floating `0.25` beside a short rule that
read as an unexplained `I 0.24` in the PDF (author review). It is replaced by
`panel_level_sets`: one axes per model, in the table's order, level sets of what that model
expects of a double over the square of its two singles.

Two differences are visible and they are different kinds of thing.

- The additive surface is not the multiplicative one. Straight level sets against
  hyperbolas, and the two differ by `(1 - f_i)(1 - f_j)`.
- The two log-scale models expect the multiplicative surface and differ in where equal
  residuals lie. Each panel's contours are spaced evenly ON THE SCALE THAT MODEL MEASURES
  ITS RESIDUAL ON: 0.2 to 1.0 in steps of 0.2 for the linear pair, the same span in five
  equal steps of the logarithm for the log pair, so those crowd near 1 and open out low.

Every panel marks (0.5, 0.5) and prints what that model expects there: 0.25, 0.00, 0.25,
0.25. That is the old floating number, said out loud, and it is the whole comparison in one
row.

Contour labels are placed with `manual=` at the point of each contour on the diagonal,
which for both surfaces is that contour's closest point to the origin (`sqrt(L)` for the
multiplicative surface, `(L+1)/2` for the additive one). Left to matplotlib the labels
landed wherever a contour left the axes and were clipped to `631` and `.1`.

### Titles are one line each

A title wider than the panel is NOT wrapped by matplotlib and is not clipped by the axes
either: it runs off the figure canvas and the draw.io embed cuts it at the image edge. Panel
b lost its two median residuals to the caption for this reason.

## 2026.09.18 - Review round 7: colors, notation, and a 1.5 square

- Model colors moved: multiplicative is now blue (`PLOT_PALETTE[4]`) and log-OLS orange (`PLOT_PALETTE[0]`); additive stays brick and GLM lilac. Orange against brick did not separate at 5 pt marker size in panel d, and those two nulls are the pair every panel contrasts. `perspective_figure_panels.MODEL_COLORS` carries the same four so Fig. 4a agrees. The measured double in panel c is black.
- Panel e (mean-variance) uses the same symbol as c and d: the axis is `f`, the y axis "replicate SD of f", and the three lines are named by the assumption ("fit: SD ~ f^1.10", "SD ~ f: the log-scale models", "SD constant: the linear-scale models"). Ticks are plain numbers (0.5, 1, 2 and 0.01, 0.1) instead of the log formatter's `6 x 10^-1`. A `\propto` was tried and Arial has no glyph for it; the SVG showed a slash.
- Level sets run to 1.5 on both axes (single deletions in this design raise titer). Levels: 0.3 to 1.5 in steps of 0.3 for the linear-scale models, a doubling apart from 0.1 to 1.6 for the log-scale ones, chosen so the top contour crosses the square as an arc rather than a corner scrap (at 2.0 the additive surface touched only the corner and its label sat on the 1.6). Labels sit on the diagonal unless that point is within 0.2 of the marked center, in which case they go on the ray `f_j = 1.8 f_i` (`label_point`). The "expects ... at (0.5, 0.5)" note is bottom-left, left-justified, on a white ground.
- Heights: the three measured panels 38 mm (were 52), the level sets 31 mm (were 36), to pay for the new top row of the figure. Two new equation images, `f_ij` and `eps_def`, for the drawio's panel b.
