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
