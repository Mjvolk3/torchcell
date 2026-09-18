---
id: 26ww8zuh23xpwp5j6opdkqr
title: Perspective_figure_panels
desc: ''
updated: 1788413077031
created: 1788413077031
---

## 2026.09.03 - Ten publication panels for the epistasis perspective

Builds every panel the perspective uses, from the committed result CSVs only. It fits
nothing: the four epistasis models and the path enumeration are upstream, which is what lets
a restyle happen without a refit.

```bash
PYTHONPATH=$PWD python experiments/008-xue-ffa/scripts/perspective_figure_panels.py
```

| panel | width | what it shows |
|---|---|---|
| `panel_interaction_distribution` | third | digenic and trigenic scores on one axis; median +0.40 to -0.72 |
| `panel_volcano_total_titer` | third | 86/119 at FDR < 0.05, 11 positive |
| `panel_model_agreement` | half | UpSet over the four models; 97 union, 73 all-four |
| `panel_scale_scatter` | third | linear vs log tau; the upper-left quadrant is empty |
| `panel_scale_slope` | third | 15 of 27 linear positives lose the sign on the log scale |
| `panel_improving_by_order` | third | 1/10, 29/45, 51/120 beat base; best 1.06/1.80/2.05x |
| `panel_greedy_walk` | third | greedy stops at 1.80x; the optimum needs a loss first |
| `panel_path_accessibility` | third | 2 of 51 improving triples end a monotone route |
| `panel_path_valley` | third | median shallowest route dips 31% |
| `panel_graph_enrichment` | half_plus | 5 of 28 tests significant, every one a depletion |

### Conventions the panels hold to

- **One file per panel, no panel letters, no shared legend.** Arrangement and lettering are
  draw.io decisions; baking either in removes the flexibility storyboarding needs.
- **Sign encoding is fixed across every panel**: amber = positive interaction, brick =
  negative. Third and fourth categories take lilac and steel.
- **Headroom is reserved, not inferred.** `bar_headroom` sets the axis top from the tallest
  bar so a value label never lands on the title and an in-axes legend never lands on a bar.
  Every legend collision in the first version came from letting matplotlib pick the top.
- **Titles state a measured fact and derive it from the same numbers the marks are drawn
  from.** The enrichment title's "every one of them a depletion" is computed, not asserted,
  so it cannot go stale if the numbers move.

### Two joins that silently return nothing

Both bit this experiment before and both are handled by helpers at the top of the file.
Gene sets are written `RPD3_SPT3_YAP6` by the linear-scale scripts and `FKH1:GCN5:MED4` by
the regression scripts; readouts are written `C14:0` and `C140` by the same two families.
Only `Total Titer` agrees between them, which is why the defect went unnoticed while the
work stayed on that readout.

Related: [[experiments.008-xue-ffa.perspective-epistasis-in-metabolic-engineering]]

## 2026.09.18 - Review round 6: three small panel fixes

- `panel_volcano` (Fig 1b): the BH threshold label is an `annotate` with a 2 pt downward
  offset instead of a `text` sitting on the line. At `va="top"` on the line itself the cap
  heights touched the dashes.
- `panel_improving_by_order` (Fig 2a): `C_BEST` moved from amber `PLOT_PALETTE[0]` to lilac
  `PLOT_PALETTE[2]`. The right axis label and its tick numbers take the line's color, and
  amber type on white does not read at 6 pt (author review). The earlier comment said lilac
  reads as a third category here; unreadable beats miscategorized.
- `panel_greedy_walk` (Fig 2b): the base strain the campaign starts from is now its own
  marker, filled `PLOT_PALETTE_FILL[1]` with a brick edge, so the start is a member of the
  campaign's series rather than one more step of the walk.

## 2026.09.18 - Review round 7: model colors, and Fig. 2's line and star

- `MODEL_COLORS`: multiplicative blue, log-OLS orange (see the panels note above for why); Fig. 4a's dots follow.
- Fig. 2a's best-titer line is blue (`C_BEST = PLOT_PALETTE[4]`); lilac and orange were both tried and not kept. Fig. 2b's star takes the base strain's pale brick fill with a brick edge, so the two endpoints of the panel are one kind of thing. The earlier "start ... background color 2" review note was about the star.
