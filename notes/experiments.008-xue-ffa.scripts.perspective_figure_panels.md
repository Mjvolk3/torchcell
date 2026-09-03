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
