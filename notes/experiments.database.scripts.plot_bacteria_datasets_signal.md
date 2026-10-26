---
id: p8nkwkzngrl7r5l3gju5cmf
title: Plot_bacteria_datasets_signal
desc: ''
updated: 1791955924153
created: 1791955924153
---

## 2026.10.14 - Instances vs Signal and the control ratios

View off the bacteria JSON: panel a Instances vs Signal by section with host as marker, panel b each control divided by Signal against Signal. Written to `ASSET_IMAGES_DIR/database/bacteria_datasets_signal.{png,svg}` and embedded in [[experiments.database.scripts.build_bacteria_datasets_table]].

## 2026.10.26 - Host names in italic, PDF output, re-probed after the LMDB rebuild

The host legend entries (E. coli, P. putida) are set in Arial italic through mathtext, and
the script writes `bacteria_datasets_signal.pdf` beside the PNG and SVG. Regenerated from
`results/pre-build/2026-10-26/ecoli-pputida_datasets.json`, a full re-probe of the 100
bacterial datasets because the LMDBs were rebuilt on 2026-10-16 and the 2026-10-14 cache
keys no longer matched; the per-organism totals are unchanged. This is the bacterial
counterpart of thesis Fig. 2.7 (the yeast instances-vs-signal scatter from
`plot_supported_datasets_signal.py`), intended for the thesis appendix.

![](./assets/images/database/bacteria_datasets_signal.svg)
