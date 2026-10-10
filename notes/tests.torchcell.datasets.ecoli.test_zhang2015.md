---
id: q7j8vyrr48hjukmazg9flks
title: Test_zhang2015
desc: ''
updated: 1791614211378
created: 1791614211378
---

## 2026.10.10 - Tests for the CeCaFDB provenance record

Tests for [[torchcell.datasets.ecoli.zhang2015]]. Hermetic tests build Download-page HTML
and template-shaped workbook grids, including counterfactuals that carry an interval
column or an uncertainty word, so `loadable_as_flux_with_interval` is shown to flip.
`--data` tests audit the 9 quotes against the pinned PMC text and assert the measured
inventory (31/297 E. coli, 2/3 P. putida, 0 interval workbooks, Haverkorn 190-long label
run) and that the committed results JSON equals a fresh measurement.
