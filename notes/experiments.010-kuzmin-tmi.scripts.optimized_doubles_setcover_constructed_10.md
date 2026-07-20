---
id: 5m3twsjapxaq8t4s6pelcu6
title: Optimized_doubles_setcover_constructed_10
desc: ''
updated: 1784566118060
created: 1784566118060
---

## 2026.07.20 - Doubles Set-Cover Restricted to the Strains We Actually Built

Exists because the wet-lab plate did not build the full inference_3 panel-12: it dropped
YIL174W and LCL2/YLR104W. The original 11-double cover
([[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover]]) therefore includes
doubles that cannot be made from the strains on hand. This script re-runs the same greedy
minimum set-cover over only the top-k constructible triples whose three genes are all
among the **10 properly-constructed** genes, so the construction plan matches reality.

Result: **8 doubles** cover all 31 within-10 top-k triples (82% fewer than C(10,2)=45,
and 3 fewer than the 12-gene cover — the three dropped doubles were exactly the ones
depending on YIL174W / YLR104W). Genes ordered `gene1 < gene2`; output
`results/optimized_doubles_setcover_constructed_10.csv` carries `triples_enabled` per double.

Full result table + figures: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]].
