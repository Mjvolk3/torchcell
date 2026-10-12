---
id: 85irvvyld1ieeqfmau4ed3v
title: Thermo_union_reprice
desc: ''
updated: 1789101200661
created: 1789101200661
---

## 2026.09.11 - What this script is

Writes the staged pipeline's metabolite table back out as
`metabolite_dgf_prime_with_external.parquet` so `parameters.resolve_reaction_delta_g` can
read it through `ThermoSource.UNION_EXTERNAL`, and reports the union reaction table before
and after the dGPredictor unit correction for every reaction the union fills from the
shipped reaction column.

Findings: [[experiments.026-metabolism-flux.yeast9-delta-g-provenance]].
