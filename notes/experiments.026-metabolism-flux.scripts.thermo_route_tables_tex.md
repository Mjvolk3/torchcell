---
id: q3rt1qrv34gr6k3847yyico
title: Thermo_route_tables_tex
desc: ''
updated: 1789186925116
created: 1789186925116
---

## 2026.09.11 - Generated route tables for the 026 document

Emits seven LaTeX tables into `notes-tex/026-kinetics-thermodynamics/tables/`: the
controlled vocabulary, the coverage ladder, the metabolite routes, the reaction routes,
the route provenance, the failure classes, the curation records and the accession
mispricings.

Counts are read from `results/reaction_thermo_coverage_final.json`,
`results/thermo_exclusions.json`, `results/thermo_curation_closure.json` and
`results/parameter_coverage.json`. The prose columns are declared in the script, because
a route's description is documentation rather than a measurement, and a route present in
a result file with no entry in the script raises rather than being dropped.
