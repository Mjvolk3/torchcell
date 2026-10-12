---
id: i1fhjs4oni0jmi4kp09nexl
title: Kinetic_tier_tables_tex
desc: ''
updated: 1789186940978
created: 1789186940978
---

## 2026.09.11 - Generated kinetic tier tables

Emits `kinetic_tiers.tex` (the four tiers, how each is keyed, what it is evidence about,
and its share of the 3,728 catalytic units) and `kinetic_default_units.tex` (the 18 units
on the organism default, one row each, with the substrate that blocks them).

Reads `results/kinetic_coverage_closure.json`. Substrate names carry braces, brackets and
arrows straight out of the GEM, so every field is escaped before it reaches the table.
