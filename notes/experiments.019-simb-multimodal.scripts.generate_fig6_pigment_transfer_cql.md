---
id: iu3hxpig2r46o10dcaoslev
title: Generate_fig6_pigment_transfer_cql
desc: ''
updated: 1785040429431
created: 1785040429431
---

## 2026.07.25 - Pigment metabolome-transfer build query

Emits `experiments/019-simb-multimodal/queries/fig6_pigment_transfer.cql`.

EXACTLY three datasets, the minimum for the designed positive/negative control pair of
[[plan.cgt-metabolism.2026.07.25]]: `BetaxanthinCachera2023Dataset` (`metabolism`),
`CarotenoidOzaydin2013Dataset` (`global`), `AminoAcidMulleder2016Dataset` (`metabolism`).
Deliberately excluded vs the seven-dataset `generate_fig6_cql.py`: the isobutanol screen +
validated subset, Zelezniak, and da Silveira - none is part of the tyrosine contrast.

The **deletion-in-gene_set** filter is carried over unchanged and is load-bearing: the
pigment strains carry their biosynthesis cassettes as `gene_addition` perturbations whose
heterologous names (`CYP76AD1`, `DOD`, `crtYB`, `crtI`) are not in the S288C `gene_set`, so
the Fig-3 all-in-gene_set clause would drop every pigment record. Verified retained:
4,669 betaxanthin + 4,406 beta-carotene rows survive into the build.
