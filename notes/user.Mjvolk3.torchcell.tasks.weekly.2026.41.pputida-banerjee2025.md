---
id: xp6c56iul1axd3jung14g9q
title: pputida-banerjee2025
desc: ''
updated: 1791452425947
created: 1791452425947
---

## 2026.10.08

- [x] Row 39 Banerjee 2025 landed as one proteome loader, 6 records over 13,413 stored abundances, all twelve L0-L4 rows PASS [[torchcell.datasets.pputida.banerjee2025]]
- [x] The row's 36-instance estimate corrected to 6, and its 12 promoter variants to 4 strains: the paper makes TWO promoter swaps, not twelve [[torchcell.datasets.pputida.banerjee2025]]
- [x] Replicate count back-solved from the released t statistic, resolving the source's own conflict: the promoter arm is n=4 (Supplementary Figure 4) and the cross-feeding arm n=3 (Methods), both exact to machine precision [[torchcell.datasets.pputida.banerjee2025]]
- [x] The released group ids `2370` and `2487` mapped onto the strain table by PP_0897's own fold changes, since no quote states the mapping [[torchcell.datasets.pputida.banerjee2025]]
- [x] Titer and growth-rate families refused whole: both are figure-only, and the only per-hour numbers in the paper are flux-balance predictions [[torchcell.datasets.pputida.banerjee2025]]
- [x] Promoter-variant modality needed no new leaf: `PromoterReplacementPerturbation` already exists and already serves as a `bacterial perturbation` [[torchcell.adapters.banerjee2025_proteome_adapter]]
- [x] Raw mirror created for `banerjeeAddressingGenomeScale2025`, five Data S2 members with `zip_member` retrieval records pinned to the archive's own sha256 [[torchcell.datasets.pputida.banerjee2025]]
- [ ] #749 `MeasurementType` has no respiration-endpoint member, which blocks the released 95-row BIOLOG OD595 grid; it is the only measured arm of this paper left unwritten
- [ ] #726 add `D-alanine` and `L-malate` to the compound-identity table, keeping D-alanine distinct from the existing L-alanine row
- [ ] #753 the UniProt-to-locus-tag crosswalk, measured here at 270 of 2,763 dropped labels
