---
id: bjmhctf3ytodt0us9lw412y
title: Test_register
desc: ''
updated: 1790649942053
created: 1790649942053
---

## 2026.09.28 - Plate orientation resolution (Phase 9)

5 tests (8 cases): the dihedral operations (rot180 reverses both axes as n + 1 - i, flip_v rows only, flip_h columns only) and the error naming an unknown op; `resolve_orientation` picks rot180 as the only op that puts the empty image spot on the blank well (agreement 1.0); with every colony present the blank disagrees under all four ops (5/6 each) and the tie-break is pinned; with `inner_row0 = inner_col0 = 1` plate and image coordinates coincide, so only the identity scores 1.0 (rot180 0.5). Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
