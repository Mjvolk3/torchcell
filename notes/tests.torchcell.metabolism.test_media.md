---
id: sitwfxn875zo2n52fm7qmty
title: Test_media
desc: ''
updated: 1790773278743
created: 1790773278743
---

## 2026.09.30 - Phase 16: annotation channels, dissociation, bounds, recipe differences

Eleven to twenty-eight tests, 91 to 100 percent. The ChEBI CURIE and bare-numeric annotation channels; salt dissociation with partial and total misses; `apply` closing every exchange then opening exactly the bounds; the larger bound winning on a shared exchange in either order; the supplement rate staying at 0.165 when carbon rises to 10; recipe relations through `diff_bounds` (SC minus SC-Ura is exactly uracil, SC minus YPD-approx is exactly adenine and uracil, SGA DM minus TM is exactly uracil).

Findings: `hydrate` is stripped before `monohydrate` and `dihydrate`, leaving "l-cysteine mono" and "calcium chloride di" (lines 239-248, 339-340); `_normalize` claims to drop punctuation but drops none (316-321).

## 2026.09.30 - Findings retired (issue #538)

- Retired: `hydrate` stripped before `monohydrate`/`dihydrate`; `_normalize` dropping no punctuation.
- Now asserted: the full candidate lists for `L-cysteine hydrochloride monohydrate`, `l-cysteine monohydrate` and `calcium chloride dihydrate`; exact normalized strings that drop `. ; ! ? "` and keep the punctuation chemical names and CURIEs carry.
