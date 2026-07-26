---
id: qg7rkqfxgwqulksqfl9pwme
title: Verify_head_decode
desc: ''
updated: 1785040451340
created: 1785040451340
---

## 2026.07.25 - Verifying the three harness decode defects are fixed

Results: `experiments/019-simb-multimodal/results/head_decode_verification.json`

All three defects produced a PLAUSIBLE NUMBER rather than an error, which is why they went
unnoticed - so this verifies decoded VALUES, not just that the code runs.

| check | result |
| --- | --- |
| 1 label collision | on one genotype carrying all three phenotypes, `betaxanthin` decodes 0.7, `mulleder19` decodes exactly its 19 values, `beta_carotene` decodes 3.0 |
| 2 broadcast guard | 19 decoded values into an 18-wide head RAISES ("would BROADCAST rather than align") |
| 2 demo | assigning 1 value into a 19-column row fills all 19 with 0.7 - what the old code did |
| 3 equal-width collision | two heads sharing `metabolite_level` with equal value counts is REJECTED at alignment time |

Check 1 is the one that matters most: betaxanthin and the Mulleder metabolome are both
`MetabolitePhenotype` and therefore both label `metabolite_level`, and the `Perturbation`
processor drops the dict keys. Selecting by label alone concatenated 1 + 19 into a 20-value
blob. `phenotype_sample_indices` survives collation, so the two experiments' value groups
stay separable by size - and `build_head_alignments` now REQUIRES those sizes to be
distinct rather than assuming it.
