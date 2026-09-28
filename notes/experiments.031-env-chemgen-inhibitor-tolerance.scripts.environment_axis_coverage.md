---
id: b6fhhwpufs6sjmpoqo8k35z
title: Environment_axis_coverage
desc: ''
updated: 1790484073616
created: 1790484073616
---

## 2026.09.26 - Wildenhain fixes the chemical coverage, and the dose axis does not pool

Measured over four of the five datasets; Hoepfner was still flattening and is not included.
All 5,378 compounds of the four have a SMILES and embed under count ECFP4 with zero failures.

**Chemical coverage. Adding Wildenhain changes the picture, and it supplies essentially all
of the gain.** Exact overlap with the Vanacloig panel goes from 2 compounds against either
Hillenmeyer arm to 10 against Wildenhain, 11 against the union. The median nearest-neighbor
Tanimoto for a Vanacloig compound rises from 0.33 and 0.32 against the Hillenmeyer arms to
0.596 against the union.

| source | exact matches | median nearest neighbor | non-exact above 0.5 |
|---|---|---|---|
| union of the other three | 11 | 0.596 | 13 |
| wildenhain2015 | 10 | 0.545 | 13 |
| hillenmeyer2008_hom | 2 | 0.327 | 1 |
| hillenmeyer2008_het | 2 | 0.317 | 9 |

**But the coverage is chemically uneven, and it fails exactly where the application needs it.**

| Vanacloig compound | nearest neighbor in the union | Tanimoto |
|---|---|---|
| ferulic acid | ferulic acid | 1.000 exact |
| vanillin | ferulic acid | 0.619 |
| syringaldehyde | syringic acid | 0.600 |
| p-coumaric acid | caffeic acid | 0.545 |
| isobutanol | (r)-(-)-2-amino-1-propanol | 0.467 |
| ethanol | (r)-(-)-2-amino-1-propanol | 0.308 |
| furfural | benzaldehyde | 0.303 |

The phenolics are now genuinely covered at 0.55 to 0.62. The furans and the small alcohols are
not. Isobutanol improves from 0.14 against sorbitol to 0.467 against an amino alcohol, which is
better but is still not an alcohol of the same class, and ethanol at two heavy atoms has no
neighbor any fingerprint can find. Since isobutanol is the application target, pooling buys
phenolic coverage and does not buy alcohol coverage.

**The dose axis does not pool at all, and this is the harder problem.** The three datasets use
three mutually non-convertible dose regimes.

| dataset | dose basis | records with a numeric dose | convertible to molar | distinct doses per compound |
|---|---|---|---|---|
| vanacloig2022 | IC30, and fixed | 3,492 of 143,218 | 0% | 1 |
| hillenmeyer2008_hom | fixed | 99.1% | 83.7% | 1, up to 7 |
| hillenmeyer2008_het | fixed | 99.8% | 95.6% | 1, up to 7 |
| wildenhain2015 | not stated | 100% | 100% | 1 |

Verified directly: **40 of Vanacloig's 41 compounds carry no numeric concentration at all.**
Only benomyl has one, in ug/mL over 3,492 records. The other 136,237 records carry
`dose_basis = IC30` with the concentration unstated, so Vanacloig's dose is a relative potency
rather than a number. Wildenhain is the opposite failure: every one of its 5,170 compounds sits
at exactly 20 uM, so its dose column is a constant and carries no information. Hillenmeyer is
the only arm with real dose variation, spanning 3e-11 to 1.5 M, nine orders of magnitude.

For the 187 compounds dosed in more than one dataset with a molar value, the median ratio of
the highest to the lowest dose across datasets is 12.5, and the 90th percentile is 50. A shared
compound is typically dosed an order of magnitude apart in two datasets.

**Consequence for the design, stated as a judgment.** Dose cannot be one shared continuous
feature. Three routes exist and they are not equivalent. Carry a dose-basis token per dataset
alongside log molar where it exists, which keeps Vanacloig's IC30 as its own basis and accepts
that the model cannot compare across bases. Or drop absolute dose and train on within-condition
gene profiles, which is what the ceiling work already scores and which makes Wildenhain's
constant dose harmless. Or learn a per-compound potency scale from Hillenmeyer's multi-dose
compounds and use it to place the single-dose ones, which is a real modeling project rather
than a preprocessing step.

Result files: `results/chemical_space_coverage.csv`, `results/chemical_space_thresholds.csv`,
`results/compound_overlap_matrix.csv`, `results/dose_axis_summary.csv`,
`results/dose_shared_compounds.csv`
