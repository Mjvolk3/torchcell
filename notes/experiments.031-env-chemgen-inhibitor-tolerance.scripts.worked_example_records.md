---
id: 7l2zsyn1j0lz9lgorkqv3nc
title: Worked_example_records
desc: ''
updated: 1790544462747
created: 1790544462747
---

## 2026.09.27 - One real record per dataset, carried through every transformation

Takes an actual served record from each dataset and prints what the model receives and what it
is asked to predict, so "the five map onto one form" can be checked against real numbers.

| dataset | strain | compound | reports | raw | o_k | standardized |
|---|---|---|---|---|---|---|
| Vanacloig | YBR058C (UBP14) + PDR1/PDR3/SNQ2 host | crystal violet | log2 ratio | -3.40 | +1 | -4.80 |
| Hillenmeyer HET | YBR058C | nocodazole | log2 ratio | +1.92 | **-1** | -4.30 |
| Hoepfner | YBR058C | Epothilon B derivative | sensitivity | -7.17 | +1 | -4.48 |
| Wildenhain | YMR263W (SAP30) | phosphocholine lipid | z score | -81.84 | +1 | -10.87 |
| Hillenmeyer HOM (dropped) | YBR058C | mitomycin C | z score | +56.29 | **-1** | -15.95 |

**The task is REGRESSION on a continuous number.** Nothing is thresholded into sick and
healthy, no class label is formed, no cutoff appears anywhere. Orienting multiplies by +/-1
(a bijection on the real line) and standardizing is affine, so no information is discarded.
This is worth stating because the sign conventions in the joinability work read like a binary
label if skimmed.

**A convention bug this script caught.** The orientation step had been DOCUMENTED as making a
sicker strain more negative while the ARITHMETIC made it more positive. The two are now
separate named quantities: `sick_sign` is what the SOURCE declares and is a property of the
data; `orientation_factor = -sick_sign` is what we multiply by and is a choice. The choice is
that a sicker strain is more negative, matching fitness convention elsewhere in torchcell and
requiring only the two Hillenmeyer arms to flip.

**Two things the table shows without prose.** The Vanacloig strain perturbs four genes because
three are the efflux regulators of the sensitized host (PDR1, PDR3, SNQ2) and the fourth is the
query. And one gene reads -3.40, +1.92, -7.17 and +56.29 across four sources measuring the same
kind of thing. The Wildenhain row is a different gene because its panel is 242 genes and does
not contain YBR058C.

Rendered as table t16 of `notes-tex/031-unified-representation`, beside the controlled
vocabulary (t15).
