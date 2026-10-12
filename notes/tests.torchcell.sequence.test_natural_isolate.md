---
id: wkqt6atwt92kgtwgjyhsztr
title: Test_natural_isolate
desc: ''
updated: 1791808482477
created: 1791808482477
---

## 2026.10.12 - What the synthetic panel exercises

Five isolates on hand-written releases (`tests/torchcell/sequence/_peter_tier.py` writes the loss-of-function matrix with CR line ends, the frameshift matrix, the Supplementary Tables workbook with byte-stable save stamps, and a gene-level GFF). Every branch of the builder is reached: a variant, an allele equal to S288C and a gene with no inferred allele under a call; a call on an ORF the presence matrix marks absent; an HO deletion that replaces a natural absence (BBB) and one beside a present HO and its own variant (CCC); 'Xn' ploidy, an empty 'Aneuploidies' cell and a two-chromosome aneuploidy; code-less frameshift rows resolved by a live id, by a one-candidate name, by the SLM6 pin, with a shifted span (LTO1), and refused for no candidate (AAD16), an unpinned ambiguity (DDI3) and a non-overlapping ORF (MOVED). The ledger is asserted field by field. With `--data` (and `--slow` for the last), the same builder over the real tier reproduces the counts #988 and #987 state over the 971 phenotyped isolates: ploidy 750 / 131 / 46 / 38 / 5 / Xn 1, zygosity 501 heterozygous / 470 homozygous, 211 aneuploid, 122 HO deletions (124 of 1,011), 231,339 nonsense/SIFT/frameshift calls on 4,069 genes split 204,038 / 26,510 / 791 by allele status, and 141,453 frameshift calls with the refused rows' calls added back.
