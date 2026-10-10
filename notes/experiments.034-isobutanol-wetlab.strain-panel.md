---
id: dybkjc0xm5rl5ljxq76mfu3
title: Strain Panel
desc: ''
updated: 1790626222106
created: 1790626222106
---

## 2026.09.28 - Mapping Eleven Tubes onto the Biosensor Paper

Experiment 034 is the wetlab side of isobutanol. Eleven strains from the Avalos
lab are in the freezer as C0043 through C0053, and the first question is which
of them a first round of RNA sequencing carries.

Typeset document: `notes-tex/034-isobutanol-wetlab/034-isobutanol-wetlab.pdf`
Generating script: `experiments/034-isobutanol-wetlab/scripts/strain_panel.py`

### Source

All eleven come from one paper, now fully mirrored with `manifest.json`,
`paper.md` and three SI markdowns:

- `zhangBiosensorBranchedchainAmino2022`, Zhang et al., Nat Commun 13, 270
  (2022), doi 10.1038/s41467-021-27852-x
- `paper.pdf` sha256 `551ba08ec3bf584ef49b72186050d4bb3230df59edfa664a8a6af3460a2a8bec`
- `si/si3.pdf` sha256 `74de9b6cff654c5d9b6988a0a0c08672eb7defad4f002ecce6b9e6b9fa48824f`
  is the Supplementary Information, which carries Supplementary Table 1. That
  table is the only place a full genotype is printed for any of these strains.

The capture was completed by `scripts/lit_sync.py --no-group --personal-root
torchcell-in`, which reported `captured=1 present=269 unsupported=9`.

### What the tubes turned out to be

The color grouping on the tubes maps onto the paper's three screens.

| Group | Tubes | Arm |
|---|---|---|
| green | C0043 C0044 C0045 | ILV6 screen, Fig. 3c Strains A, C, D |
| blue | C0046 C0047 C0048 | LEU4 screen, isopentanol |
| orange | C0049 C0050 C0051 | Ll_ilvD screen, cytosolic isobutanol |
| gray | C0052 C0053 | the two screening hosts |

The green group was the one nothing could be said about from the main text.
Supplementary Table 1 places all three at once: they are three cells of the
Fig. 3c 2x2, two hosts crossed with two ILV6 alleles. C0043 = YZy311 = Strain A
(YZy91 + wild-type ILV6), C0044 = YZy313 = Strain C (YZy363 + wild-type ILV6),
C0045 = YZy314 = Strain D (YZy363 + ILV6 V110E). The fourth cell, Strain B, is
YZy312 and it is not in the box.

### The three matched contrasts

Titers in the panel span four measurement conditions and do not compare across
rows. What does compare:

- C0050 vs C0051, isobutanol, 15% glucose. Ll_ilvD wild type vs I433V. 3.1-fold,
  227 +/- 13 mg/L for C0051. Largest single-allele effect in the panel.
- C0044 vs C0045, isobutanol. ILV6 wild type vs V110E. 1.6-fold.
- C0047 vs C0048, isopentanol. LEU4 wild type vs LEU4 delta S547. 842 +/- 23
  mg/L for C0048, the highest titer of any tube here.

C0043 does not pair with C0044 even though both carry wild-type ILV6: their
hosts differ by the bat2 delta ilv6 delta deletions and by the mitochondrial
pathway at once.

### The P_GAL10 detail that makes a three-point ladder

YZy452 (C0049) is ilv3 delta, and its Ll_ilvD sits behind P_GAL10. Grown in
glucose it therefore has no dihydroxyacid dehydratase at all, while C0050
supplies the wild-type bacterial enzyme and C0051 the improved allele, both
from P_TDH3 on a CEN plasmid. That is three levels of one step in one host on
one carbon source, and it is the backbone of the proposed round.

Consequence: C0049 will not reproduce its published 310 +/- 15 mg/L on glucose,
because that was a 15% galactose number.

### Relationship to the YKO isobutanol work

The dissertation is `lopezSystemsMetabolicEngineering2024` (Jose de Jesus
Montano Lopez, Princeton, November 2024, adviser Jose L. Avalos), which cites
the biosensor paper as its reference 158. Its Supplementary Table 5 gives the
sensor strain as `YKOC, his3::HIS3-PLEU1-yEGFP-PEST-TADH1-PTPI1-LEU41-410-TPGK1`,
which is the paper's isobutanol configuration term for term.

The cassette matches. The background does not: the screen is BY4741 / S288C and
this panel is CEN.PK2-1C. None of the eleven is a strain from that screen, and
the dissertation contains zero occurrences of `YZy`. The strain that would
bridge them is yJM1, BY4741 carrying the same cassette at HIS3, and it is not in
the shipment.

### Proposed round one

Five strains, two independent single-allele contrasts, both isobutanol:
C0049, C0050, C0051 (the Ll_ilvD ladder on glucose), plus C0044 and C0045 (the
ILV6 pair). C0053 YZy502 is the sixth if there is room, because it is the direct
parent of YZy505, the paper's best isobutanol strain at 681 +/- 29 mg/L.

### Three strains worth requesting

- YZy121: CEN.PK2-1C with the isobutanol-configured biosensor and nothing else.
  The panel has no unengineered reference, and this is it.
- YZy312: Strain B, one transformation of C0052 with pYZ228. Completes the 2x2.
- YZy453: YZy452 with an empty CEN URA3 plasmid. C0049 carries no plasmid while
  C0050 and C0051 do, so the plasmid and its marker are currently confounded
  with the dehydratase.

### Open

- No Zotero collection named `034-isobutanol-wetlab` exists, so `references.bib`
  is empty and outside works are given by DOI in prose, following
  `notes-tex/025-additive-baselines`. Creating the collection is a curation
  decision and waits for an instruction.
- The nightly `lit_sync` personal-root default is `torchcell,thesis`, and
  `torchcell` does not resolve. The run used `--personal-root torchcell-in`
  explicitly. Unfixed.
