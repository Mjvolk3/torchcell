---
id: 0ard6egkfcecug210u3pvok
title: strain-selection
desc: ''
updated: 1791166409071
created: 1791166409071
---

## 2026.10.04 - Twelve Strains on Hand, Six Selected for the First Run

Working note behind `notes-tex/w037-isobutanol-scrnaseq`. The document is the thing
to read; this note records how it was sourced and what is still open.

### Source and provenance

Twelve strains from Zhang et al. 2022, *Nature Communications* 13:270,
doi `10.1038/s41467-021-27852-x`, citation key
`zhangBiosensorBranchedchainAmino2022`.

The tc-lit mirror on GilaHyper was unreachable while this was written, so values were
read from the canonical Zotero PDFs instead. Both are pinned by sha256 in the
generating script and re-verify with `--verify`:

| role | sha256 |
|---|---|
| primary article | `551ba08ec3bf584ef49b72186050d4bb3230df59edfa664a8a6af3460a2a8bec` |
| Supplementary Information | `74de9b6cff654c5d9b6988a0a0c08672eb7defad4f002ecce6b9e6b9fa48824f` |

Local paths, which are Zotero storage and therefore machine-specific:

- `/Users/michaelvolk/Zotero/storage/RW8U8ERB/Zhang et al. - 2022 - Biosensor for branched-chain amino acid metabolism in yeast and applications in isobutanol and isope.pdf`
- `/Users/michaelvolk/Zotero/storage/669BLB7N/41467_2021_27852_MOESM1_ESM.pdf`

A third attachment, `41467_2021_27852_MOESM3_ESM.pdf`, is the Peer Review File and
carries no strain data. The Source Data file, which holds the per-strain figure bar
values, is not in Zotero, which is why four titers in the document are derived from
fold changes rather than stated.

### The JC mapping

The `JC00xx` ids are ours and appear nowhere in the paper, so the mapping has to be
carried rather than re-derived:

| JC | paper | JC | paper |
|---|---|---|---|
| JC0043 | YZy311 | JC0049 | YZy452 |
| JC0044 | YZy313 | JC0050 | YZy454 |
| JC0045 | YZy314 | JC0051 | YZy469 |
| JC0046 | YZy148 | JC0052 | YZy91 |
| JC0047 | YZy148 + LEU4 WT (pYZ149) | JC0053 | YZy502 |
| JC0048 | YZy148 + LEU4dS547 (pYZ154) | JC0054 | SHy134 |

JC0047 and JC0048 are not separately named strains in the paper. They are YZy148
carrying pYZ149 or pYZ154, and the plasmid identities were resolved from
Supplementary Table 2 rather than Table 1.

### Host

All twelve are CEN.PK2-1C, `MATa ura3-52 trp1-289 leu2-3,112 his3-1 MAL2-8c SUC2`.
No S288C or BY4741 anywhere in the paper. This is the fact that decides the
reference genome, and it was the original question.

### Selection

Six selected: JC0052, JC0043 (ILV6 deletion against add-back), JC0044, JC0045
(ILV6 allele in a mitochondrial-pathway background), JC0050, JC0051 (Ll_ilvD allele
in a cytosolic-pathway background). All six ferment 15 percent glucose, so one
protocol covers them, and each pair isolates one variable.

Held back: JC0049 produces only on galactose, JC0053 needs a light schedule, and
JC0046 through JC0048 plus JC0054 make isopentanol.

### Discrepancy found in the paper

Supplementary Table 2 labels pYZ24 a "Modified isobutanol-configured biosensor"
while listing its construct as `His3INT, PLEU1-yEGFP-TADH1`, with no PEST tag. The
paper's own convention is that the PEST tag marks the isobutanol configuration, so
Supplementary Table 1's "modified isopentanol-configured" is the label that matches
the construct. Functionally pYZ24 is the reporter alone with no leucine-insensitive
Leu4p, which is the point, since YZy148 receives LEU4 variants on a plasmid.

### Open items

- [ ] Transform JC0052 with empty pYZ125 so all six share SC-ura. Without it Axis 1
      carries a medium difference on top of its genetic one, and the two cannot be
      separated after the fact.
- [ ] Decide whether Axis 1 needs an isogenic native-level ILV6 wild type. JC0043's
      add-back is driven by P_TDH3, so the pair is deletion against overexpression,
      not deletion against wild type.
- [ ] ddPCR the four delta-integrated strains (JC0044, JC0045, JC0050, JC0051) if
      transgene expression level enters the analysis. Copy number is per-isolate.
- [ ] Confirm which genes the isobutanol YKO dataset covers. If it includes TMA29
      and ILV3 then Axis 3 is as valuable as Axis 1 rather than optional.
- [ ] Decide whether to obtain the paper's actual top producers, none of which are
      on hand: YZy505 (681 mg/L isobutanol), YZy470 (443), YZy312 (378), and YZy148
      with LEU4 mutant 6 (963 mg/L isopentanol).
- [ ] Turn the inline paper reference into a real `\cite` once a Zotero collection
      for it is chosen. Adding the paper to a collection is a curation decision and
      was left alone.

### Regenerating

```bash
python experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py
python experiments/W037-isobutanol-scrnaseq/scripts/strain_tables.py --verify
make -C notes-tex/w037-isobutanol-scrnaseq
make -C notes-tex/w037-isobutanol-scrnaseq check
```
