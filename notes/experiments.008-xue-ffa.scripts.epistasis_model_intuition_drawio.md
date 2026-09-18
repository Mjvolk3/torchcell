---
id: x6c86igzh97r10njprsb7yf
title: Epistasis_model_intuition_drawio
desc: ''
updated: 1789598715078
created: 1789598715078
---

## 2026.09.16 - The schematic and the table

Assembles Fig. 3 of the 008 perspective: the three measured panels from
[[experiments.008-xue-ffa.scripts.epistasis_model_intuition_panels]] across the top, then
two native draw.io blocks.

**Panel d, where the two families of null come from.** Left, a flux through two steps, each
keeping a fraction, which is how independent steps of one flux compose and is why the
growth interaction screens use a multiplicative null. Right, one pool that each deletion
takes an absolute amount out of, which is the reading a titer in mg/L invites. The second
counts the part of the first deletion's loss that the second would have taken again, and
that double-counted term, $(1-f_i)(1-f_j)$, is the entire difference between them.

**Panel e**, the four models on one grid: what each expects, the scale its residual is
measured on, what it is fit to, and the reading of the system under which it is the natural
null.

### Layout traps

- The print box must be created with `parent="printbox"`. Left on the default layer it is
  exported, and the page comes out 179.4 x 170 mm plus a border regardless of the content.
- The table columns must sum to the panel width minus the left inset. A first pass summed
  to 179 starting at x = 2 and exported 181.5 mm wide, over the cap.

## 2026.09.18 - Four full-width rows

The page was three rows with the surface panel and the schematic sharing row 2. It is now
four rows, each the full width: a b c at 5 mm, the four models' level sets at 63 mm, the
schematic at 104 mm, the table at 131 mm, content to 168.5 mm of the 170 mm cap.

The schematic is a row of its own and its two halves sit side by side, fractions left and
amounts right, with the identity that separates them centered underneath. Stacked in half
the page width it ran 52 mm tall for 22 mm of content and made the reader meet the second
story only after the first, though the two are a pair.

Table rows went 10 mm to 8 mm and the header 6 to 5.5. Two lines of 5.98 pt type occupy
4.2 mm, so 8 mm is the text plus a millimetre of air above and below it; at 10 mm every row
carried a visible band of empty space (author review).

Every expression in the schematic and in the table is still a rendered math image placed at
its measured size. draw.io's HTML `<sub>` is not typesetting, and it put a serif fallback in
the exported PDF.
