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
