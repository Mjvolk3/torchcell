---
id: htda0chgr6vbserphk5ir76
title: Check Figures
desc: ''
updated: 1781801263730
created: 1781801263730
---

## 2026.06.18 - Figure gate for the WYSIWYG contract

`paper/nature-biotech/check-figures.sh` enforces, in CI/builds, the invariant that makes paper
figures What-You-See-Is-What-You-Get: every exported figure must be placeable verbatim. It exists
so a figure can never silently violate Nature's print box or get rescaled out of its drawn font
size -- the two ways true-size placement breaks. See [[paper.nature-biotech.figures]].

### What it checks

- **SIZE** -- each `figures/*.pdf` must fit `<=180 x 170 mm` (with a 2 mm grace for the draw.io
  guide-box stroke / export rounding). Within the box, the draw.io mm/pt sizing maps 1:1 into the document.
- **SCALE** -- no `\tcfigfit` anywhere in `sections/` or `content.tex`; figures must be placed
  true-size with `\tcfig`. Any scaling re-sizes the PDF and breaks the font-size promise.

Colored `OK`/`OVER` table (red/green on a TTY, plain when piped); exits non-zero on any violation.

### How it is wired (Makefile)

- `make checkfigs` -- strict standalone gate.
- All three manuscript views (`submission`/`editing`/`twocolumn`) depend on `checkfigs`, so
  **no view compiles while a figure is out of spec** (the `make checkfigs` report names the offender).
- `make fig` runs it non-fatally after a force re-export so sizes print on every export.

### Why true-size matters

Because `\tcfig` places figures with no scaling, a font set to N pt at `<=180x170 mm` in draw.io
prints at exactly N pt. If a figure overflows the box, the fix is in draw.io (pull content inside
the box), never LaTeX scaling -- so the gate is also, indirectly, a font-compliance check (Nature's
5-7 pt floor).
