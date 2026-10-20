---
name: fig-swap
description: Audit and refresh the matplotlib SVG panels embedded in a draw.io figure (notes/assets/drawio/*.drawio.svg) without opening the GUI -- swap regenerated panels in at true size, rebuild the preview/PDF, and rebuild editing-nature-biotech.pdf if it is a paper figure. Use after re-running plot scripts, or to check which panels of a figure are stale.
---

Scripted panel maintenance for draw.io figures. The engine is
`notes/assets/publish/scripts/drawio_true_size_import.py`; this skill is the workflow
around it. Everything runs headless -- draw.io the app is never needed, and MUST NOT
have the figure open (see Hazards).

Background: a `.drawio.svg` stores each placed plot as an image cell -- a base64 copy
of the SVG in the cell style, plus a separate `<mxGeometry>` (position + size). Plots
are therefore frozen at placement time and go stale when their generating script
re-renders them. The tool swaps the payload while leaving position untouched, so
layout survives every refresh. Sizes are true physical size: 100 draw.io units = 1
inch; slot widths third=227.6, half=346.5, half_plus=348.4, full=706.7 units.

## Step 0: Preconditions

1. **The figure must be CLOSED in draw.io / VS Code's draw.io editor.** draw.io saves
   whole-file from memory and will silently revert scripted edits (this has happened).
   Ask the author to close/save first if in doubt.
2. Back up the figure to the session scratchpad (`cp` -- plain file copy).

## Step 1: Audit -- which panels are stale?

```bash
python notes/assets/publish/scripts/drawio_true_size_import.py \
  --identify notes/assets/drawio/<FIG>.drawio.svg
```

Each image cell prints as `CURRENT <file>` (payload byte-identical to the file under
`notes/assets/images/`, recursively) or `STALE/unknown` with its geometry and text-label
hints. Interpretation:

- `CURRENT` -- up to date; nothing to do.
- `STALE/unknown` with labels matching a known plot -- stale copy of that plot, OR a
  panel the author drag-imported (draw.io re-encodes bytes on drag-import, so
  drag-placed panels never hash-match even when visually current). Disambiguate by
  comparing the labels/geometry against the candidate SVG, or ask the author.
- Geometry differing from the source SVG's true size (e.g. squashed height) usually
  means the author resized it deliberately -- do NOT "fix" that without asking.

## Step 2: Regenerate the source panels (if the plots themselves changed)

Re-run the generating scripts from repo root (they live in `experiments/<id>/` per the
plot-script frontmatter; `--identify`'s filename tells you which). Verify the new SVG's
header size (`head -c 300 <svg> | grep -o 'width="[^"]*" height="[^"]*"'`) is on the
`PANEL_WIDTHS_MM` grid -- width is STRICT (Nature: 89/183 mm columns; our panel grid).

## Step 3: Swap

Same-size refresh (geometry untouched):

```bash
python .../drawio_true_size_import.py <new-panel>.svg [<panel2>.svg ...] \
  --into notes/assets/drawio/<FIG>.drawio.svg
```

Size changed (match cells by their OLD geometry; rewrites cells to the new true size,
position kept):

```bash
python .../drawio_true_size_import.py <new>.svg --into <FIG>.drawio.svg \
  --match-size <oldW>x<oldH>
```

Multiple SVGs fill matching cells in DOCUMENT order, which is not visual order -- run
`--identify` first and order the SVG arguments to match the cell order it printed. The
tool refuses when matching-cell count != SVG count; never work around that by loosening
the match.

First placement of a brand-new panel: emit paste-XML instead (no `--into`), copy with
`| pbcopy`, author pastes in draw.io -- placement position is a layout decision.

## Step 4: Rebuild derived artifacts (in this order)

```bash
# 1. outer SVG preview (the .drawio.svg's visible layer is stale after any swap)
"/Applications/draw.io.app/Contents/MacOS/draw.io" -x -f svg --embed-diagram \
  -o <scratchpad>/FIG-new.drawio.svg notes/assets/drawio/<FIG>.drawio.svg
# verify before installing (Step 5), then cp over the original
# 2. paper figure PDF (only if the figure is referenced by the manuscript)
make -C paper/nature-biotech figures/<FIG>.pdf
# 3. manuscript (only if a paper figure)
make -C paper/nature-biotech editing
```

## Step 5: Verify -- never report done without this

1. `--identify` again: every swapped cell must now print `CURRENT <expected file>`.
2. Render + look: `qlmanage -t -s 1200 -o <scratchpad-dir> <FIG>.drawio.svg`, then
   Read the PNG and visually confirm the panels (right plot, right labels, nothing
   clipped).
3. If the paper build ran: check `editing-nature-biotech.log` for LaTeX errors
   (`grep -iE '! LaTeX Error|Undefined' ... | grep -iv warning`).
4. Report figure page size if it changed (parse the exported PDF MediaBox). Width
   over the slot is a blocker; height overage is currently advisory (author policy,
   see [[paper.nature-biotech.style-guide]] Figures).

## Hazards

- **Open-in-draw.io clobber**: the app's save reverts scripted edits silently. Close
  first; reload after.
- **Drag-imported cells never hash-match** -- don't misread them as needing a swap.
- **Do not un-squash author-resized cells** via `--match-size` without asking.
- Figure captions may cite panel letters/numbers from the plots -- if a swap changes
  content (not just styling), check the caption in `sections/backmatter.tex` (respect
  the `\secstatus` stoplight; use /paper-edit).

Reference: [[paper.nature-biotech.figures]] (2026.08.16 section) and
[[paper.nature-biotech.style-guide]] (Figures: Nature specs + axis-label style).
\n
## Standards for what goes into a panel (owner, 2026-10-19)

- **Chemical structures follow Nature's ChemDraw stylesheet** (`nr-chemdraw-stylesheet.cds`,
  the "Nature" preset): Arial or Helvetica atom labels at 6 pt, a fixed bond of 0.381 cm
  (10.8 pt), double-bond spacing 18 percent of the bond, a 0.021 cm (0.60 pt) line, 120
  degree chain angle, no labels on terminal carbons, implicit hydrogens on heteroatoms.
  It is not ChemDraw's "ACS Document 1996" preset (0.508 cm bond, 10 pt labels). RDKit has
  only the ACS 1996 preset (`rdMolDraw2D.SetACS1996Mode`), so draw through
  `torchcell.utils.acs_structure` / `acs_structure_svg`, which start from that preset and
  set the bond, label size and font to Nature's (`NATURE_BOND_PT`, `NATURE_FONT_PT`,
  `NATURE_LINE_PT`); never call RDKit's drawer directly for a figure. A sheet that packs
  structures smaller passes **both** `bond_pt=` and `font_pt=`, scaled by the same factor,
  and holds the line width: Communications Physics says a structure "should be scaled in
  proportion ... without changing the bond thickness". Passing `bond_pt=` alone leaves the
  label at 6 pt standing over a shrunken bond, which collapses small molecules (formic acid
  reads as two labels joined by a stub; a salt's counter-ion lands on the carbonyl). Pick
  the scale from the panel: measure the grid's height against the axes before choosing.
  Say "Nature's stylesheet" in captions, not "ACS 1996".
- **Mathematics is set by LaTeX**, never by matplotlib's mathtext: `usetex=True` on the text
  artist, under the PDF backend (Agg has no dvipng on GilaHyper; rasterize the PNG from the
  PDF with `pdftoppm`), preamble `amsmath` + `mathptmx` + `helvet` +
  `\renewcommand{\rmdefault}{phv}` so words are Helvetica beside the Arial labels and the
  math is Times. Not Computer Modern: at a figure's type size its hairlines vanish beside
  Arial, and Times is both heavier and about a fifth narrower, which is the width an
  equation needs to hold a legible size. The Helvetica line comes AFTER `mathptmx`, which
  sets the roman default to Times. A `usetex` size is quantized by LaTeX's font selection,
  so measure the string at the size you will use rather than scaling a measurement. Name
  components under braces (`\underbrace{...}_{\text{...}}`) rather than in prose. The 040
  gate sheet (`experiments/040-inhibitor-synergy-wetlab/scripts/figure_gate_panels.py`,
  `draw_evaluation`) is the worked example. draw.io renders LaTeX too (its math typesetting
  option), so a draw.io figure sets equations the same way, never as plain text.
- **Panel letters obey the white-cross rule**: on a line of their own above the panel, at
  its outer left, the row's letters on one baseline (`letters_status` in the gate sheet);
  one-line titles no wider than the panel; `torchcell.utils.text_clashes` run before every
  save, and the review card (`notes-tex/common/figure_cards.py`) on one page.
