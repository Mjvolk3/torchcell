---
name: fig-swap
description: Audit and refresh the matplotlib SVG panels embedded in a draw.io figure (notes/assets/drawio/*.drawio.svg) without opening the GUI -- swap regenerated panels in at true size, rebuild the preview/PDF, and rebuild editing.pdf if it is a paper figure. Use after re-running plot scripts, or to check which panels of a figure are stale.
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
3. If the paper build ran: check `editing.log` for LaTeX errors
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
