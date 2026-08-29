# figures/

Final figure assets for the manuscript. Compose in **draw.io** (diagrams.net;
the "Draw.io Integration" VS Code extension edits `.drawio` files inline).

## Workflow

1. Create `figN_name.drawio` here (e.g. `fig1_overview.drawio`).
2. Lay out panels to Nature print size: **180 mm** full-width or **88 mm** single
   column, max **170 mm** tall (see the outline's figure-prep guidelines). Use
   `figure-proto.tex` to check the full-page fit.
3. Export to vector PDF: draw.io `File -> Export as -> PDF` (crop to drawing,
   uncheck page background) -> save as `figN_name.pdf` here.
4. In the section file, swap the `\figph{...}{...}` placeholder for
   `\includegraphics[...]{figures/figN_name.pdf}`.

## True-to-size (WYSIWYG) -- figure prints at the size you drew

To make the figure render at the size you designed in draw.io, do NOT cap it with
`height=`; control it by a width that equals your design width, on a vector PDF:

- full-width figure designed at 180 mm:   `\includegraphics[width=180mm]{figures/figN.pdf}`
- single-column figure designed at 88 mm:  `\includegraphics[width=88mm]{figures/figN.pdf}`
- pure WYSIWYG (no scaling at all):        `\includegraphics{figures/figN.pdf}`
  -- renders at the PDF's own exported size; verify draw.io exported at true scale once.

Rules:
- **Vector PDF only.** A raster PNG's physical size depends on flaky DPI metadata,
  so true-to-size is unreliable -- use it only as a draft stopgap.
- **Never mix `width=` and `height=` with `keepaspectratio`** -- the smaller scale
  wins, so a tall figure gets capped by the height and is no longer true-to-size.
- **Design within the print box** (180 mm / 88 mm wide, <= 170 mm tall) so the
  true size still fits the page.

## draw.io sizing (design at TRUE size)

`\includegraphics[width=\textwidth]` rescales the exported PDF to 180 mm
regardless of the draw.io canvas, so the only risk is fonts scaling out of the
5-7 pt range. Avoid it by drawing at final size. draw.io default = 100 units/inch
(1 mm = 3.94 units):

| Final | mm | draw.io units |
|-------|------|---------------|
| Full width | 180 mm | ~709 units wide |
| Single column | 88 mm | ~347 units wide |
| Max height | 170 mm | ~669 units tall |

Draw a full-width figure to ~709 units wide, then export crop-to-drawing -> PDF
and include at `width=\textwidth` (scale ~1, fonts intact). Drawing much wider
than 709 shrinks your text further on import. Aspect ratio of the drawing sets
the final height (must stay <= 170 mm at 180 mm wide).

### Font size: the draw.io number is NOT points

The same 100-units-per-inch canvas applies to the font-size field, which is why
this is the easiest thing in the pipeline to get wrong. A point is 1/72 inch, so
**the number you type is multiplied by 0.72 on the page**: a label set to `8`
prints at 5.76 pt, not 8 pt.

Measured rather than inferred: `Fig1-torchcell-overview` is 707 units wide and
its exported PDF page is 509 pt, i.e. 0.7199 pt per unit, and `\tcfig` places it
at natural size so nothing rescales it afterwards.

    rendered pt = drawio number x (placed width in pt) / (canvas width in units)

So type `target pt / 0.72`. Nature allows 5-7 pt for figure text, and 8 pt bold
lowercase for panel letters only.

## The ladder: four values, and nothing in between

Do not pick an arbitrary in-band number. Every label sits on one of these:

| Type in draw.io | Prints at | Use for |
|-----------------|-----------|---------|
| `7`    | 5.04 pt | the floor; labels too cramped to grow further |
| `8.3`  | 5.98 pt | **the default** for figure text; matches the matplotlib panels |
| `9.7`  | 6.98 pt | the maximum for ordinary figure text |
| `11.1` | 7.99 pt | **panel letters only** (bold, upright, lowercase) |

A legal but off-ladder size is still wrong. Fig 1 carried labels at 5.76 pt
beside labels at 5.98 pt: a 0.2 pt difference no reader can see and nobody chose.

**Why tenths and not whole printed points.** A whole point needs a repeating
decimal on the canvas, because `0.72 = 18/25` and only point values that are
multiples of 18 come out whole, so exactness is not available at any sane
precision. draw.io keeps two decimals (typing `6.944` stores `6.94`), and tenths
already land within 0.03 pt of target, far below what prints differently. 5 pt
rounds UP to `7`: the exact value is 6.944 and `6.9` prints at 4.97 pt, still
under the floor.

The full conversion, for reference:

| Want on the page | Type in draw.io | Nature |
|------------------|-----------------|--------|
| 5 pt   | 7    | minimum allowed (6.9 falls just under) |
| 5.5 pt | 7.6  | allowed, off-ladder |
| 6 pt   | 8.3  | allowed, matches the matplotlib panels |
| 6.5 pt | 9.0  | allowed, off-ladder |
| 7 pt   | 9.7  | maximum allowed |
| 8 pt   | 11.1 | panel letters only (bold, upright, lowercase) |
| 9 pt   | 12.5 | over the maximum |
| 10 pt  | 13.9 | over |
| 11 pt  | 15.3 | over |
| 12 pt  | 16.7 | over |

And in reverse, for a canvas you have already drawn:

| On the canvas | Prints at | Verdict |
|---------------|-----------|---------|
| 6    | 4.32 pt | under the 5 pt minimum |
| 7    | 5.04 pt | on the ladder, the floor |
| 8    | 5.76 pt | in band but OFF the ladder; snap to 8.3 |
| 8.3  | 5.98 pt | on the ladder, the default |
| 9.7  | 6.98 pt | on the ladder, the maximum |
| 10   | 7.20 pt | 0.2 pt over; use 9.7 for exactly 7 |
| 11.1 | 8.00 pt | correct for panel letters, too big for anything else |
| 12   | 8.65 pt | over |

Matplotlib panels need no conversion: `fontsize=6` is 6 real points, and
`savefig_true_size_svg` plus the true-size `make plots` conversion preserve it.
Do not apply the 0.72 factor to those.

Figure float *placement* on the page is automatic (LaTeX floats, `[t]/[b]/[p]`);
only panel layout inside the figure is set in draw.io.

## Naming convention

Exported PDFs **keep their draw.io source's name**: `figures/NAME.pdf` is built from
`../../notes/assets/drawio/NAME.drawio.png` via one Makefile pattern rule
(`figures/%.pdf: $(DRAWIO_SRC)/%.drawio.png`). So source and export are trivially matched
(easy to find, edit, keep in sync). Add a figure by appending `figures/NAME.pdf` to the
`figures:` target in the Makefile -- no new rule needed.

## Expected files (placeholders live in content.tex until these exist)

- `TorchCell-Supervised-Learning-and-Teacher-Forcing-Generic-Phenotypes.pdf` -- TorchCell overview (R1)
- `fig2_cgt_architecture.pdf` -- CGT architecture (R2)
- `fig3_ggi.pdf` -- trigenic GGI state of the art (R3)
- `fig4_multitask.pdf` -- multitask generalization (R4)
- `fig5_design.pdf` -- strain design + DBTL (R5)
- `figS1_classical_ml.pdf` ... `figS6_inference.pdf` -- Supplementary

Keep panel text sans-serif, 5-7 pt at final size; line weights >= 0.25 pt.

## Auditing a figure

`paper/nature-biotech/scripts/drawio_font_band.py` reports every size in a
draw.io source, converted to points, and flags anything off the ladder:

```bash
python paper/nature-biotech/scripts/drawio_font_band.py --check notes/assets/drawio/NAME.drawio.svg
python paper/nature-biotech/scripts/drawio_font_band.py --fix   notes/assets/drawio/NAME.drawio.svg
```

`--check` exits non-zero when anything is out of band. Two things it handles that
a hand audit misses:

- **A size hides in two places.** An mxCell's style has `fontSize=N`, but a cell
  whose value is HTML can also carry an inline `font-size: Npx`, and the inline
  rule WINS. Fig 7's node numbers say `fontSize=12` (8.64 pt) and
  `font-size: 6px` (4.32 pt); 4.32 pt is what prints.
- **Growing a label can overflow its box silently.** draw.io does not reflow a
  shape when its text grows, and nothing errors. `--fix` widens a grown label in
  proportion, but only where the cell has no stroke and no fill and a wider box
  is therefore invisible.

**Re-render and look after any size change.** The audit passing is not evidence
the figure is intact.
