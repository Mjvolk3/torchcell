---
id: k1o94gkud7dmo7489p7937c
title: Style Guide
desc: ''
updated: 1784590893960
created: 1784590893960
---

Canonical standards for the Nature Biotechnology manuscript (`paper/nature-biotech/`):
prose, citations, figures, tables, proofs, provenance. **Read this before any paper
writing, figure creation, or table creation, and record every new preference or standard
here** under the relevant section. `CLAUDE.md` points here so agents load it. This is a
living *topical* reference (not a date-logged work note): update the rule in place under
its heading rather than appending dated sections. Where a rule is enforced repo-wide (the
palette, figure standards) this note gives the canonical summary and points to the
authoritative source (code + `CLAUDE.md`) rather than re-listing values that would drift.

Related: [[paper.proof-writing-standard]], [[paper.nature-biotech.figures]],
[[torchcell.utils.utils]], [[paper.nature-biotech.check-figures]].

## Prose & typography

- **No em-dashes, ever.** Never use `---` (the "—" glyph) in prose. Use a spaced en-dash
  ` -- ` or a comma. This is a general standing preference across all of the author's writing.
  - **Keep** range/name en-dashes `--`: `$10^{4\text{--}6}$`, `gene--gene`,
    `Kullback--Leibler`, `(a)--(d)`. Those are correct; only the 3-hyphen `---` is the target.
  - **Do not touch** editorial comment dividers (`%% ---- ... ----`) or markdown table
    separators (`|---|`) -- structure/syntax, not prose.
  - For auto-generated tables, fix the em-dash in the **generator script's** caption string
    and regenerate; never hand-edit the `.tex`.
  - Sweep used before: a Python pass replacing `*--- *` -> ` -- ` on non-comment lines of
    `sections/*.tex` + `content.tex`.

## Nature Supplementary citations

- Spell out with the "Supplementary" prefix: **Supplementary Fig. N**, **Supplementary
  Table N**, **Supplementary Note N**. Only "Fig." abbreviates; "Table"/"Note" do not.
- **"Fig. S1" / "Table S1" is WRONG** (no "S" prefix; SI floats are numbered from 1).
- Use the macros so the correct form is the only convenient one: `\suppfig{...}`,
  `\supptab{...}`, `\suppnoteref{...}` (defined in `preamble.tex`).

## Color palette

- **Canonical source is code + the swatch SVG, not this note:**
  `torchcell.utils.PLOT_PALETTE` / `PLOT_PALETTE_FILL` / `PLOT_PALETTE_NAMES` and
  `notes/assets/images/color-palette.svg` (from `notes/assets/scripts/generate_color_palette.py`).
  Full rules in `CLAUDE.md` "Figure & Plotting Standards".
- **18 colors** = orange · red · purple · yellow · blue · gray, repeated 3 times; a series of
  N takes the first N (primaries spent before blue/gray). Green-free.
- Tiers differ by **chroma, not lightness** (lightness encodes validation-vs-test within a
  series). 18 is the ceiling; cap new colors at `C* <= 36`; for >18 series disambiguate with
  hatching, not more color.
- **Use the LINE/border colors** (`PLOT_PALETTE`) for plot marks; the pale `PLOT_PALETTE_FILL`
  is only the lighter member of a two-level bar (validation = line color, test = fill).
  Hatches/edges solid black. Draw.io Fig 1 primaries (1--6) are **LOCKED**.

## Figures

- **Panel width is STRICT; height is loose.** Use `torchcell.utils.PANEL_WIDTHS_MM`
  (full 179 / wide 118.9 / half 88 / third 57.8 / sixth 28.3 mm) + `mm_to_in`; height
  `<= MAX_HEIGHT_MM` (170). Box all four spines (~0.5 pt). Arial 6 pt, `svg.fonttype: none`.
- Tenth gridlines on 0--1 axes (`MultipleLocator(0.2)` major + `0.1` minor, minor ticks hidden).
- Export true-size via `torchcell.utils.savefig_true_size_svg` (rescales 72 -> 100 dpi for
  draw.io); do NOT pass `bbox_inches="tight"` on a fixed-width panel.
- Compose in draw.io at Nature print size (180 mm full / 88 mm column, `<= 170` mm tall);
  export vector PDF into `paper/nature-biotech/figures/` (auto via `make fig`/`make paper`).
  The size gate allows `+2` mm grace and requires `\tcfig` (never `\tcfigfit`). See
  `paper/nature-biotech/figures/README.md` and [[paper.nature-biotech.figures]].
- **Nature's official figure specs** (verified 2026.08.17 from
  `https://www.nature.com/documents/nature-final-artwork.pdf` "Guide to preparing final
  artwork" and `https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/`):
  standard widths **89 mm (single column)** and **183 mm (double column)**; 1.5-column
  figures **120 or 136 mm**; full page depth **247 mm**; "The maximum height for a
  *Nature* figure is 170 mm, to allow space for the figure legend to fit underneath."
  Text: sans-serif (Helvetica/Arial), max 7 pt, min 5 pt; panel letters "8-pt bold,
  upright (not italic) and lowercase a, b, c"; do not outline text. **Our stance:
  WIDTH is enforced strictly** (panels on the `PANEL_WIDTHS_MM` grid); height warnings
  are advisory for now and resolved before submission.
- **draw.io font numbers are NOT points. Multiply by 0.72.** draw.io's canvas is 100
  units per inch and its font-size field is in those canvas units, the same unit as
  every coordinate; a point is 1/72 inch. So `72/100 = 0.72` converts, and a label
  typed as `8` prints at **5.76 pt**. Measured, not inferred: `Fig1-torchcell-overview`
  is 707 units wide and its exported PDF page is 509 pt (0.7199 pt per unit), and
  `\tcfig` places it at natural size so nothing scales afterward. The general form is
  `rendered pt = drawio number x (placed width in pt) / (canvas width in units)`.

  To hit a target, divide by 0.72 (equivalently, multiply by 1.3889):

  | Want on the page | Type in draw.io | Nature |
  |---|---|---|
  | 5 pt | 6.9 | minimum allowed |
  | 5.5 pt | 7.6 | allowed |
  | 6 pt | 8.3 | allowed, matches our matplotlib panels |
  | 6.5 pt | 9.0 | allowed |
  | 7 pt | 9.7 | maximum allowed |
  | 8 pt | 11.1 | **panel letters only** (bold, upright, lowercase) |
  | 9 pt | 12.5 | over the maximum |
  | 10 pt | 13.9 | over |
  | 11 pt | 15.3 | over |
  | 12 pt | 16.7 | over |

  Read the other way, for numbers already on a canvas:

  | On the canvas | Prints at | Verdict |
  |---|---|---|
  | 6 | 4.32 pt | under the 5 pt minimum |
  | 8 | 5.76 pt | in band |
  | 10 | 7.20 pt | 0.2 pt over; use 9.7 for exactly 7 |
  | 11.1 | 8.00 pt | correct for panel letters, too big for anything else |
  | 12 | 8.65 pt | over |

  The matplotlib panels need no conversion: `fontsize=6` is 6 real points, and
  `savefig_true_size_svg` plus the true-size `make plots` conversion preserve it.
- **Composed SI figures: the "white cross" layout.** Every compose script (the
  `*_compose_figure.py` / `compose_*_si_figures.py` scripts that write a `FigS-*.drawio`)
  uses the same explicit constants: `COL_GAP = 12` units (3 mm) between columns,
  `ROW_GAP = 22` units (5.5 mm) between rows, a `TOP_STRIP = 16` unit strip above every
  row, and each panel letter placed in that strip at the panel's top-left
  (`x = panel_x, y = row_top`). A letter therefore never sits over a panel's y-axis label
  or a neighbor's title, and clear white gutters cross the figure both ways. Figures stay
  `<= 709 x 669` units (+8 grace).
- **Equations in draw.io figures are real LaTeX, typeset by MathJax.** Set `math="1"` on
  the `mxGraphModel` and write the label as `$$...$$`; the headless PDF export honors it
  (verified with draw.io 31.3.1, glyphs exported as vector paths). MathJax renders about
  1.19x the cell's `fontSize` (measured 2026.09.03 by cap height against Arial), so a math
  cell typed at `fontSize=7` prints at ~6 pt, matching the matplotlib panels, while
  `drawio_font_band.py` reads it as 5.04 pt (on the ladder). Do not use `8.3` for math
  cells; it prints at ~7.1 pt, over Nature's maximum. Worked example: panel a of
  `FigS-dcell-model` (`experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure.py`).
- **Bar charts with replicate points:** bar = mean, whisker = SD or SEM named in the
  caption, replicates as open circles; when a group has one run, say so instead of drawing
  a whisker.
- **Panel letters in captions are bold, no parentheses** (Nature form): `\textbf{a},~Text.
  \textbf{b},~Text.` and ranges `\textbf{a}--\textbf{e}`. Never `(a)` in a caption. Applied
  2026.09.05 across every figure caption (main and SI); cross-references in prose keep the
  plain suffix form (`\suppfig{fig:x}a`).
- **Axis-label style: sentence case, first word capitalized, proper nouns/initialisms
  keep their capitals.** Nature's spec gives the exemplar "All axes to be labelled with
  units in parentheses, e.g. Data (unit)" -- sentence case; unitless metrics omit the
  parenthetical. House forms: `Dataset size` (never `Dataset Size`), `Percent of
  dataset`, `Samples`, `Test Pearson`, `Pearson`, `Spearman`, `MSE` (never lowercase
  `pearson`/`spearman`/`mse`). Applied 2026.08.17 across the traditional-ML plot
  scripts (`{"mse": "MSE"}.get(metric, metric.capitalize())` for metric axes).
- **When two categories overlap heavily in one node-link diagram, repeat the diagram per
  category rather than encoding both in one drawing.** A small unlabeled copy at the
  SAME node positions, one per category, is read instantly; a combined drawing is not,
  and no amount of color, opacity or z-order fixes it. Established 2026.09.16 on the FFA
  network figure, where 45 pairs carry a negative interaction and 17 also carry a
  positive one. Two attempts failed first: drawing positive over negative hid every
  positive edge, and drawing each triple as a translucent filled triangle turned 75
  overlapping negatives into one lens-shaped blob with no structure in it. What worked
  was shrinking the labeled circle and adding the same ring twice beside it, positives
  above and negatives below, unlabeled and at identical angles. The shapes then carry
  the finding on their own: negatives reach all 45 pairs, positives reach 17 and miss
  three factors entirely. Keep the orientation identical across the copies, or the
  reader has to re-find each node before comparing; labels belong on the main copy only.
  **Follow-up (2026.09.16): on the small unlabeled copies the unit becomes the TRIPLE.**
  Each interaction is one translucent triangle at the same opacity in both rings, so ink is
  proportional to how many interactions land there and the two signs are read on one scale.
  Translucent triangles failed on the 23 mm labeled circle and work on an 8 mm ring beside
  its opposite sign, because there the contrast between the two IS the reading. An arrow
  beside each ring, in the ring's own color, gives its sign without a label.
- **Type size governs what a borrowed figure may carry.** A reference map, pathway diagram
  or screenshot brings its own labels at its own scale. Placed at panel width they usually
  print below the 5 pt floor: iPath3's global metabolic map at 100 mm puts its largest
  label near 2.5 pt. Strip the borrowed text and re-set what the reader needs at 5.98 pt in
  a key beside the panel. Measure before assuming a borrowed label survives the reduction.
- **A headless draw.io export's success cannot be read from its exit code or its output
  file.** `xvfb-run` returns 1 after a success, drawio-desktop returns 0 after writing
  nothing, and the previous export's file is still on disk, so a failed run silently
  reports the LAST good figure's size. Require the file to have been written during the
  call (an mtime stamp taken before the run). A ~300 kB base64 image in one style attribute
  is enough to make the export fail; assemble such a panel as SVG instead.

- **A panel letter placed by LaTeX is set in the DOCUMENT font, not the figure's.** `\textbf{a}`
  in a `\panel`-style macro prints Latin Modern Bold next to Arial panels, and it is easy to
  miss because the letter is one glyph. Tectonic runs XeTeX, so declare a fontspec family for
  the letter alone (`\newfontfamily\panelletterfont{Arial}[BoldFont={Arial Bold}]`) and leave
  the body font untouched. Audit with `pdffonts <doc>.pdf`: anything outside Arial and the
  document's own Latin Modern / CM families is a figure shipping a foreign typeface.
- **draw.io's HTML `<sub>` is not math typesetting.** Variables come out upright where they
  should be italic, subscripts sit at the wrong size, and the PDF export sets the subscript
  runs in a serif fallback. Render each expression as math (matplotlib mathtext with the
  Arial families, which is how every other panel sets `$\tau$`), record its measured size,
  and place it as an image. A serif subset containing only a space is a different thing and
  is harmless: draw.io emits it for whitespace inside an embedded image.
- **A borrowed reference map goes in whole when the point is "where does this sit".** A crop
  answers that only for a reader who already knows the map. Size the panel to the drawing's
  own aspect ratio, and put its labels on opaque plates rather than a halo on the glyphs:
  over a dense map a label routinely crosses three lines and a stroke halo is not enough.
- **Two panels that show "the same" entities must show the same set, and a borrowed map
  that lacks some of them gets those attached from the panel that has them.** Fig 5 had two
  measured species on the iPath3 map and five in the GEM network beside it; a reviewer read
  the mismatch as an error. Attach the missing ones from the same source the other panel
  uses, by a computed rule (the GEM's shortest path from a drawn compound, genes named),
  drawn in a style that cannot be mistaken for the map's own (open rings, dashed links),
  and say in the key what was attached and why. Never drop entities from the richer panel
  to match the poorer one.
- **A label may not sit on the route, cover a node, or cross another leader; place by
  search, not by a fixed side.** Over a dense map "all labels to the right, pushed apart"
  put a leader through a neighboring label and a plate on a neighboring node. Sample the
  drawing into an ink grid with the route weighted far above the faint background, try
  many directions and leader lengths per label, reject any candidate that covers a node or
  a drawn line or whose leader crosses a plate or another leader, and place a crowded
  group in the best of all orders rather than greedily. `experiments/008-xue-ffa/scripts/map_labels.py`
  is the worked implementation, and each rule in it was added after a render showed the failure.
- **Borrow a reference map's coordinates, not its rendering.** iPath3 returns a finished
  drawing that is older than KEGG and cannot be restyled below its own type; KEGG's KGML
  (`get/<org>01100/kgml`) gives every reaction polyline and compound circle with coordinates
  and identifiers, so the map can be redrawn as vector in the document's palette, at the
  document's line weights, with a genome-scale model's membership overlaid by id. A
  coherent whole-metabolism map cannot be laid out from a stoichiometric model; the layout
  has to be borrowed, and KGML is the form of it that can be re-rendered honestly.
- **A negative claim in a caption ("none of the deleted genes is on this map") is measured,
  not asserted.** Probe each gene's orthology group on the map and record the count in the
  provenance file; the caption and key are generated from those counts. The same probe
  found that the map lacks four of the thirteen pathway genes, a fact the caption now
  carries instead of the assumed "all thirteen".

## Tables

- **Every paper table comes from a committed script** in the relevant `experiments/<id>/`
  folder (STRICT RULE in `CLAUDE.md`); never hand-author numbers. Generated `.tex` carries a
  `%% SOURCE:` header and "AUTO-GENERATED -- do not hand-edit"; regenerate, don't edit.
  - Classical-ML tables: `experiments/smf-dmf-tmf-001/traditional_ml-summary_table.py --write-tables`.
  - Entity-corpora table: `.../persistent_entity_corpus_sizes.py --from-csv --write-table`
    (offline re-render from the frozen snapshot; never re-hammer the archives for a format fix).
- **Compressed sizes are reported in BITS, everywhere.** Both corpora in the information
  accounting are measured the same way -- a `gzip` byte count -- and both are reported as the
  codelength $L_C(D) = 8\lvert C(s(D))\rvert$ in bits, because bits is the unit the
  Proposition, Eqs. (24)/(25), Fig. 1c, and every ratio are stated in. This covers the
  persistent-entity table (`Bits`) and the supported-datasets table (`Signal (gzip, bits)`);
  the two columns are meant to be read against each other, so they must not differ in unit.
  Keep the x8 in the generating script, never in the `.tex`.

## Proofs & formal claims

- Follow [[paper.proof-writing-standard]]: Setup -> Claim -> Proof -> Consequence ->
  Interpretation; `proposition`/`lemma` + `proof` environments, `\pfstep{...}` step headers,
  no bullets inside a proof. Prefer **Proposition** for main claims; do NOT use **Theorem**
  (the paper is empirical).

## Provenance

- Any artifact used in the paper or `notes/` (figure, table, derived number) MUST be produced
  by a committed script that reads the real result files, and the artifact should point to its
  generating script.
