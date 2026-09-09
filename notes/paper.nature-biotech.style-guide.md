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
- **Panel letters top-left, outside the axes box** (`torchcell.utils.panel_label`), never
  inside the plotting area. The letter's LEFT edge is flush with the panel's outer left
  edge (the y-axis label column, measured from the axes' tight bounding box, so call
  `panel_label` after the y label and ticks are set) and it is raised
  `PANEL_LABEL_RAISE_PT` (12 pt) above the axes top edge, one text line above the title
  band. A cross the width of the letter laid over it then meets no title, spine, tick
  label, or axis label. The old placement (1.5 pt off the axes corner) fails: the
  downward arm crosses the topmost tick label and the rightward arm crosses any title
  that starts near the left edge.
- **The white-cross rule for text: no line may touch a label.** Every piece of text in a
  panel (panel letter, legend entry, annotation, in-panel caption) must sit on clear
  background: lay a white cross the width of the text box over it and no data line, curve,
  spine, or bar may intersect it. A legend that a curve passes under, an annotation that a
  fitted line runs through, or region text that touches a spine all fail. Move the text,
  widen an axis limit, or shorten the label. Broken only in extraordinary circumstances,
  and then said in the caption.
- **Legends are framed.** White face, 0.5 pt black edge, square corners
  (`legend.frameon: True`, `legend.fancybox: False`, `legend.framealpha: 1`). The frame
  still sits in a clear region; it is a border, not a license to cover data.
- **No mathtext accents in SVG panels.** With `svg.fonttype: none`, `\hat{y}` and
  `\hat F` render as a detached dotted glyph in every SVG viewer (VS Code, rsvg). Write
  the precomposed character (`\u0177` for y-hat) or drop the hat and name the object in
  words (`predictive CDF F`). Plain `\tau`, `\alpha`, `\rho`, `\log_2` are fine.
- **Check the SVG render, not the PNG.** The PNG is rasterized by matplotlib with the
  text baked in; the SVG is what the note shows and what goes into draw.io, and its text
  is laid out by the viewer. Verify with `rsvg-convert -w 2400 -o out.png panel.svg`
  and look at that.
- **Colored text sits on white, not on a fill.** Red text on a purple band, or any text
  over a shaded region, is unreadable at 6 pt. Shaded regions carry no text; label them
  from a legend entry or from text placed in an unshaded part of the panel.
- **Aligned columns for compared numbers.** When a legend compares the same statistic
  across series (coverage per arm, a score per model), lay the numbers out as aligned
  columns (a small `ax.text` table in axes coordinates), not as prose inside legend labels,
  where proportional Arial defeats alignment.
- **Axis-label style: sentence case, first word capitalized, proper nouns/initialisms
  keep their capitals.** Nature's spec gives the exemplar "All axes to be labelled with
  units in parentheses, e.g. Data (unit)" -- sentence case; unitless metrics omit the
  parenthetical. House forms: `Dataset size` (never `Dataset Size`), `Percent of
  dataset`, `Samples`, `Test Pearson`, `Pearson`, `Spearman`, `MSE` (never lowercase
  `pearson`/`spearman`/`mse`). Applied 2026.08.17 across the traditional-ML plot
  scripts (`{"mse": "MSE"}.get(metric, metric.capitalize())` for metric axes).

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
