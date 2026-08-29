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
