# `paper/notes` — lab notes in LaTeX

A down-selected, typeset mirror of the Dendron notes. Not the manuscript, but deliberately
wearing the manuscript's **editing** look, so a figure drafted in a note appears at the same
size and type it will have in the paper.

A note graduates to LaTeX when it has outgrown a markdown render — when it needs a table of
contents, real cross-references, sectioning deeper than markdown reads well, or figures placed
at true print size. The Dendron note then becomes a stub that links here, so there is exactly
one copy of the prose.

## Build

```bash
make -C paper/notes                 # every note
make -C paper/notes <name>.pdf      # one note
make -C paper/notes figs            # just the SVG -> PDF conversions
make -C paper/notes clean
```

Needs `tectonic` (`conda install -c conda-forge tectonic`) and `rsvg-convert`
(`librsvg2-tools`) — the same two tools the manuscript and the markdown → PDF script already
use.

## Naming

A note's `.tex` and its `.pdf` keep the **Dendron fname** of the note they came from, e.g.

```
experiments.010-kuzmin-tmi.scripts.construction_validation_doubles.tex
```

so a file is trivially matched to its Dendron stub, its generating script, and its results
directory. Sections live in `sections/<short-slug>/NN-topic.tex`.

## How the manuscript's look is inherited

Nothing about the style is re-declared here — it is symlinked or `\input` from
`../nature-biotech/`, so a change there reaches the notes with no second edit:

| here | is | from |
|---|---|---|
| `sn-jnl.cls` | symlink | `../nature-biotech/sn-jnl.cls` |
| `sn-nature.bst` | symlink | `../nature-biotech/sn-nature.bst` |
| `references.bib` | symlink | `../nature-biotech/references.bib` |
| `assets` | symlink | `../../notes/assets/images` |
| page geometry, block paragraphs | copied from | `../nature-biotech/editing.tex` |
| packages, `\tcfig`, proof style | `\input` | `../nature-biotech/preamble.tex` |

`notes-preamble.tex` adds only what a note needs and a journal submission does not: a table of
contents, section numbering to three levels, and coloured hyperlinks. It also no-ops the
manuscript's editing-only macros (`\cb`, `\wc`, `\secstatus`) so a paragraph lifted from the
paper compiles here unchanged.

## Figures

Plotting scripts write **both** `.png` and `.svg` into `notes/assets/images/` per the repo
figure standard. `assets` is a symlink to that tree, so a note references the *same file* the
Dendron note and draw.io use — no copy, nothing to drift.

LaTeX cannot read SVG, so use `\incsvg` for vector figures:

```latex
\incsvg{010-kuzmin-tmi/construction_validation_doubles_table}
\incsvg[width=0.7\linewidth]{W019-echo-crispr-array/run3/run3_se_batch_effect}
```

The path is relative to `assets/`, **without** the extension. The Makefile greps the sources
for `\incsvg{...}`, converts each target with `rsvg-convert` into `build/figs/`, and the macro
includes that PDF — vector in, vector out. Adding a figure needs no Makefile edit. A missing
SVG is a hard build error, not a silently dropped figure.

Rasters (plate photographs, Cellpose detection overlays) are PNG and go in directly:

```latex
\includegraphics[width=\linewidth]{assets/W019-echo-crispr-array/run3/run3_overlay_P1.png}
```

Wrap either in `\notefig{<graphic>}{<caption>}` to get a captioned float.

## Known benign build warning

`Invalid UTF-8 byte or sequence at line 11 replaced by U+FFFD.` comes from `algorithm.sty` in
Tectonic's own package bundle, pulled in by `../nature-biotech/preamble.tex`. The manuscript
build emits it too. Not ours, not harmful.
