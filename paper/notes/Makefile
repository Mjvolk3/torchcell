# paper/notes/Makefile -- build the lab-note PDFs with Tectonic.
#
#   make            -> build every note in this directory
#   make <name>.pdf -> build one note
#   make figs       -> convert only the SVGs the notes reference to PDF
#   make clean      -> remove build/ and the generated PDFs
#
# These are NOTES wearing the manuscript's editing look; see notes-preamble.tex for how
# that is inherited (sn-jnl.cls, sn-nature.bst, references.bib are symlinks to
# ../nature-biotech/, and preamble.tex is \input from there).
#
# FIGURES. Plotting scripts write BOTH .png and .svg into notes/assets/images/ per the repo
# figure standard, and `assets` here is a symlink to that tree -- so a note references the
# SAME file the Dendron note and draw.io use, with no copy to drift. LaTeX cannot read SVG,
# so every \incsvg{PATH} is converted to build/figs/PATH.pdf by rsvg-convert (vector in,
# vector out -- no rasterizing). PNGs (plate photos, detection overlays) are included
# directly with \includegraphics{assets/...}; they are rasters with nothing to preserve.
#
# The \incsvg targets are discovered by grepping the sources, so adding a figure needs no
# edit here.

TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo /tmp/tectonic)
RSVG     ?= rsvg-convert
ASSETS   := assets

# Every note in this directory (top-level .tex files; sections/ and the preamble are inputs).
NOTES := $(filter-out notes-preamble.tex,$(wildcard *.tex))
PDFS  := $(NOTES:.tex=.pdf)

# Sources depended on for rebuilds (everything), versus sources SCANNED for \incsvg.
# notes-preamble.tex is deliberately excluded from the scan: it DEFINES \incsvg and shows a
# usage example in a comment, both of which would otherwise be picked up as figure targets.
SRC  := $(NOTES) notes-preamble.tex $(shell find sections -name '*.tex' 2>/dev/null)
SCAN := $(NOTES) $(shell find sections -name '*.tex' 2>/dev/null)

# SVG figures referenced as \incsvg{PATH} -> build/figs/PATH.pdf. The path is relative to
# assets/ and carries no extension, so PATH may contain slashes; mkdir -p in the rule
# handles the nesting.
SVG_REFS := $(shell grep -rhoE '\\incsvg(\[[^]]*\])?\{[^}]*\}' $(SCAN) 2>/dev/null \
              | sed -E 's/.*\{(.*)\}/\1/' | sort -u)
SVG_PDFS := $(addprefix build/figs/,$(addsuffix .pdf,$(SVG_REFS)))

.PHONY: all figs clean
all: $(PDFS)

figs: $(SVG_PDFS)

# Vector SVG -> vector PDF. A missing source is a hard error, not a skipped figure: a note
# that silently drops a figure is worse than one that fails to build.
build/figs/%.pdf: $(ASSETS)/%.svg
	@mkdir -p $(dir $@)
	$(RSVG) -f pdf -o $@ $<

# Each note depends on every source (they share the preamble and section files) and on the
# converted figures, so editing any section or adding any \incsvg rebuilds correctly.
%.pdf: %.tex $(SRC) $(SVG_PDFS)
	$(TECTONIC) -X compile $<

clean:
	rm -rf build $(PDFS)
