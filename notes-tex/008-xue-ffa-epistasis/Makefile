# notes-tex/008-xue-ffa-epistasis/Makefile
# [[experiments.008-xue-ffa.perspective-epistasis-in-metabolic-engineering]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/008-xue-ffa-epistasis/Makefile
#
#   make          -> editing.pdf, the drafting view (status chips, print-approximate margins)
#   make plots    -> convert the true-size panel SVGs into figures/*.pdf
#   make figures  -> re-export the draw.io sources this document references
#   make clean    -> remove build artifacts
#
# This document does NOT include ../common/Makefile.common. That file is written for the
# tcdoc article style: it symlinks tcdoc.sty, builds `main.pdf`, and runs a checker that
# expects tcdoc's conventions. This document is a Nature Biotechnology draft in the sn-jnl
# class, so it needs sn-jnl.cls and the sn-nature bibliography style symlinked instead, and
# it builds `editing.pdf` to match the manuscript's naming.

DOC      := editing
TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo /tmp/tectonic)
PYTHON   ?= python
REPO     := ../..
COMMON   := ../common
PAPER    := $(REPO)/paper/nature-biotech

# Tectonic does not honor TEXINPUTS, so the class and bibliography style are symlinked in
# at build time rather than copied. Symlinks so there is exactly one copy of each in the
# repo and it cannot drift; both are gitignored and any build recreates them.
SHARED := sn-jnl.cls sn-nature.bst

SECTIONS := $(wildcard sections/*.tex)
DEPS     := $(DOC).tex content.tex preamble.tex $(SECTIONS)

# Where `make plots` looks for the matplotlib SVGs. The document directory is named for the
# perspective, the images directory for the experiment, so unlike every other notes-tex
# document these two differ and the slug has to be written out.
PLOT_SLUG := 008-xue-ffa
PLOT_SRC  := $(REPO)/notes/assets/images/$(PLOT_SLUG)
DRAWIO_SRC := $(REPO)/notes/assets/drawio

ifeq ($(shell uname -s),Darwin)
  DRAWIO        ?= /Applications/draw.io.app/Contents/MacOS/draw.io
  DRAWIO_EXPORT  = "$(DRAWIO)" -x -f pdf --crop -o
else
  DRAWIO        ?= /tmp/drawio.AppImage
  DRAWIO_EXPORT  = APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a $(DRAWIO) --no-sandbox --disable-gpu -x -f pdf --crop -o
endif

.PHONY: all plots figures clean

all: $(DOC).pdf

sn-jnl.cls: $(PAPER)/sn-jnl.cls
	@ln -sf ../../paper/nature-biotech/sn-jnl.cls $@
sn-nature.bst: $(PAPER)/sn-nature.bst
	@ln -sf ../../paper/nature-biotech/sn-nature.bst $@

$(DOC).pdf: $(DEPS) $(SHARED)
	$(TECTONIC) -X compile $(DOC).tex --keep-logs --synctex --outdir .

# Conversion goes through svg_true_size_pdf.py, NOT bare rsvg-convert. rsvg misreads both
# SVG forms this repo writes and silently oversizes a panel by 1.389 (unitless) or 1.25
# (millimetres), which puts 6 pt panel type near 8 pt and breaks the panel widths chosen to
# tile a 180 mm page. The script derives the correct zoom from each file's own header.
SVG2PDF := $(PYTHON) $(COMMON)/svg_true_size_pdf.py
plots: $(SHARED)
	@mkdir -p figures
	@for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	  n=$$(basename $$f .pdf); \
	  s="$(PLOT_SRC)/$$n.svg"; \
	  if [ -f "$$s" ]; then $(SVG2PDF) "$$s" "$$f"; fi; \
	done

# Re-export any draw.io source this document references as figures/NAME.pdf. Use this once
# a figure's arrangement is settled in draw.io and its \panelrow block has been replaced by
# a single \tcfig on the exported PDF.
figures:
	@for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	  n=$$(basename $$f .pdf); \
	  for ext in drawio.svg drawio.png drawio; do \
	    s="$(DRAWIO_SRC)/$$n.$$ext"; \
	    if [ -f "$$s" ]; then echo "  $$s -> $$f"; $(DRAWIO_EXPORT) "$$f" "$$s"; break; fi; \
	  done; \
	done

clean:
	rm -f $(SHARED) $(DOC).pdf $(DOC).log $(DOC).aux $(DOC).out $(DOC).toc \
	      $(DOC).bbl $(DOC).blg $(DOC).synctex.gz
