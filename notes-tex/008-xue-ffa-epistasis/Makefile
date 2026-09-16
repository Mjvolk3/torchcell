# notes-tex/008-xue-ffa-epistasis/Makefile
# [[experiments.008-xue-ffa.perspective-epistasis-in-metabolic-engineering]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/008-xue-ffa-epistasis/Makefile
#
#   make           -> editing.pdf, the drafting view (status chips, print-approximate margins)
#   make plots     -> convert the true-size panel SVGs into figures/*.pdf
#   make figures   -> re-export the draw.io sources this document references
#   make bib-store -> rebuild references.bib from the served library, by cited key
#   make clean     -> remove build artifacts
#
# This document does NOT include ../common/Makefile.common. That file is written for the
# tcdoc article style: it symlinks tcdoc.sty, builds `main.pdf`, and runs a checker that
# expects tcdoc's conventions. This document is a Nature Biotechnology draft in the sn-jnl
# class, so it needs sn-jnl.cls and the sn-nature bibliography style symlinked instead, and
# it builds `editing.pdf` to match the manuscript's naming.

DOC      := editing
TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo /tmp/tectonic)
PYTHON   ?= python
BIBSTORE ?= library
REPO     := ../..
COMMON   := ../common
PAPER    := $(REPO)/paper/nature-biotech

# Tectonic does not honor TEXINPUTS, so the class and bibliography style are symlinked in
# at build time rather than copied. Symlinks so there is exactly one copy of each in the
# repo and it cannot drift; both are gitignored and any build recreates them.
SHARED := sn-jnl.cls sn-nature.bst

SECTIONS := $(wildcard sections/*.tex)
DEPS     := $(DOC).tex content.tex preamble.tex $(SECTIONS)
# The figure PDFs are a build dependency, not just an input. Without them here, a run of
# `make plots && make` after regenerating a panel leaves editing.pdf untouched, because
# none of the .tex files changed: the document then SHIPS the previous render of the panel
# and nothing says so. FIGS is expanded at parse time, so a brand new figure still needs
# one `make` to be noticed.
FIGS     := $(wildcard figures/*.pdf)
BIB      := references.bib

# Where `make plots` looks for the matplotlib SVGs. The document directory is named for the
# perspective, the images directory for the experiment, so unlike every other notes-tex
# document these two differ and the slug has to be written out.
PLOT_SLUG := 008-xue-ffa
# The growth-model figure (ffa_panel_model_score) is written by the 010 experiment's
# script into its own image directory; `make plots` searches both, first match wins.
PLOT_SLUGS := 008-xue-ffa 010-kuzmin-tmi
PLOT_SRCS := $(foreach slug,$(PLOT_SLUGS),$(REPO)/notes/assets/images/$(slug))
DRAWIO_SRC := $(REPO)/notes/assets/drawio

ifeq ($(shell uname -s),Darwin)
  DRAWIO ?= /Applications/draw.io.app/Contents/MacOS/draw.io
  drawio_export = "$(DRAWIO)" -x -f pdf --crop -o "$(2)" "$(1)"
else
  # Linux: the AppImage is extracted once into DRAWIO_ROOT and its inner binary is called
  # with the INPUT PATH FIRST. drawio-desktop 24.7 rejects an input placed after the
  # Electron flags ("input file/directory not found"), and APPIMAGE_EXTRACT_AND_RUN prints
  # the extraction listing without running the export.
  DRAWIO_APPIMAGE ?= /tmp/drawio.AppImage
  DRAWIO_ROOT     ?= /tmp/drawio-squashfs
  DRAWIO           = $(DRAWIO_ROOT)/squashfs-root/drawio
  drawio_export = xvfb-run -a "$(DRAWIO)" "$(1)" --no-sandbox --disable-gpu -x -f pdf --crop -o "$(2)"
endif

.PHONY: all plots figures drawio-bin bib-store bib-check clean

all: $(DOC).pdf

# This document has no Zotero collection of its own, so neither `make bib` (Better BibTeX
# export of a collection pair) nor `make bib-pull` (a bibliography tc-lit serves by the
# document's name) applies. bib_from_store.py takes the served WHOLE-LIBRARY export,
# verifies it against the store manifest's sha256, and writes only the keys this document
# cites. Curation still happens in Zotero: a key that is not in the library is an error
# here, never a stub, and adding it there is a separate human decision.
bib-store:
	@$(PYTHON) $(COMMON)/bib_from_store.py --name $(BIBSTORE) --out $(BIB)

bib-check:
	@$(PYTHON) $(COMMON)/bib_from_store.py --name $(BIBSTORE) --out $(BIB) --check

sn-jnl.cls: $(PAPER)/sn-jnl.cls
	@ln -sf ../../paper/nature-biotech/sn-jnl.cls $@
sn-nature.bst: $(PAPER)/sn-nature.bst
	@ln -sf ../../paper/nature-biotech/sn-nature.bst $@

$(DOC).pdf: $(DEPS) $(FIGS) $(SHARED) $(BIB)
	$(TECTONIC) -X compile $(DOC).tex --keep-logs --synctex --outdir .

# Conversion goes through svg_true_size_pdf.py, NOT bare rsvg-convert. rsvg misreads both
# SVG forms this repo writes and silently oversizes a panel by 1.389 (unitless) or 1.25
# (millimetres), which puts 6 pt panel type near 8 pt and breaks the panel widths chosen to
# tile a 180 mm page. The script derives the correct zoom from each file's own header.
#
# A figure with a draw.io source is SKIPPED here and belongs to `make figures`. Both rules
# match on the same stem, and the draw.io export writes its own true-size SVG next to the
# matplotlib panels, so without the skip `make plots` would convert that SVG with
# rsvg-convert, which cannot render the foreignObject text draw.io emits for HTML labels:
# the figure then compiles to a PDF reading "Text is not SVG - cannot display" where its
# labels should be. draw.io's own PDF export has no such problem.
SVG2PDF := $(PYTHON) $(COMMON)/svg_true_size_pdf.py
plots: $(SHARED)
	@mkdir -p figures
	@for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	  n=$$(basename $$f .pdf); \
	  if [ -f "$(DRAWIO_SRC)/$$n.drawio" ] || [ -f "$(DRAWIO_SRC)/$$n.drawio.svg" ]; then \
	    echo "  skip $$n (draw.io source; use \`make figures\`)"; continue; \
	  fi; \
	  for d in $(PLOT_SRCS); do \
	    s="$$d/$$n.svg"; \
	    if [ -f "$$s" ]; then $(SVG2PDF) "$$s" "$$f"; break; fi; \
	  done; \
	done

# Re-export any draw.io source this document references as figures/NAME.pdf. Use this once
# a figure's arrangement is settled in draw.io and its \panelrow block has been replaced by
# a single \tcfig on the exported PDF.
drawio-bin:
ifneq ($(shell uname -s),Darwin)
	@test -x "$(DRAWIO)" || (mkdir -p "$(DRAWIO_ROOT)" && cd "$(DRAWIO_ROOT)" && "$(DRAWIO_APPIMAGE)" --appimage-extract >/dev/null)
endif

figures: drawio-bin
	@for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	  n=$$(basename $$f .pdf); \
	  for ext in drawio.svg drawio.png drawio; do \
	    s="$(DRAWIO_SRC)/$$n.$$ext"; \
	    if [ -f "$$s" ]; then echo "  $$s -> $$f"; $(call drawio_export,$$s,$$f) >/dev/null; break; fi; \
	  done; \
	done

clean:
	rm -f $(SHARED) $(DOC).pdf $(DOC).log $(DOC).aux $(DOC).out $(DOC).toc \
	      $(DOC).bbl $(DOC).blg $(DOC).synctex.gz
