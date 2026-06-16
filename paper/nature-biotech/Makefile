# paper/nature-biotech/Makefile -- build the manuscript PDFs with Tectonic.
#
#   make paper       -> the 3 journal/typeset views (submission, editing, twocolumn)
#   make submission  -> submission.pdf  (journal submission; official sn-nature, single column)
#   make editing     -> editing.pdf     (our drafting/typeset look; print-approx margins, char budgets)
#   make twocolumn   -> twocolumn.pdf   (published-like Nature double column)
#   make figproto    -> figure-proto.pdf (true-scale figure-sizing canvas)
#   make all         -> paper + figproto
#   make clean       -> remove generated PDFs
#
# Uses an installed `tectonic` if on PATH, else the /tmp binary we fetched.
# Install permanently with:  conda install -c conda-forge tectonic

TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo /tmp/tectonic)

# draw.io headless export (Electron under Xvfb). Point DRAWIO at the drawio-desktop
# AppImage/binary (must be executable). APPIMAGE_EXTRACT_AND_RUN avoids needing FUSE.
DRAWIO     ?= /tmp/drawio.AppImage
DRAWIO_SRC := ../../notes/assets/drawio
DRAWIO_EXPORT = APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a $(DRAWIO) --no-sandbox --disable-gpu -x -f pdf --crop -o

SHARED := content.tex preamble.tex references.bib sn-jnl.cls sn-nature.bst $(wildcard sections/*.tex)
VIEWS  := submission editing twocolumn

.PHONY: paper all submission editing twocolumn figproto figures flat clean
paper: $(addsuffix .pdf,$(VIEWS))
all: paper figproto

submission: submission.pdf
editing: editing.pdf
twocolumn: twocolumn.pdf
figproto: figure-proto.pdf

# Each view wrapper \input's the shared body, so rebuild when either changes.
submission.pdf editing.pdf twocolumn.pdf: %.pdf: %.tex $(SHARED)
	$(TECTONIC) -X compile $<

figure-proto.pdf: figure-proto.tex preamble.tex
	$(TECTONIC) -X compile $<

# Export figures from draw.io sources to TRUE-TO-SIZE vector PDFs (headless via Xvfb).
# Include the result with \includegraphics{...} (no width/height) to keep WYSIWYG.
# Add one rule per figure as you create the draw.io sources.
figures: figures/fig1_overview.pdf
figures/fig1_overview.pdf: $(DRAWIO_SRC)/TorchCell-Supervised-Learning-and-Teacher-Forcing-Generic-Phenotypes.drawio.png
	$(DRAWIO_EXPORT) $@ $<

# Flatten to a single .tex for Springer Nature submission (no \input; .bbl inlined
# if a submission.bbl is present). Springer requires one file -- do this at submit time.
flat: submission.tex $(SHARED)
	python3 flatten_tex.py submission.tex submission-flat.tex

clean:
	rm -f submission.pdf editing.pdf twocolumn.pdf figure-proto.pdf submission-flat.tex
