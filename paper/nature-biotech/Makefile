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

SHARED := content.tex preamble.tex references.bib sn-jnl.cls sn-nature.bst
VIEWS  := submission editing twocolumn

.PHONY: paper all submission editing twocolumn figproto clean
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

clean:
	rm -f submission.pdf editing.pdf twocolumn.pdf figure-proto.pdf
