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

.PHONY: paper all submission editing twocolumn figproto figlimits figures fig checkfigs flat clean
paper: $(addsuffix .pdf,$(VIEWS))
all: paper figproto figlimits

submission: submission.pdf
editing: editing.pdf
twocolumn: twocolumn.pdf
figproto: figure-proto.pdf
figlimits: figure-limits.pdf

# Each view wrapper \input's the shared body, so rebuild when either changes.
# HARD GATE: every manuscript view depends on checkfigs, so ALL builds (submission,
# editing, twocolumn -- and `make paper`, which includes them) refuse to compile while
# any figure is out of spec, keeping figure sizing consistent across every view.
# checkfigs is .PHONY, so it runs on every build and aborts before Tectonic if a figure
# is over 180x170 mm or scaled. Read the `make checkfigs` report to see which figure is
# over and by how much (figproto is a sizing tool, not a view, so it stays ungated).
submission.pdf editing.pdf twocolumn.pdf: %.pdf: %.tex $(SHARED) checkfigs
	$(TECTONIC) -X compile $<

figure-proto.pdf: figure-proto.tex preamble.tex
	$(TECTONIC) -X compile $<

# Figure-sizing reference card for collaborators (Nature print-box limits: 180x170 full,
# 88 mm column, 5-7 pt text, 8 pt bold labels). Source is a plain .drawio (not .drawio.png).
# Output lives in the paper root (like figure-proto.pdf) so the figure gate (checkfigs globs
# figures/*.pdf) does NOT apply to it; sync-overleaf.sh publishes it to the Overleaf figures/.
figure-limits.pdf: $(DRAWIO_SRC)/figure-limits.drawio
	$(DRAWIO_EXPORT) $@ $<

# Export figures from draw.io sources to vector PDFs (headless via Xvfb).
# CONVENTION: the exported PDF keeps the draw.io source's name --
#   figures/NAME.pdf  <-  $(DRAWIO_SRC)/NAME.drawio.png
# so source and export are trivially matched (find, edit, keep in sync). One pattern
# rule covers every figure; to add one, just append its NAME.pdf to the list below.
figures: figures/TorchCell-Supervised-Learning-and-Teacher-Forcing-Generic-Phenotypes.pdf \
         figures/ontology_pydantic_hourglass_data_model.pdf \
         figures/neo4j_cell-conversion-deduplication-aggregation.pdf \
         figures/s288c_selecting_gene_sequence.pdf \
         figures/fungal_up_down_transformer_upstream_1003bp_pad_for_undersized.pdf \
         figures/uncharacterized-genes-profile-enricment.pdf
	-@bash check-figures.sh   # non-fatal: report sizes/scaling after every export

figures/%.pdf: $(DRAWIO_SRC)/%.drawio.png
	$(DRAWIO_EXPORT) $@ $<

# `make fig` -- FORCE-(re)render every figure the manuscript references, ignoring
# timestamps (unlike `figures`, which only rebuilds when a source is newer and otherwise
# says "Nothing to be done"). It collects `figures/NAME.pdf` paths straight from the .tex,
# maps each to its draw.io source ($(DRAWIO_SRC)/NAME.drawio.png), re-exports it, then runs
# the check. New figures are picked up automatically -- no list to maintain. Use this right
# after editing a draw.io source.
.PHONY: fig
fig:
	@set -e; \
	names=$$(grep -rhE 'figures/[A-Za-z0-9._-]+\.pdf' $(SHARED) 2>/dev/null \
	         | grep -vE '^[[:space:]]*%' \
	         | grep -oE 'figures/[A-Za-z0-9._-]+\.pdf' \
	         | sed -E 's#figures/(.*)\.pdf#\1#' | sort -u); \
	[ -n "$$names" ] || { echo "no figures/NAME.pdf references found in the manuscript"; exit 0; }; \
	for n in $$names; do \
	  src="$(DRAWIO_SRC)/$$n.drawio.png"; \
	  if [ -f "$$src" ]; then \
	    echo "  render  figures/$$n.pdf  <-  $$src"; \
	    $(DRAWIO_EXPORT) figures/$$n.pdf "$$src"; \
	  else \
	    echo "  SKIP    figures/$$n.pdf  (no draw.io source: $$src)"; \
	  fi; \
	done; \
	echo; bash check-figures.sh || true

# WYSIWYG figure contract (codifies steps 2-3): each figure PDF must fit the 180x170 mm
# print box, and all figures must be placed true-size (\tcfig, never \tcfigfit). Strict --
# exits non-zero on any violation, so it can gate a build or CI:  make checkfigs
checkfigs:
	@bash check-figures.sh

# Flatten to a single .tex for Springer Nature submission (no \input; .bbl inlined
# if a submission.bbl is present). Springer requires one file -- do this at submit time.
flat: submission.tex $(SHARED)
	python3 flatten_tex.py submission.tex submission-flat.tex

clean:
	rm -f submission.pdf editing.pdf twocolumn.pdf figure-proto.pdf figure-limits.pdf submission-flat.tex
