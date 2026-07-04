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
DRAWIO_SRC := ../../notes/assets/drawio
# Platform-aware draw.io headless export. On macOS use the installed draw.io.app CLI
# directly (no Xvfb/FUSE); on Linux fall back to the drawio-desktop AppImage under Xvfb.
# APPIMAGE_EXTRACT_AND_RUN avoids needing FUSE. Override DRAWIO to point elsewhere.
ifeq ($(shell uname -s),Darwin)
  DRAWIO        ?= /Applications/draw.io.app/Contents/MacOS/draw.io
  DRAWIO_EXPORT  = "$(DRAWIO)" -x -f pdf --crop -o
else
  DRAWIO        ?= /tmp/drawio.AppImage
  DRAWIO_EXPORT  = APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a $(DRAWIO) --no-sandbox --disable-gpu -x -f pdf --crop -o
endif

SHARED := content.tex preamble.tex references.bib sn-jnl.cls sn-nature.bst $(wildcard sections/*.tex)
VIEWS  := submission editing twocolumn

# Figure PDFs the manuscript references THAT have a .drawio.png source. The gate depends on
# these so `make paper` re-exports any figure whose source is newer BEFORE measuring it --
# closing the "edited draw.io but the gate still reads the stale PDF" trap. Figures with no
# source yet (fig2..5 placeholders) are omitted here so they never break the build; the
# size gate simply ignores PDFs that do not exist.
DRAWIO_FIGS := $(shell for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(SHARED) 2>/dev/null | grep -vE '^[[:space:]]*%' | sort -u); do \
                 n=$$(basename $$f .pdf); \
                 [ -f "$(DRAWIO_SRC)/$$n.drawio.png" ] && echo "figures/$$n.pdf"; \
               done)

.PHONY: paper all submission editing twocolumn figproto figlimits figures fig checkfigs wordcount flat clean
# `make paper` builds the three views, then prints the word-count report (non-fatal: the
# leading '-' means a bad/absent report never fails the build).
paper: $(addsuffix .pdf,$(VIEWS))
	-@$(PYTHON) wordcount.py
all: paper figproto figlimits

submission: submission.pdf
editing: editing.pdf
twocolumn: twocolumn.pdf
figproto: figure-proto.pdf
figlimits: figure-limits.pdf

# Per-section word-count report vs the \cb{...words} budgets (body prose only; figure and
# table environments and \cb tags stripped, then texcount counts the rest). Informational,
# never gates. Override the interpreter with PYTHON=... if needed (stdlib + texcount only).
PYTHON ?= python3
wordcount:
	@$(PYTHON) wordcount.py

# The EDITING view embeds a per-section word-budget table on a final page. wordcount.tex is
# regenerated from the sources whenever they (or the script) change, and editing.pdf depends
# on it so the embedded counts always match the compiled draft. submission/twocolumn define
# \cb as a no-op and do NOT include it, so only the editing view carries the report.
editing.pdf: wordcount.tex
wordcount.tex: wordcount.py $(wildcard sections/*.tex)
	@$(PYTHON) wordcount.py --tex $@

# Each view wrapper \input's the shared body, so rebuild when either changes.
# HARD GATE: every manuscript view depends on checkfigs, so ALL builds (submission,
# editing, twocolumn -- and `make paper`, which includes them) refuse to compile while
# any figure is out of spec, keeping figure sizing consistent across every view.
# checkfigs is .PHONY, so it runs on every build and aborts before Tectonic if a figure
# is over 180x240 mm or scaled. Read the `make checkfigs` report to see which figure is
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

# Post-export crop-to-ink for figures whose draw.io PDF has phantom whitespace.
# When a MathJax ($$...$$) label renders wider than its cell box, draw.io's --crop
# extends the page to the formula's *invisible* bounding box, padding the figure with
# blank margin (Fig 1's panel g/h equations do this: ink ends at 178.8 mm but the crop
# ran to 185.6 mm). pdfcrop trims the page to the true ink box -- NO scaling, fonts
# unchanged, only dead margin removed -- so the figure lands at ~179 mm and is placed
# true-size with \tcfig (passes both the size gate and the no-\tcfigfit scale gate).
# Listed by figure NAME; a figure not listed is exported verbatim. This is preferred
# over an export -s down-scale, which would shrink the fonts too.
PDFCROP    ?= pdfcrop
CROP_FIGS  := Fig1-torchcell-overview
# crop-to-ink in place if NAME ($1) is in CROP_FIGS; safe no-op otherwise.
define crop_if_needed
	@if printf ' %s ' '$(CROP_FIGS)' | grep -q ' $(1) '; then \
	  echo "  crop-to-ink  figures/$(1).pdf  (chop phantom MathJax margin)"; \
	  $(PDFCROP) --margins 0 "figures/$(1).pdf" "figures/$(1).pdf.crop" >/dev/null && \
	  mv "figures/$(1).pdf.crop" "figures/$(1).pdf"; \
	fi
endef

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
         figures/uncharacterized-genes-profile-enricment.pdf \
         figures/Fig2-gene-interactions.pdf \
         figures/Fig3-systems-biology-fitness-expression-morphology.pdf \
         figures/Fig4-natural-variation.pdf
	-@bash check-figures.sh   # non-fatal: report sizes/scaling after every export

figures/%.pdf: $(DRAWIO_SRC)/%.drawio.png
	$(DRAWIO_EXPORT) $@ $<
	$(call crop_if_needed,$*)

# Fig 2 concept diagram (nested cell representation + iBioFoundry loop) has a PLAIN
# .drawio source (not .drawio.png), so this explicit rule -- which wins over the pattern
# rule above -- exports it. math=0, no MathJax overflow, so no crop-to-ink needed.
figures/Fig2-gene-interactions.pdf: $(DRAWIO_SRC)/Fig2-gene-interactions.drawio
	$(DRAWIO_EXPORT) $@ $<

# Fig 3 (systems biology: fitness / expression / morphology) is likewise a PLAIN
# .drawio source, so it needs its own explicit rule (the pattern rule expects .drawio.png).
figures/Fig3-systems-biology-fitness-expression-morphology.pdf: $(DRAWIO_SRC)/Fig3-systems-biology-fitness-expression-morphology.drawio
	$(DRAWIO_EXPORT) $@ $<

# Fig 4 (natural variation vs model-designed perturbation) -- PLAIN .drawio, explicit rule.
figures/Fig4-natural-variation.pdf: $(DRAWIO_SRC)/Fig4-natural-variation.drawio
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
	    if printf ' %s ' '$(CROP_FIGS)' | grep -q " $$n "; then \
	      echo "  crop-to-ink  figures/$$n.pdf  (chop phantom MathJax margin)"; \
	      $(PDFCROP) --margins 0 "figures/$$n.pdf" "figures/$$n.pdf.crop" >/dev/null && \
	      mv "figures/$$n.pdf.crop" "figures/$$n.pdf"; \
	    fi; \
	  else \
	    echo "  SKIP    figures/$$n.pdf  (no draw.io source: $$src)"; \
	  fi; \
	done; \
	echo; bash check-figures.sh || true

# WYSIWYG figure contract (codifies steps 2-3): each figure PDF must fit the 180x240 mm
# print box, and all figures must be placed true-size (\tcfig, never \tcfigfit). Strict --
# exits non-zero on any violation, so it can gate a build or CI:  make checkfigs
# DEPENDS on $(DRAWIO_FIGS): any figure whose .drawio.png source is newer than its PDF is
# re-exported (via the figures/%.pdf rule) BEFORE the check runs, so `make paper` measures
# what you just drew -- not a stale export. This is the fix for the "edited draw.io, saved,
# but make paper still reports the old size" trap.
checkfigs: $(DRAWIO_FIGS)
	@bash check-figures.sh

# Flatten to a single .tex for Springer Nature submission (no \input; .bbl inlined
# if a submission.bbl is present). Springer requires one file -- do this at submit time.
flat: submission.tex $(SHARED)
	python3 flatten_tex.py submission.tex submission-flat.tex

clean:
	rm -f submission.pdf editing.pdf twocolumn.pdf figure-proto.pdf figure-limits.pdf submission-flat.tex wordcount.tex
