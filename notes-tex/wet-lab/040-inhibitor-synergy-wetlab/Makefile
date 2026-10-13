# notes-tex/wet-lab/040-inhibitor-synergy-wetlab/Makefile
# [[experiments.040-inhibitor-synergy-wetlab]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/wet-lab/040-inhibitor-synergy-wetlab/Makefile
#
#   make         -> 040-inhibitor-synergy-wetlab.pdf
#   make check   -> the formatting / provenance / style gate
#   make plots   -> re-convert the eleven matplotlib panels into figures/
#   make tables  -> re-generate tables/*.tex from the result CSVs
#
# No Zotero collection yet: the document cites nothing through natbib, so
# references.bib is empty and `make check` reports zero citations. The outside
# works it names (Vanacloig 2022, Hoepfner 2014, Hillenmeyer 2008, Lian 2019,
# Bliss 1939, Loewe) are given by DOI or by name in prose, as in
# 025-additive-baselines. When a collection exists, set
# ZOTERO_PERSONAL_COLLECTION, run `python scripts/lit_bib_store.py --name
# 040-inhibitor-synergy-wetlab`, then `make bib-pull`.

PYTHON ?= $(firstword $(wildcard $(HOME)/miniconda3/envs/torchcell/bin/python) python)
PLOT_SLUGS := 040-inhibitor-synergy-wetlab

include ../../common/Makefile.common

# The 040 panels still carry their generation timestamp
# (single_agent_curves_2026-10-10-02-26-56.svg), because the figure set is still
# being iterated and the repo convention keeps the timestamp until it is settled.
# Makefile.common's `plots` target resolves figures/NAME.pdf from an exactly
# named NAME.svg, so the newest timestamped file of each panel is linked to that
# name first, in a gitignored staging directory, and PLOT_SRCS is pointed at it.
# Assigned AFTER the include so it wins over the default (`:=` there would
# otherwise overwrite anything set above it).
PANEL_DIR  := panel-svg
PLOT_SRCS  := $(PANEL_DIR)
PANEL_SRC  := $(REPO)/notes/assets/images/040-inhibitor-synergy-wetlab
TABLE_SCRIPT := $(REPO)/experiments/040-inhibitor-synergy-wetlab/scripts/notes_tex_tables.py

plots: panel-links

.PHONY: panel-links tables
panel-links:
	@mkdir -p $(PANEL_DIR)
	@for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	  n=$$(basename $$f .pdf); \
	  s=$$(ls -1 $(PANEL_SRC)/$${n}_*.svg 2>/dev/null | sort | tail -1); \
	  if [ -n "$$s" ]; then ln -sf "$$(realpath $$s)" $(PANEL_DIR)/$$n.svg; \
	  else echo "  no timestamped SVG for $$n in $(PANEL_SRC)"; exit 1; fi; \
	done

tables:
	@$(PYTHON) $(TABLE_SCRIPT)
