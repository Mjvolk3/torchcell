# notes-tex/wet-lab/hydrolysate-figure-gate/Makefile
# [[experiments.040-inhibitor-synergy-wetlab.scripts.figure_gate_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/wet-lab/hydrolysate-figure-gate/Makefile
#
# Mockups of the hydrolysate inhibitor figure (manuscript R5, drug exposure) and its
# supplement, held for a decision. No Zotero collection and no citations: the document
# holds our own results.
#
#   make numbers   re-draw the sheets and the number macros from the result files
#   make plots     convert the newest sheets into figures/
#   make           build the PDF
#   make check     formatting / provenance / status gate

DOC := hydrolysate-figure-gate
ZOTERO_COLLECTION          :=
ZOTERO_PERSONAL_COLLECTION :=
PLOT_SLUGS := 040-inhibitor-synergy-wetlab
PYTHON ?= $(HOME)/miniconda3/envs/torchcell/bin/python

include ../../common/Makefile.common

ROOT    := $(abspath $(REPO))
SCRIPTS := $(ROOT)/experiments/040-inhibitor-synergy-wetlab/scripts
.PHONY: numbers
numbers:
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTHON) $(SCRIPTS)/figure_gate_panels.py
