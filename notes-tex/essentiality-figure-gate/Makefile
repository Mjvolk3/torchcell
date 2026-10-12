# notes-tex/essentiality-figure-gate/Makefile
# [[experiments.028-gene-essentiality.scripts.essentiality_gate_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/essentiality-figure-gate/Makefile
#
# Mockups of the manuscript's gene-essentiality figure, held for a decision. No Zotero
# collection and no citations: the document holds our own results, and Merzbacher 2025,
# yeast-GEM 9.0.2 and SGD are named in prose with their identifiers, as 028 does.
#
#   make numbers   re-read every 028 result file into the macros and the table rows
#   make plots     convert the newest summary sheets of 028 into figures/
#   make           build the PDF
#   make check     formatting / provenance / status gate

DOC := essentiality-figure-gate
ZOTERO_COLLECTION          :=
ZOTERO_PERSONAL_COLLECTION :=
PLOT_SLUGS := 028-gene-essentiality
PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common

ROOT    := $(abspath $(REPO))
SCRIPTS := $(ROOT)/experiments/028-gene-essentiality/scripts
.PHONY: numbers
numbers:
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTHON) $(SCRIPTS)/essentiality_gate_numbers.py
