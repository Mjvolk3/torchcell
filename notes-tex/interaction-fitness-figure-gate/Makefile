# notes-tex/interaction-fitness-figure-gate/Makefile
# [[experiments.030-solid-growth-multi.scripts.figure2_gate_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/interaction-fitness-figure-gate/Makefile
#
# Mockups of manuscript Figure 2 (fitness and gene-gene interactions) and its supplement,
# held for a decision. No Zotero collection and no citations: the document holds our own
# results, and the outside works are named in prose.
#
#   make numbers   re-read every result file into the macros, the table rows and the
#                  fitness panel (figure2_gate_numbers.py; WANDB=1 refetches the
#                  transformer rows, default 1)
#   make plots     convert the newest renders into figures/
#   make           build the PDF
#   make check     formatting / provenance / status gate

DOC := interaction-fitness-figure-gate
ZOTERO_COLLECTION          :=
ZOTERO_PERSONAL_COLLECTION :=
PLOT_SLUGS := 030-solid-growth-multi
PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
WANDB ?= 1

include ../common/Makefile.common

ROOT    := $(abspath $(REPO))
SCRIPTS := $(ROOT)/experiments/030-solid-growth-multi/scripts
.PHONY: numbers
numbers:
	# the numbers script first: it writes the fitness-head cache the sheet's panel f reads
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTHON) $(SCRIPTS)/figure2_gate_numbers.py $(if $(filter 1,$(WANDB)),--wandb,)
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTHON) $(SCRIPTS)/figure2_gate_sheet.py $(if $(filter 1,$(WANDB)),--wandb,)
	# then the figure-name macros, so they point at the sheets just rendered
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTHON) $(SCRIPTS)/figure2_gate_numbers.py --names-only
