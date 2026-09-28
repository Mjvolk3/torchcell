# notes-tex/034-isobutanol-wetlab/Makefile
# [[experiments.034-isobutanol-wetlab.strain-panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/034-isobutanol-wetlab/Makefile
#
# No Zotero collection yet, so references.bib is empty and the two works this
# document rests on are given by DOI in prose, the same way 025-additive-baselines
# and 010-additive-baselines do it. When a collection named 034-isobutanol-wetlab
# exists, set ZOTERO_PERSONAL_COLLECTION below, run
# `python scripts/lit_bib_store.py --name 034-isobutanol-wetlab`, then `make bib-pull`.
#
# Tables are emitted by experiments/034-isobutanol-wetlab/scripts/strain_panel.py
# and are never hand-edited. Regenerate with:
#   ~/miniconda3/envs/torchcell/bin/python \
#       experiments/034-isobutanol-wetlab/scripts/strain_panel.py

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 034-isobutanol-wetlab

include ../common/Makefile.common

.PHONY: tables
tables:
	@$(PYTHON) $(REPO)/experiments/034-isobutanol-wetlab/scripts/strain_panel.py
