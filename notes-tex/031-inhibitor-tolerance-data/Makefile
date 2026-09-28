# notes-tex/031-inhibitor-tolerance-data/Makefile
# [[experiments.031-env-chemgen-inhibitor-tolerance]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/031-inhibitor-tolerance-data/Makefile
#
# `make` builds 031-inhibitor-tolerance-data.pdf; `make check` runs the gate;
# `make plots` re-converts every referenced panel from the true-size SVGs in
# $ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/; `make tables` regenerates
# tables/*.tex from the experiment's result CSVs.
#
# No Zotero collection yet: the document cites nothing through natbib. The works it
# names (Vanacloig-Pedros 2022, Hillenmeyer 2008, Lian 2019) are given by DOI in
# prose, as 025-additive-baselines does.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 031-env-chemgen-inhibitor-tolerance

include ../common/Makefile.common

.PHONY: tables
tables:
	cd $(abspath $(REPO)) && PYTHONPATH=$(abspath $(REPO)) $(PYTHON) experiments/031-env-chemgen-inhibitor-tolerance/scripts/notes_tex_tables.py
