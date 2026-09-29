# notes-tex/033-post-query-analysis/Makefile
# [[experiments.033-env-chemgen-pooled]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/033-post-query-analysis/Makefile
#
# `make` builds 033-post-query-analysis.pdf; `make check` runs the gate;
# `make plots` converts the panels from the true-size SVGs in
# $ASSET_IMAGES_DIR/033-env-chemgen-pooled/; `make tables` regenerates tables/*.tex
# from the experiment's result CSVs.
#
# No Zotero collection: the document cites nothing through natbib. The works it names
# (Vanacloig-Pedros 2022, Hillenmeyer 2008, Hoepfner 2014, Wildenhain 2015) are given
# by DOI in prose, as the 031 documents do.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 033-env-chemgen-pooled

include ../common/Makefile.common

.PHONY: tables
tables:
	cd $(abspath $(REPO)) && PYTHONPATH=$(abspath $(REPO)) $(PYTHON) experiments/033-env-chemgen-pooled/scripts/notes_tex_post_query_tables.py
