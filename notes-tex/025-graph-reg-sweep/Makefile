# notes-tex/025-graph-reg-sweep/Makefile
# [[experiments.025-solid-growth.graph-reg-sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/025-graph-reg-sweep/Makefile
#
# No citations: the arms are named by config and read from W&B. The tables under tables/
# and the CSVs the figures read are written by
# experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py; the figures by
# graph_reg_sweep_plots.py into $ASSET_IMAGES_DIR/025-solid-growth/ (`make plots` converts
# the true-size SVGs into figures/).
#
#   make refresh   -> pull W&B again, refill the tables, redraw the figures, rebuild
#   make           -> 025-graph-reg-sweep.pdf
#   make check     -> formatting / provenance / style gate

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 025-solid-growth

include ../common/Makefile.common

refresh:
	cd $(REPO) && PYTHONPATH=$(REPO) $(PYTHON) experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py
	cd $(REPO) && PYTHONPATH=$(REPO) $(PYTHON) experiments/025-solid-growth/scripts/graph_reg_sweep_plots.py
	$(MAKE) plots
	$(MAKE)

.PHONY: refresh
