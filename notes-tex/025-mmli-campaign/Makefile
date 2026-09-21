# notes-tex/025-mmli-campaign/Makefile
# [[experiments.025-solid-growth.mmli-campaign]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/025-mmli-campaign/Makefile
#
# No citations and no figures: the document is two generated tables and the reading of
# them. Both tables come from experiments/025-solid-growth/scripts/mmli_campaign_plan.py.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common

# Refill the tables from W&B and the live queue, then rebuild.
refresh:
	cd $(REPO) && PYTHONPATH=$(REPO) $(PYTHON) experiments/025-solid-growth/scripts/mmli_campaign_plan.py --refresh-queue
	$(MAKE)

.PHONY: refresh
