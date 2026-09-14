# notes-tex/025-additive-baselines/Makefile
# [[experiments.025-solid-growth.scripts.additive_baselines_025_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/025-additive-baselines/Makefile
#
# Two builds from one set of sections. `make` builds the long version,
# 025-additive-baselines.pdf; `make short` builds 025-additive-baselines-short.pdf,
# whose root sets \longfalse so every \iflong block in sections/ is dropped.
# `make check` gates the long build, `make check-short` the short one. Publish
# either with notes-tex/common/zotero_publish.py (the short one needs
# `--pdf 025-additive-baselines-short`).
#
# No Zotero collection yet: the document cites nothing through natbib. The outside
# works it names (the Visani preprint, Tran 2026, Kuzmin 2018) are given by DOI in
# prose, as in 010-additive-baselines. When a collection exists, set
# ZOTERO_PERSONAL_COLLECTION, run `python scripts/lit_bib_store.py --name
# 025-additive-baselines`, then `make bib-pull`.
#
# Panels come from two experiments' image directories: the 025 panels and tables
# from additive_baselines_025_panels.py, and two 010 figures (graph_penalty_vs_loss,
# paired_prediction_agreement) the long version carries over, so PLOT_SLUGS lists both.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 025-solid-growth 010-kuzmin-tmi

include ../common/Makefile.common

SHORT := $(DOC)-short

.PHONY: short check-short

short: $(SHORT).pdf

$(SHORT).pdf: $(SHORT).tex $(SECTIONS) $(TABLES) $(COMMON)/tcdoc.sty references.bib $(FIGURES) $(SHARED_STY)
	$(TECTONIC) -X compile $(SHORT).tex --keep-logs --synctex --outdir .

check-short: $(SHORT).pdf
	@$(PYTHON) $(COMMON)/check_doc.py $(SHORT)
