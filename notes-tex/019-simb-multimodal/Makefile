# notes-tex/019-simb-multimodal/Makefile
# [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/019-simb-multimodal/Makefile

DOC := main

# No Zotero collection is set. Every work this document refers to is named inline
# with its DOI rather than cited, because the document's subject is our own runs
# and the outside literature enters only as the provenance of a dataset. Setting a
# collection here would create a second curated list to keep in sync with the
# manuscript's, for four references. `make bib` is therefore not available in this
# document; `make check` still runs, since it fails only on \cite keys, of which
# there are none.
ZOTERO_COLLECTION          :=
ZOTERO_PERSONAL_COLLECTION :=

# `make plots` (in Makefile.common) reads ONE directory. This document draws on
# four experiments' image directories, so PLOT_SLUG names the primary one and
# `plots-all` below sweeps the rest. Panels referenced here are written by their
# scripts under STABLE names (no timestamp), because a timestamped filename cannot
# be the target of a Makefile rule.
PLOT_SLUG := 019-simb-multimodal
PLOT_EXTRA_SLUGS := 020-cachera-betaxanthin 023-metabolome-betaxanthin-joint \
                    012-sameith-kemmeren-expression

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common

.PHONY: plots-all
plots-all: plots
	@for slug in $(PLOT_EXTRA_SLUGS); do \
	  for f in $$(grep -rhoE 'figures/[A-Za-z0-9._-]+\.pdf' $(DEPS) 2>/dev/null | sort -u); do \
	    n=$$(basename $$f .pdf); \
	    s="$(REPO)/notes/assets/images/$$slug/$$n.svg"; \
	    if [ -f "$$s" ]; then echo "  $$s -> $$f"; \
	      rsvg-convert --zoom $(PLOT_ZOOM) -f pdf -o "$$f" "$$s"; fi; \
	  done; \
	done
