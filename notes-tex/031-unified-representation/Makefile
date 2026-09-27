# notes-tex/031-unified-representation/Makefile
# [[experiments.031-env-chemgen-inhibitor-tolerance]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/031-unified-representation/Makefile
#
# `make` builds 031-unified-representation.pdf; `make check` runs the gate;
# `make tables` regenerates tables/*.tex from the experiment's result CSVs;
# `make diagrams` re-renders the mermaid representation figure and fits it to the
# text width.
#
# No Zotero collection yet: the works named (Vanacloig-Pedros 2022, Hillenmeyer 2008,
# Hoepfner 2014, Wildenhain 2015) are given by DOI in prose, as 025-additive-baselines
# does.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 031-env-chemgen-inhibitor-tolerance

include ../common/Makefile.common

DIAGRAMS := experiments.031-env-chemgen-inhibitor-tolerance.mermaid.unified-input
PDF_OUT  := $(REPO)/notes/assets/pdf-output
FIG_W_PT := 500

.PHONY: tables
tables:
	cd $(abspath $(REPO)) && PYTHONPATH=$(abspath $(REPO)) $(PYTHON) experiments/031-env-chemgen-inhibitor-tolerance/scripts/notes_tex_representation_tables.py

.PHONY: diagrams
diagrams:
	@for d in $(DIAGRAMS); do \
	  (cd $(REPO) && bash notes/assets/publish/scripts/mermaid_pdf.sh notes/$$d.md >/dev/null); \
	  src=$(PDF_OUT)/$$d.pdf; dst=figures/unified_input.pdf; \
	  w=$$(pdfinfo $$src | awk '/Page size/{print $$3}'); h=$$(pdfinfo $$src | awk '/Page size/{print $$5}'); \
	  H=$$(awk -v w=$$w -v h=$$h -v W=$(FIG_W_PT) 'BEGIN{printf "%.2f", h*W/w}'); \
	  gs -q -o $$dst -sDEVICE=pdfwrite -dFIXEDMEDIA -dPDFFitPage \
	     -dDEVICEWIDTHPOINTS=$(FIG_W_PT) -dDEVICEHEIGHTPOINTS=$$H $$src; \
	  echo "  $$src ($$w x $$h pt) -> $$dst ($(FIG_W_PT) x $$H pt)"; \
	done
