# notes-tex/publishing-system/Makefile
# [[publishing-system]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/publishing-system/Makefile
#
# The document about the build and publishing system itself. No Zotero collection:
# it cites nothing. Its two figures are mermaid diagrams, not experiment panels, so
# `make diagrams` (below) renders them through the repo's mermaid pipeline and scales
# the page to the 182 mm text block; `make plots` has nothing to convert here.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common

# Mermaid sources are Dendron notes, rendered by notes/assets/publish/scripts/mermaid_pdf.sh
# to notes/assets/pdf-output/<fname>.pdf. mmdc fits the page to the diagram at 600 pt
# (212 mm), wider than the text block, so the page is scaled to 510 pt (180 mm) with
# Ghostscript, aspect preserved. Fonts scale with it (13 px -> about 8 pt).
DIAGRAMS := publishing-system.mermaid.paper-build publishing-system.mermaid.notes-tex-build
PDF_OUT  := $(REPO)/notes/assets/pdf-output
FIG_W_PT := 510

.PHONY: diagrams
diagrams:
	@for d in $(DIAGRAMS); do \
	  (cd $(REPO) && bash notes/assets/publish/scripts/mermaid_pdf.sh notes/$$d.md >/dev/null); \
	  src=$(PDF_OUT)/$$d.pdf; dst=figures/$${d#publishing-system.mermaid.}.pdf; \
	  w=$$(pdfinfo $$src | awk '/Page size/{print $$3}'); h=$$(pdfinfo $$src | awk '/Page size/{print $$5}'); \
	  H=$$(awk -v w=$$w -v h=$$h -v W=$(FIG_W_PT) 'BEGIN{printf "%.2f", h*W/w}'); \
	  gs -q -o $$dst -sDEVICE=pdfwrite -dFIXEDMEDIA -dPDFFitPage \
	     -dDEVICEWIDTHPOINTS=$(FIG_W_PT) -dDEVICEHEIGHTPOINTS=$$H $$src; \
	  echo "  $$src ($$w x $$h pt) -> $$dst ($(FIG_W_PT) x $$H pt)"; \
	done
