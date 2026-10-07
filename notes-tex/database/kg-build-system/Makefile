# notes-tex/database/kg-build-system/Makefile
# [[kg-build-system]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/database/kg-build-system/Makefile
#
# The document about the knowledge-graph build, release and serving system. No
# Zotero collection: it cites nothing from the library. Its three figures are mermaid
# diagrams, so `make diagrams` renders them through the repo's mermaid pipeline and
# scales the page to the text block; `make plots` has nothing to convert here.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../../common/Makefile.common

# Mermaid sources are Dendron notes, rendered by notes/assets/publish/scripts/mermaid_pdf.sh
# to notes/assets/pdf-output/<fname>.pdf. mmdc fits the page to the diagram, wider than
# the text block, so the page is scaled to 510 pt (180 mm) with Ghostscript, aspect
# preserved; fonts scale with it.
DIAGRAMS := kg-build-system.mermaid.build-flow kg-build-system.mermaid.serve-flow kg-build-system.mermaid.release-lifecycle
PDF_OUT  := $(REPO)/notes/assets/pdf-output
FIG_W_PT := 510

.PHONY: diagrams
diagrams:
	@for d in $(DIAGRAMS); do \
	  (cd $(REPO) && bash notes/assets/publish/scripts/mermaid_pdf.sh notes/$$d.md >/dev/null); \
	  src=$(PDF_OUT)/$$d.pdf; dst=figures/$${d#kg-build-system.mermaid.}.pdf; \
	  w=$$(pdfinfo $$src | awk '/Page size/{print $$3}'); h=$$(pdfinfo $$src | awk '/Page size/{print $$5}'); \
	  H=$$(awk -v w=$$w -v h=$$h -v W=$(FIG_W_PT) 'BEGIN{printf "%.2f", h*W/w}'); \
	  gs -q -o $$dst -sDEVICE=pdfwrite -dFIXEDMEDIA -dPDFFitPage \
	     -dDEVICEWIDTHPOINTS=$(FIG_W_PT) -dDEVICEHEIGHTPOINTS=$$H $$src; \
	  echo "  $$src ($$w x $$h pt) -> $$dst ($(FIG_W_PT) x $$H pt)"; \
	done
