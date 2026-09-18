# notes-tex/025-s3-closure/Makefile
# [[experiments.025-solid-growth.s3-closure]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/025-s3-closure/Makefile
#
# No Zotero collection yet: the sources (Baryshnikova 2010, Costanzo 2016, Kuzmin 2018,
# Kuzmin 2020) are named in prose with their mirror keys. Once a collection of this
# document's own exists, set ZOTERO_PERSONAL_COLLECTION and switch to \citep.
#
# Panels come from experiments/025-solid-growth/scripts/s3_closure_recompute.py, which
# writes the SVGs into $ASSET_IMAGES_DIR/025-solid-growth/ (`make plots` converts them).
# Figure a is a mermaid diagram kept in a dendron note; `make mermaid` renders it and
# copies the PDF into figures/.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python
PLOT_SLUGS := 025-solid-growth
MERMAID_NOTE := $(REPO)/notes/experiments.025-solid-growth.s3-closure.mermaid.pipeline.md

include ../common/Makefile.common

# The mermaid render is fitted to the diagram, not to the page, so it comes out wider
# than the 182 mm text block; Ghostscript rescales it to 179 mm (Nature full width)
# keeping the aspect ratio, and `make check` reads that width from the MediaBox.
MERMAID_PDF := $(REPO)/notes/assets/pdf-output/experiments.025-solid-growth.s3-closure.mermaid.pipeline.pdf
mermaid:
	cd $(REPO) && bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.025-solid-growth.s3-closure.mermaid.pipeline.md "#F5EEDD"
	@w=$$(grep -a -m1 -o '/MediaBox \[[^]]*\]' $(MERMAID_PDF) | tr -dc '0-9. ' | awk '{print $$3}'); \
	h=$$(grep -a -m1 -o '/MediaBox \[[^]]*\]' $(MERMAID_PDF) | tr -dc '0-9. ' | awk '{print $$4}'); \
	tw=$$(awk 'BEGIN{printf "%.2f", 179/25.4*72}'); \
	th=$$(awk -v w="$$w" -v h="$$h" -v tw="$$tw" 'BEGIN{printf "%.2f", h*tw/w}'); \
	echo "  mermaid $$w x $$h pt -> $$tw x $$th pt"; \
	gs -q -o figures/s3_closure_pipeline.pdf -sDEVICE=pdfwrite -dFIXEDMEDIA -dPDFFitPage \
	   -dDEVICEWIDTHPOINTS=$$tw -dDEVICEHEIGHTPOINTS=$$th $(MERMAID_PDF)

.PHONY: mermaid
