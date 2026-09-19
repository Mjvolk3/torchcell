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
PLOT_SLUGS := 025-solid-growth 029-solid-growth-ko
MERMAID_NOTE := $(REPO)/notes/experiments.025-solid-growth.s3-closure.mermaid.pipeline.md

include ../common/Makefile.common

# The mermaid renders are fitted to the diagram, not to the page, so they come out wider
# than the 182 mm text block; Ghostscript rescales each to 179 mm (Nature full width)
# keeping the aspect ratio, and `make check` reads that width from the MediaBox.
# Two diagrams: the pipeline (figure a) and the 025-versus-029 build stages.
MERMAID_NOTES := experiments.025-solid-growth.s3-closure.mermaid.pipeline:s3_closure_pipeline \
                 experiments.025-solid-growth.s3-closure.mermaid.label-policy:closure_label_policy
mermaid:
	@for pair in $(MERMAID_NOTES); do \
	  note=$${pair%%:*}; out=$${pair##*:}; \
	  (cd $(REPO) && bash notes/assets/publish/scripts/mermaid_pdf.sh notes/$$note.md "#F5EEDD" >/dev/null); \
	  src=$(REPO)/notes/assets/pdf-output/$$note.pdf; \
	  w=$$(grep -a -m1 -o '/MediaBox \[[^]]*\]' $$src | tr -dc '0-9. ' | awk '{print $$3}'); \
	  h=$$(grep -a -m1 -o '/MediaBox \[[^]]*\]' $$src | tr -dc '0-9. ' | awk '{print $$4}'); \
	  tw=$$(awk 'BEGIN{printf "%.2f", 179/25.4*72}'); \
	  th=$$(awk -v w="$$w" -v h="$$h" -v tw="$$tw" 'BEGIN{printf "%.2f", h*tw/w}'); \
	  echo "  mermaid $$note $$w x $$h pt -> $$tw x $$th pt"; \
	  gs -q -o figures/$$out.pdf -sDEVICE=pdfwrite -dFIXEDMEDIA -dPDFFitPage \
	     -dDEVICEWIDTHPOINTS=$$tw -dDEVICEHEIGHTPOINTS=$$th $$src; \
	done

.PHONY: mermaid
