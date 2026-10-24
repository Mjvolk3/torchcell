# notes-tex/multimodal/043-datasets-primer/Makefile
# [[experiments.043-expression-proteome-morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/multimodal/043-datasets-primer/Makefile
#
# `make` builds 043-datasets-primer.pdf; `make check` gates it; `make plots` converts the
# 043 panels from $ASSET_IMAGES_DIR/043-expression-proteome-morphology/ into figures/;
# `make publish` uploads the build to Zotero under torchcell/notes-tex/multimodal/
# 043-datasets-primer (our own PDF, the one sanctioned write).
#
# Tables are written by experiments/043-expression-proteome-morphology/scripts/
# primer_043_tables.py from the result files; `make tables` reruns it.
#
# The three mermaid diagrams (data flow, input side, output side) are rendered from their
# Dendron notes by notes/assets/publish/scripts/mermaid_pdf.sh into notes/assets/pdf-output/
# and copied into figures/ here, so the PDF rebuilds whenever a diagram is re-rendered.
#
# No Zotero collection: the document cites nothing through natbib. Papers are named by
# citation key and DOI in prose.

PLOT_SLUG := 043-expression-proteome-morphology

include ../../common/Makefile.common

.PHONY: tables diagrams publish publish-dry

MERMAID_SRC  := $(REPO)/notes/assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid
MERMAID_FIGS := $(foreach m,data-flow input-side output-side,figures/mermaid-$(m).pdf)

$(DOC).pdf: $(MERMAID_FIGS)

# mermaid_pdf.sh renders at 211.7 mm, past the 182 mm text block, so the copy is scaled
# (pypdf, vector, content and MediaBox together) to 180 mm wide: the width `make check`
# accepts and the width a full-width Nature figure prints at.
figures/mermaid-%.pdf: $(MERMAID_SRC).%.pdf
	$(PYTHON) -c "import sys, pypdf; r = pypdf.PdfReader(sys.argv[1]); w = pypdf.PdfWriter(); \
	p = r.pages[0]; p.scale_by(180 / 25.4 * 72 / float(p.mediabox.width)); w.add_page(p); \
	w.write(sys.argv[2])" $< $@

diagrams: $(MERMAID_FIGS)

tables:
	cd $(REPO) && $(PYTHON) experiments/043-expression-proteome-morphology/scripts/primer_043_tables.py

publish: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC)

publish-dry: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC) --dry-run
