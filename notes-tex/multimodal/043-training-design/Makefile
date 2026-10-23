# notes-tex/multimodal/043-training-design/Makefile
# [[experiments.043-expression-proteome-morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/multimodal/043-training-design/Makefile
#
# `make` builds 043-training-design.pdf; `make check` gates it; `make tables` reruns
# experiments/043-expression-proteome-morphology/scripts/training_design_043_tables.py;
# `make diagrams` copies the rendered mermaid diagram into figures/; `make publish` uploads
# the build to Zotero under torchcell/notes-tex/multimodal/043-training-design (our own PDF,
# the one sanctioned write).
#
# The data and model flow diagram is rendered from its Dendron note
# (notes/experiments.043-expression-proteome-morphology.mermaid.training-design.md) by
# notes/assets/publish/scripts/mermaid_pdf.sh into notes/assets/pdf-output/ and copied here.
#
# No Zotero collection: the document cites nothing through natbib.

include ../../common/Makefile.common

.PHONY: tables diagrams publish publish-dry

MERMAID_SRC  := $(REPO)/notes/assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid
MERMAID_FIGS := figures/mermaid-training-design.pdf

$(DOC).pdf: $(MERMAID_FIGS)

# mermaid_pdf.sh renders at 211.7 mm, past the 182 mm text block, so the copy is scaled
# (pypdf, vector, content and MediaBox together) to 180 mm wide.
figures/mermaid-%.pdf: $(MERMAID_SRC).%.pdf
	$(PYTHON) -c "import sys, pypdf; r = pypdf.PdfReader(sys.argv[1]); w = pypdf.PdfWriter(); \
	p = r.pages[0]; p.scale_by(180 / 25.4 * 72 / float(p.mediabox.width)); w.add_page(p); \
	w.write(sys.argv[2])" $< $@

diagrams: $(MERMAID_FIGS)

tables:
	cd $(REPO) && $(PYTHON) experiments/043-expression-proteome-morphology/scripts/training_design_043_tables.py

publish: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC)

publish-dry: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC) --dry-run
