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
# No Zotero collection: the document cites nothing through natbib. Papers are named by
# citation key and DOI in prose.

PLOT_SLUG := 043-expression-proteome-morphology

include ../../common/Makefile.common

.PHONY: tables publish publish-dry

tables:
	cd $(REPO) && $(PYTHON) experiments/043-expression-proteome-morphology/scripts/primer_043_tables.py

publish: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC)

publish-dry: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC) --dry-run
