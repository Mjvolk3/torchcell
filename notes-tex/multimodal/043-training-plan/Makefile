# notes-tex/multimodal/043-training-plan/Makefile
# [[experiments.043-expression-proteome-morphology]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/multimodal/043-training-plan/Makefile
#
# `make` builds 043-training-plan.pdf; `make check` gates it; `make publish` uploads the
# build to Zotero under torchcell/notes-tex/multimodal/043-training-plan (our own PDF, the
# one sanctioned write).
#
# Companion of notes-tex/multimodal/043-training-design/ (notation, model, objective,
# overnight results) and notes-tex/multimodal/043-datasets-primer/ (datasets, decisions).
#
# No Zotero collection: the document cites nothing through natbib.

include ../../common/Makefile.common

.PHONY: publish publish-dry

publish: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC)

publish-dry: $(DOC).pdf
	cd $(REPO) && $(PYTHON) notes-tex/common/zotero_publish.py notes-tex/multimodal/$(DOC) --dry-run
