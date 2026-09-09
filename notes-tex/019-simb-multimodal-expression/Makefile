# notes-tex/019-simb-multimodal-expression/Makefile
# [[experiments.019-simb-multimodal.expression-strand-retrospective]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/019-simb-multimodal-expression/Makefile

DOC := 019-simb-multimodal-expression

# No Zotero collection: the document's subject is our own runs, and the outside
# literature enters only through the SIMB retrospective it continues, which names
# every work inline with its DOI. `make bib` is therefore not available here;
# `make check` still runs, failing only on \cite keys, of which there are none.
ZOTERO_COLLECTION          :=
ZOTERO_PERSONAL_COLLECTION :=

# Every figure here is written by a script under experiments/019-simb-multimodal/
# to the 019-simb-multimodal image directory under a STABLE name (no timestamp),
# because a timestamped filename cannot be the target of the `plots` rule.
PLOT_SLUG := 019-simb-multimodal

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common
