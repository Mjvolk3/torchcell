# notes-tex/024-perturb-seq-pilot/Makefile
# [[experiments.024-perturb-seq-costing.pilot-plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/024-perturb-seq-pilot/Makefile

# Same two Zotero collections as 024-perturb-seq-costing, because this document
# is the bench plan that falls out of that review and cites a subset of the same
# works. Group `microbe-perturb-seq` union the personal collection of the same
# name; see notes-tex/README.md for why the two tiers differ.
ZOTERO_COLLECTION          := FE8DQKUH
ZOTERO_PERSONAL_COLLECTION := AC8MFJXK

# Figures come from two experiments, so both image directories are searched and
# the first one holding NAME.svg wins. 024 supplies the costing panel this
# document generates; 028 supplies the three knockout-expression panels behind
# Sec. 3, whose scripts live on branch `multimodal-phenotype-retrospective`
# (PR #265) and whose built PDFs are committed here.
PLOT_SLUGS := 024-perturb-seq-costing 028-knockout-expression

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../common/Makefile.common
