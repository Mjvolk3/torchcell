# notes-tex/microbe-perturb-seq/Makefile
# [[microbe-perturb-seq.method-review-and-costing]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/microbe-perturb-seq/Makefile

DOC := main

# Zotero group `torchcell`, collection `microbe-perturb-seq`. This collection is
# the single source of truth for what this document may cite: `make check` fails
# on any \cite key that is not in the exported bib, which is the mechanical form
# of "add the paper to the library before citing it".
# Group `microbe-perturb-seq` (curated, shared) and the personal collection of the
# same name. The bib is their UNION -- see notes-tex/README.md for why the two
# tiers differ. Keep the assignments comment-free: a trailing `# comment` leaves
# the whitespace before it inside the variable and corrupts the export URL.
ZOTERO_COLLECTION          := FE8DQKUH
ZOTERO_PERSONAL_COLLECTION := AC8MFJXK

# Where `make plots` looks for the matplotlib SVGs. The document is named for its
# subject; the plots are named for the EXPERIMENT that generates them, and the two
# do not have to match, so it is stated rather than inferred from the directory.
PLOT_SLUG := 019-perturb-seq-costing

PYTHON ?= /Users/michaelvolk/miniconda3/bin/python

include ../common/Makefile.common
