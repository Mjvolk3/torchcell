# notes-tex/024-perturb-seq-costing/Makefile
# [[experiments.024-perturb-seq-costing.method-review-and-costing]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/024-perturb-seq-costing/Makefile

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

# Where `make plots` looks for the matplotlib SVGs. Since 2026.08.25 the document
# directory is named for its experiment, so this now equals the default
# (`$(notdir $(CURDIR))`). It stays written out rather than inferred: the coupling
# between this document and experiments/024-perturb-seq-costing/ is the thing that
# would break silently if either were ever renamed again, and an explicit
# assignment fails loudly instead of quietly resolving to the wrong directory.
PLOT_SLUG := 024-perturb-seq-costing

PYTHON ?= /Users/michaelvolk/miniconda3/bin/python

include ../common/Makefile.common
