# notes-tex/database/pputida-cell-state/Makefile
# [[experiments.database.pputida-cell-state]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/database/pputida-cell-state/Makefile
#
# No ZOTERO_COLLECTION: every source is named inline by author and year with a DOI
# link rather than a \cite, so there is no collection to export and `make bib` is not
# meant to run here. references.bib is an empty placeholder because the shared build
# rules list it as a prerequisite. Same arrangement as
# notes-tex/database/database-expansion-bacteria.
#
# Regenerate the tables before rebuilding; never hand-edit anything under tables/.
# The probe reads 2.7 million records off the served stores, so it is cached and the
# render is the default:
#   python experiments/database/scripts/build_pputida_cell_state_table.py --probe
#   python experiments/database/scripts/build_pputida_cell_state_table.py

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../../common/Makefile.common
