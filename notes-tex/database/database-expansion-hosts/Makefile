# notes-tex/database/database-expansion-hosts/Makefile
# [[experiments.database.expansion-hosts]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/database/database-expansion-hosts/Makefile
#
# No ZOTERO_COLLECTION: every source is carried as a resolvable link in
# tables/sources.tex rather than a \cite, so there is no collection to export and
# `make bib` is not meant to run here. references.bib is an empty placeholder
# because the shared build rules list it as a prerequisite. Same arrangement as
# notes-tex/database/database-expansion-200 and database-expansion-bacteria.
#
# Regenerate the tables before rebuilding; never hand-edit anything under tables/:
#   python experiments/database/scripts/build_hosts_candidate_datasets_table.py
#   python experiments/database/scripts/audit_provenance_mirror.py   # writes tables/mirror.tex
# The two previous passes, database-expansion-100 and database-expansion-200, are
# frozen: their tables are the record of the rankings that chose what KG 4.0 added,
# and neither script writes here.

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../../common/Makefile.common
