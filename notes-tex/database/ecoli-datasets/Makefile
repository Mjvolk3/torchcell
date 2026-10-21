# notes-tex/database/ecoli-datasets/Makefile
# [[experiments.database.ecoli-datasets]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/database/ecoli-datasets/Makefile
#
# No ZOTERO_COLLECTION: the document cites no paper by key (every dataset is named by
# its loader class, whose module and Dendron note carry the citation), so there is no
# collection to export and `make bib` is not meant to run here. references.bib is an
# empty placeholder because the shared build rules list it as a prerequisite. Same
# arrangement as notes-tex/database/database-expansion-bacteria.
#
# Regenerate the tables before rebuilding; never hand-edit anything under tables/.
#   python experiments/database/scripts/build_bacteria_datasets_table.py --organisms ecoli
#   python experiments/database/scripts/build_ecoli_genslm_benchmark_table.py

PYTHON ?= /home/michaelvolk/miniconda3/envs/torchcell/bin/python

include ../../common/Makefile.common
