# torchcell/lmdb_map_size.py
# [[torchcell.lmdb_map_size]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/lmdb_map_size.py
"""The address-space reservation every build-stage LMDB is opened with for writing.

LMDB's ``map_size`` is the ceiling on how large the data file may grow, not an
allocation: the file is sparse and takes disk only as pages are written, so a large
value costs nothing. A small one is fatal at the worst moment. Every stage of the
query build (raw, conversion, deduplication, aggregation, the processed copy) opened
its store at 1e12 bytes, and the 030 raw store, at about 23 KB per record, reached
that ceiling 40.6 million records into a 43.8 million-record query after 18 hours
(slurm 2687, MDB_MAP_FULL). One constant, sized for a 43.8 million-record build to
grow eightfold, so no stage can be the one that hits the wall.

Read-only opens do not need it: LMDB takes the size recorded in the store's meta page
when a reader asks for less.
"""

BUILD_LMDB_MAP_SIZE = int(8e12)
