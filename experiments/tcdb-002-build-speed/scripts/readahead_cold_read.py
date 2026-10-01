# experiments/tcdb-002-build-speed/scripts/readahead_cold_read.py
# [[experiments.tcdb-002-build-speed.scripts.readahead_cold_read]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/readahead_cold_read
"""Cold LMDB read throughput with and without readahead, in the build's access order.

Every read the query build keeps after stage 2 walks keys in cursor order: aggregation
pass 2 ``get``s each group's raw keys with groups in cursor order, and the processed
copy is a cursor walk. The slice's own stores sit in the page cache, so this reads a
store that is not: the 030 processed LMDB (889.7 GB, 884 KB resident when checked with
``fincore``), whose values are JSON lists under integer-string keys like the 033
build's. Each (readahead, region) reads ``COUNT`` values by cursor from a key prefix
not read before, copying every value out so every page is touched. Run once: a second
run reads warm pages. Writes ``results/readahead_cold_read.csv``.

    nice -n 10 ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/readahead_cold_read.py
"""

from __future__ import annotations

import csv
import os.path as osp
import time

import lmdb

STORE = "/db/experiments/030-solid-growth-multi-001-multi-build/processed/lmdb"
COUNT = 50_000
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "readahead_cold_read.csv",
)
# Two disjoint key prefixes per setting, alternating so neither setting always reads
# the later (possibly differently laid out) part of the store first.
RUNS = [(False, b"3"), (True, b"4"), (False, b"6"), (True, b"7")]


def cold_read(readahead: bool, prefix: bytes) -> tuple[float, int]:
    """Seconds and bytes to copy out ``COUNT`` values from ``prefix`` by cursor."""
    env = lmdb.open(STORE, readonly=True, lock=False, readahead=readahead)
    total = 0
    t = time.perf_counter()
    with env.begin() as txn:
        cur = txn.cursor()
        cur.set_range(prefix)
        for i, (_k, v) in enumerate(cur):
            total += len(bytes(v))
            if i + 1 >= COUNT:
                break
    wall = time.perf_counter() - t
    env.close()
    return wall, total


def main() -> None:
    """Measure each run and write the CSV."""
    rows = []
    for readahead, prefix in RUNS:
        wall, total = cold_read(readahead, prefix)
        mb_s = total / 1e6 / wall
        rows.append(
            [
                "on" if readahead else "off",
                prefix.decode(),
                COUNT,
                total,
                f"{wall:.2f}",
                f"{mb_s:.0f}",
                f"{wall / COUNT * 1e3:.4f}",
            ]
        )
        print(
            f"readahead {'on ' if readahead else 'off'} prefix {prefix.decode()}: "
            f"{total / 1e9:.2f} GB in {wall:.1f} s = {mb_s:.0f} MB/s, "
            f"{wall / COUNT * 1e3:.4f} ms/value"
        )
    with open(RESULTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "readahead",
                "key_prefix",
                "values",
                "bytes",
                "seconds",
                "mb_per_s",
                "ms_per_value",
            ]
        )
        w.writerows(rows)
    print(RESULTS)


if __name__ == "__main__":
    main()
