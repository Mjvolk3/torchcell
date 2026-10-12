# scripts/served_count_drift.py
# [[scripts.served_count_drift]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/served_count_drift
"""Compare the served release's per-dataset record counts with today's dev stores.

The data is the data: a revised loader of a served dataset must not lose a record
unless the authors withdrew it. This script reads the newest committed served-counts
table (``docs/source/datasets/_generated/served_counts.md``) and the dev-tree LMDB
entry count of every dataset it lists, and prints one row per dataset with the delta.
Exit 1 when any dataset DECREASED, so it can gate a landing or a preflight.

Read-only: it opens each LMDB with ``readonly=True, lock=False`` and reads ``stat``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel


class CountRow(BaseModel):
    dataset: str
    served: int
    dev: int | None
    note: str

    @property
    def delta(self) -> int | None:
        return None if self.dev is None else self.dev - self.served


def read_served_counts(path: str) -> dict[str, int]:
    served: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\| `(\w+)` \| ([\d,]+) \|", line)
            if m:
                served[m.group(1)] = int(m.group(2).replace(",", ""))
    return served


def dev_count(root: str) -> int | None:
    lmdb_dir = os.path.join(root, "processed", "lmdb")
    if not os.path.isdir(lmdb_dir):
        return None
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, subdir=True, max_dbs=0)
    try:
        with env.begin() as txn:
            return int(txn.stat()["entries"])
    finally:
        env.close()


def collect(served_md: str, data_root: str) -> list[CountRow]:
    from torchcell.database.build_dataset_lmdb import dataset_default_root
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    by_name = {cls.__name__: cls for cls in dataset_adapter_map}
    rows: list[CountRow] = []
    for name, served in read_served_counts(served_md).items():
        cls = by_name.get(name)
        if cls is None:
            rows.append(CountRow(dataset=name, served=served, dev=None, note="not mapped"))
            continue
        root = dataset_default_root(cls)
        root = root if os.path.isabs(root) else os.path.join(data_root, root)
        dev = dev_count(root)
        if dev is None:
            rows.append(CountRow(dataset=name, served=served, dev=None, note="no dev store"))
            continue
        manifest = os.path.exists(os.path.join(root, "build_manifest.json"))
        rows.append(
            CountRow(
                dataset=name,
                served=served,
                dev=dev,
                note="manifest" if manifest else "no manifest (retired or rebuilding)",
            )
        )
    rows.sort(key=lambda r: (r.dev is None, -abs(r.delta or 0)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--served-md",
        default="docs/source/datasets/_generated/served_counts.md",
        help="the committed served-counts table",
    )
    parser.add_argument("--data-root", default=None, help="defaults to $DATA_ROOT")
    args = parser.parse_args()
    load_dotenv()
    data_root = args.data_root or os.environ["DATA_ROOT"]
    rows = collect(args.served_md, data_root)
    print(f"{'dataset':40s} {'served':>11s} {'dev':>11s} {'delta':>9s}  note")
    decreased = 0
    for r in rows:
        dev = "" if r.dev is None else f"{r.dev:,}"
        delta = "" if r.delta is None else f"{r.delta:+,}"
        print(f"{r.dataset:40s} {r.served:>11,} {dev:>11s} {delta:>9s}  {r.note}")
        if r.delta is not None and r.delta < 0:
            decreased += 1
    unchanged = sum(1 for r in rows if r.delta == 0)
    print(f"\nunchanged {unchanged} of {len(rows)}; decreased {decreased}")
    return 1 if decreased else 0


if __name__ == "__main__":
    sys.exit(main())
