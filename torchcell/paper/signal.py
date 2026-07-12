# torchcell/paper/signal.py
# [[torchcell.paper.signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/paper/signal.py
# Test file: tests/torchcell/paper/test_tables.py
"""CLI to compute a dataset's gzip 'signal' + derived shape/graph-role from its
built LMDB. Use as new datasets are added, without touching the paper table.

The gzip signal is a Kolmogorov-complexity proxy (size of the concatenated
serialized phenotypes); shape/role are read from the phenotype record. All the
work lives in ``torchcell.paper.tables`` -- this is just a command-line front.

Run from the repo root (``$DATA_ROOT`` from .env):
  python -m torchcell.paper.signal data/torchcell/amino_acid_mulleder2016
  python -m torchcell.paper.signal /abs/path/to/dataset_root   # explicit
  python -m torchcell.paper.signal <subpath> --lmdb            # arg already ends at processed/lmdb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from torchcell.paper.tables import (
    human_bytes,
    phenotype_descriptor,
    read_first_record,
    scientific,
    stream_gzip_signal,
)


def resolve_lmdb(arg: str, data_root: str, is_lmdb: bool) -> Path:
    """Resolve a CLI path to a ``processed/lmdb`` dir.

    Accepts an absolute path or a ``$DATA_ROOT``-relative dataset subpath;
    appends ``processed/lmdb`` unless ``--lmdb`` says the arg already points
    there.
    """
    base = Path(arg) if os.path.isabs(arg) else Path(data_root) / arg
    lmdb_dir = base if is_lmdb else base / "processed" / "lmdb"
    if not (lmdb_dir / "data.mdb").exists():
        raise FileNotFoundError(f"No data.mdb under {lmdb_dir}")
    return lmdb_dir


def main() -> None:
    """Compute + print the gzip signal, record count, shape, and graph role."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="Dataset root (abs, or relative to $DATA_ROOT)")
    ap.add_argument(
        "--lmdb", action="store_true", help="Arg already ends at processed/lmdb"
    )
    args = ap.parse_args()

    load_dotenv()
    lmdb_dir = resolve_lmdb(args.dataset, os.environ["DATA_ROOT"], args.lmdb)

    shape, role = phenotype_descriptor(read_first_record(lmdb_dir))
    n, nbytes = stream_gzip_signal(lmdb_dir, label=args.dataset)

    print(f"lmdb        {lmdb_dir}")
    print(f"records     {n:,}")
    print(f"shape       {shape}")
    print(f"graph role  {role}")
    print(
        f"signal      {human_bytes(nbytes)}  ({scientific(nbytes).md} bytes, {nbytes:,})"
    )


if __name__ == "__main__":
    main()
