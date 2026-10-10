# torchcell/data/pool_store.py
# [[torchcell.data.pool_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/pool_store
# Test file: tests/torchcell/data/test_pool_store.py
"""A pool store: the records of one training pool, copied out of a full build.

A full build's LMDB holds every record the knowledge graph returned (13.5M records,
955 GB on the 030 build), and a training arm reads a pool of about a million of them.
On a parallel filesystem that is the expensive shape: the loader workers fault random
pages of one huge memory-mapped file, and on IGB's GPFS every fault takes the file's
mmap lock, so 28 workers serialize on it and an epoch runs at a third of its uncontended
pace. A pool store copies the pool's records, under their ORIGINAL keys, into a small
LMDB that fits in a compute node's RAM disk (zlib-6 compresses the JSON records about
20x, so the 030 S3Q pool is a few GB), and the dataset reads it with no other change.

The store is a directory with the full build's layout, so ``Neo4jCellDataset`` opens it as
a root:

    <store>/processed/lmdb/{data.mdb, lock.mdb, STORE.json}   the pool's records
    <store>/processed/{*.json, label_df.parquet, pre_*.pt, STAGE_COMPLETE}  copied
    <store>/raw/lmdb/                                          an empty stub
    <store>/data_module_cache/                                 the split cache of the pool

``STORE.json`` (``PoolStoreManifest``) is the contract the reader needs: the key space of
the source build (so ``len(dataset)`` keeps reporting the full build's record count, which
is what PyG indexes against) and the codec of the stored values. A store with no
``STORE.json`` is a plain full build, read exactly as before.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import os.path as osp
import time
import zlib
from collections.abc import Iterable, Sequence
from typing import Literal

import lmdb
from pydantic import BaseModel

__all__ = [
    "MANIFEST_NAME",
    "PoolStoreManifest",
    "decode_value",
    "load_manifest",
    "pool_sha256",
    "write_pool_lmdb",
]

MANIFEST_NAME = "STORE.json"
Codec = Literal["none", "zlib"]


class PoolStoreManifest(BaseModel):
    """What a reader must know about a pool store, kept beside its ``data.mdb``."""

    n_records_keyspace: int
    """Record count of the SOURCE build; the keys in this store lie in ``range(n)``."""
    codec: Codec
    zlib_level: int | None = None
    source_store: str
    pool_name: str
    n_pool: int
    pool_sha256: str
    source_bytes: int
    store_bytes: int
    created: str


def pool_sha256(indices: Iterable[int]) -> str:
    """Hash of the sorted pool, so a store can be matched to the index file it came from."""
    payload = ",".join(map(str, sorted(indices))).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(lmdb_dir: str) -> PoolStoreManifest | None:
    """The manifest of a pool store, or ``None`` for a plain full build."""
    path = osp.join(lmdb_dir, MANIFEST_NAME)
    if not osp.exists(path):
        return None
    with open(path) as f:
        return PoolStoreManifest.model_validate_json(f.read())


def decode_value(value: bytes, codec: Codec) -> bytes:
    """The stored bytes back to the record's serialized form."""
    if codec == "zlib":
        return zlib.decompress(value)
    return value


def encode_value(value: bytes, codec: Codec, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(value, level)
    return value


def read_indices(path: str) -> list[int]:
    """A pool index file (``.json`` or ``.json.gz``) as a list of record indices."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [int(i) for i in json.load(f)]


def write_pool_lmdb(
    source_lmdb_dir: str,
    dest_lmdb_dir: str,
    indices: Sequence[int],
    pool_name: str,
    codec: Codec = "zlib",
    zlib_level: int = 6,
    map_size: int = 256 << 30,
    commit_every: int = 20_000,
    verify_sample: int = 2_000,
) -> PoolStoreManifest:
    """Copy ``indices`` from the source LMDB into a new LMDB at ``dest_lmdb_dir``.

    Keys are copied verbatim (the decimal record index as bytes); values are encoded
    with ``codec``. Keys are written in LMDB's byte order with ``append=True``, which is
    the fast path. Afterwards the entry count is checked against the pool and a random
    sample of records is read back and compared to the source byte for byte. Returns
    the manifest it wrote to ``dest_lmdb_dir/STORE.json``.
    """
    os.makedirs(dest_lmdb_dir, exist_ok=True)
    keys = sorted({str(i).encode() for i in indices})
    if len(keys) != len(indices):
        raise ValueError("pool indices must be unique")
    src = lmdb.open(
        source_lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False
    )
    dst = lmdb.open(dest_lmdb_dir, map_size=map_size, subdir=True, lock=True)
    t0 = time.time()
    source_bytes = 0
    with src.begin() as rtxn:
        keyspace = int(rtxn.stat()["entries"])
        wtxn = dst.begin(write=True)
        for n, key in enumerate(keys, start=1):
            value = rtxn.get(key)
            if value is None:
                raise KeyError(f"record {key.decode()} is not in {source_lmdb_dir}")
            source_bytes += len(value)
            wtxn.put(key, encode_value(value, codec, zlib_level), append=True)
            if n % commit_every == 0:
                wtxn.commit()
                wtxn = dst.begin(write=True)
                if n % (commit_every * 10) == 0:
                    rate = n / (time.time() - t0)
                    print(
                        f"  {n:,}/{len(keys):,} records, {source_bytes / 1e9:.1f} GB read, "
                        f"{rate:.0f} rec/s",
                        flush=True,
                    )
        wtxn.commit()
    dst.sync()
    with dst.begin() as txn:
        n_entries = int(txn.stat()["entries"])
    if n_entries != len(keys):
        raise RuntimeError(f"wrote {n_entries} entries for a pool of {len(keys)}")
    # Read-back check against the source, on a deterministic sample.
    step = max(1, len(keys) // verify_sample)
    with src.begin() as rtxn, dst.begin() as wtxn:
        for key in keys[::step]:
            stored = wtxn.get(key)
            if stored is None or decode_value(stored, codec) != rtxn.get(key):
                raise RuntimeError(f"read-back mismatch at key {key.decode()}")
    src.close()
    dst.close()
    store_bytes = osp.getsize(osp.join(dest_lmdb_dir, "data.mdb"))
    manifest = PoolStoreManifest(
        n_records_keyspace=keyspace,
        codec=codec,
        zlib_level=zlib_level if codec == "zlib" else None,
        source_store=osp.abspath(source_lmdb_dir),
        pool_name=pool_name,
        n_pool=len(keys),
        pool_sha256=pool_sha256(indices),
        source_bytes=source_bytes,
        store_bytes=store_bytes,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(osp.join(dest_lmdb_dir, MANIFEST_NAME), "w") as f:
        f.write(manifest.model_dump_json(indent=2))
    return manifest
