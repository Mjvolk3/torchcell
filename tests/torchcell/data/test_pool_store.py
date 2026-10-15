# tests/torchcell/data/test_pool_store.py
"""A pool store copies a pool's records under their original keys, compressed, and the
dataset reads them through its manifest with the full build's key space.
"""

from __future__ import annotations

import json
import os.path as osp
import zlib
from pathlib import Path

import lmdb
import pytest

from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.pool_store import (
    MANIFEST_NAME,
    decode_value,
    load_manifest,
    pool_sha256,
    write_pool_lmdb,
)


def _full_build(path: Path, n: int = 12) -> list[bytes]:
    """A tiny full build: keys 0..n-1, JSON-list values of growing size."""
    env = lmdb.open(str(path), map_size=64 << 20)
    values = []
    with env.begin(write=True) as txn:
        for i in range(n):
            value = json.dumps(
                [{"experiment": {"idx": i, "pad": "x" * (100 * i)}}]
            ).encode()
            txn.put(str(i).encode(), value)
            values.append(value)
    env.close()
    return values


def _dataset_over(lmdb_dir: Path) -> Neo4jCellDataset:
    """A dataset whose only live attributes are the ones the LMDB readers touch;
    ``__init__`` opens a real build, which the readers do not need.
    """
    dataset = object.__new__(Neo4jCellDataset)
    dataset.__dict__.update({"env": None, "root": str(lmdb_dir.parent.parent)})
    return dataset


def test_pool_store_round_trips_and_keeps_the_keyspace(tmp_path: Path) -> None:
    src = tmp_path / "full" / "processed" / "lmdb"
    src.parent.mkdir(parents=True)
    values = _full_build(src)
    pool = [1, 5, 10]
    dst = tmp_path / "pool" / "processed" / "lmdb"
    manifest = write_pool_lmdb(
        str(src),
        str(dst),
        pool,
        pool_name="toy",
        codec="zlib",
        zlib_level=6,
        map_size=64 << 20,
        commit_every=2,
        verify_sample=3,
    )
    assert manifest.n_pool == 3
    assert manifest.n_records_keyspace == 12
    assert manifest.codec == "zlib"
    assert manifest.pool_sha256 == pool_sha256(pool)
    assert manifest.source_bytes == sum(len(values[i]) for i in pool)
    assert osp.exists(dst / MANIFEST_NAME)
    assert load_manifest(str(dst)) == manifest
    assert load_manifest(str(src)) is None

    env = lmdb.open(str(dst), readonly=True, lock=False)
    with env.begin() as txn:
        assert txn.stat()["entries"] == 3
        stored = txn.get(b"10")
        assert stored is not None and stored != values[10]
        assert zlib.decompress(stored) == values[10]
        assert decode_value(stored, "zlib") == values[10]
        assert txn.get(b"2") is None
    env.close()


def test_dataset_reads_a_pool_store_through_its_manifest(tmp_path: Path) -> None:
    src = tmp_path / "full" / "processed" / "lmdb"
    src.parent.mkdir(parents=True)
    values = _full_build(src)
    dst = tmp_path / "pool" / "processed" / "lmdb"
    write_pool_lmdb(
        str(src), str(dst), [3, 7], pool_name="toy", map_size=64 << 20, verify_sample=2
    )
    pool_ds = _dataset_over(dst)
    # The key space is the full build's, so PyG's range(len) indexing still reaches key 7.
    assert pool_ds.len() == 12
    pool_ds._init_lmdb_read()
    assert pool_ds._read_from_lmdb(7) == values[7]
    assert pool_ds._read_from_lmdb(3) == values[3]
    assert pool_ds._read_from_lmdb(4) is None  # not in the pool
    assert pool_ds._deserialize_json(values[7])[0]["experiment"]["idx"] == 7
    pool_ds.close_lmdb()

    full_ds = _dataset_over(src)
    assert full_ds.len() == 12
    full_ds._init_lmdb_read()
    assert full_ds._pool_store is None
    assert full_ds._read_from_lmdb(4) == values[4]
    full_ds.close_lmdb()


def test_write_refuses_a_missing_record_and_duplicate_indices(tmp_path: Path) -> None:
    src = tmp_path / "full" / "processed" / "lmdb"
    src.parent.mkdir(parents=True)
    _full_build(src, n=4)
    with pytest.raises(KeyError):
        write_pool_lmdb(
            str(src), str(tmp_path / "a"), [0, 99], pool_name="toy", map_size=64 << 20
        )
    with pytest.raises(ValueError):
        write_pool_lmdb(
            str(src), str(tmp_path / "b"), [0, 0], pool_name="toy", map_size=64 << 20
        )
