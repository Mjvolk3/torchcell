# experiments/030-solid-growth-multi/scripts/make_pool_store_030.py
# [[experiments.030-solid-growth-multi.scripts.make_pool_store_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/make_pool_store_030
r"""Copy one arm's pool out of the 030 full build into a pool store that fits in RAM.

WHY. On IGB's compute-5-7 the 030 build is a 955 GB LMDB on GPFS, and the loader workers
of a 2-GPU job spend their time in GPFS's per-file mmap lock (``probe_workers.sh``,
2026-10-10 01:38: 27 of 28 workers in ``gpfsNode_t::mmapLock``, GPUs at 39%): epochs of
the Q arm took 84 min when the node was quiet and 150 to 190 min beside another group's
job. The arm reads 1.12M of the 13.5M records. Written as a pool store
(``torchcell.data.pool_store``: the pool's records under their original keys, zlib-6, about
20x smaller than the 92 KB JSON records) the whole training set is a few GB, which a job
copies to the node's RAM disk (``/dev/shm``, 252 GB on compute-5-7) in seconds and reads
with no GPFS page fault at all. Nothing in the dataset, split or indices changes: the
store's ``STORE.json`` carries the full build's key space, so ``len(dataset)`` and every
record index are the ones the split cache and the normalization constants were made with.

The store directory mirrors the build layout the loader expects:

    <dest>/processed/lmdb/{data.mdb, lock.mdb, STORE.json}
    <dest>/processed/{dataset_name_index, perturbation_count_index, phenotype_label_index,
                      experiment_types, gene_set}.json, label_df.parquet, pre_filter.pt,
                      pre_transform.pt, STAGE_COMPLETE                     copied verbatim
    <dest>/raw/lmdb/                                                        empty stub
    <dest>/data_module_cache/index*_seed_*<pool tag>.json                   this pool's split cache

The ``.lock`` files are not copied (FileLockHelper recreates them). The dest root is given
to the trainer as ``dataset.root_rel=<absolute dest>``; ``osp.join(DATA_ROOT, abs)`` is
the absolute path.

Run on GilaHyper, where the build is on local NVMe (1 ms per random record read), under
``gh_make_pool_store_030.slurm``; then ``REL=<dest rel> bash sync_igb_030_build.sh``
mirrors the store to IGB scratch, and ``STAGE_SHM=1`` in ``igb_mmli_cgt_030.slurm``
stages it onto the node.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/make_pool_store_030.py \\
        --indices subset_S3Q_indices_030.json.gz --suffix s3q
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import shutil
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.data.pool_store import read_indices, write_pool_lmdb

EXPERIMENT = "030-solid-growth-multi"
BUILD_REL = f"data/torchcell/experiments/{EXPERIMENT}/001-multi-build"
PROCESSED_FILES = (
    "dataset_name_index.json",
    "experiment_types.json",
    "gene_set.json",
    "label_df.parquet",
    "perturbation_count_index.json",
    "phenotype_label_index.json",
    "pre_filter.pt",
    "pre_transform.pt",
    "STAGE_COMPLETE",
)


class PoolStoreSummary(BaseModel):
    """Written beside the pool index in results/ so the store is traceable from the repo."""

    indices_file: str
    dest_rel: str
    n_pool: int
    n_records_keyspace: int
    codec: str
    zlib_level: int | None
    source_bytes: int
    store_bytes: int
    compression_ratio: float
    seconds: float
    cache_files: list[str]
    processed_files: list[str]


def subset_tag(indices: list[int]) -> str:
    """The data module's cache tag for this pool (``CellDataModule._subset_tag``)."""
    payload = ",".join(map(str, sorted(indices))).encode()
    return f"_sub{len(indices)}-{hashlib.sha256(payload).hexdigest()[:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--indices", required=True, help="pool index file in results/")
    parser.add_argument("--suffix", required=True, help="dest is <build>-<suffix>")
    parser.add_argument("--codec", default="zlib", choices=["zlib", "none"])
    parser.add_argument("--zlib-level", type=int, default=6)
    args = parser.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, EXPERIMENT, "results")
    src_root = osp.realpath(osp.join(data_root, BUILD_REL))
    dest_rel = f"{BUILD_REL}-{args.suffix}"
    dest_root = osp.join(data_root, dest_rel)
    indices = read_indices(osp.join(results, args.indices))
    print(f"pool {args.indices}: {len(indices):,} records; source {src_root}")
    print(f"dest {dest_root}")
    if osp.exists(osp.join(dest_root, "processed", "lmdb", "data.mdb")):
        raise FileExistsError(f"{dest_root} already holds a store; move it aside first")

    t0 = time.time()
    manifest = write_pool_lmdb(
        osp.join(src_root, "processed", "lmdb"),
        osp.join(dest_root, "processed", "lmdb"),
        indices,
        pool_name=args.indices,
        codec=args.codec,
        zlib_level=args.zlib_level,
    )
    for name in PROCESSED_FILES:
        shutil.copy2(
            osp.join(src_root, "processed", name),
            osp.join(dest_root, "processed", name),
        )
    os.makedirs(osp.join(dest_root, "raw", "lmdb"), exist_ok=True)
    tag = subset_tag(indices)
    cache_src = osp.join(src_root, "data_module_cache")
    cache_dst = osp.join(dest_root, "data_module_cache")
    os.makedirs(cache_dst, exist_ok=True)
    cache_files = sorted(f for f in os.listdir(cache_src) if tag in f)
    if not cache_files:
        raise FileNotFoundError(f"no split cache for pool tag {tag} in {cache_src}")
    for name in cache_files:
        shutil.copy2(osp.join(cache_src, name), osp.join(cache_dst, name))
    seconds = time.time() - t0
    summary = PoolStoreSummary(
        indices_file=args.indices,
        dest_rel=dest_rel,
        n_pool=manifest.n_pool,
        n_records_keyspace=manifest.n_records_keyspace,
        codec=manifest.codec,
        zlib_level=manifest.zlib_level,
        source_bytes=manifest.source_bytes,
        store_bytes=manifest.store_bytes,
        compression_ratio=manifest.source_bytes / manifest.store_bytes,
        seconds=seconds,
        cache_files=cache_files,
        processed_files=list(PROCESSED_FILES),
    )
    out = osp.join(results, f"pool_store_030_{args.suffix}_summary.json")
    with open(out, "w") as f:
        f.write(summary.model_dump_json(indent=2))
    print(json.dumps(summary.model_dump(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
