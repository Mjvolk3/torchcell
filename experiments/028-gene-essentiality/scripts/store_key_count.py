# experiments/028-gene-essentiality/scripts/store_key_count.py
# [[experiments.028-gene-essentiality.scripts.store_key_count]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/store_key_count

"""How many solves does the store need over the solid-growth build?

The scalable module form solves the growth program once per distinct (genotype,
medium) pair, offline, and reads the stored vector through the affine (rounds 27 to 57).
A genotype's row is fixed by the GEM genes it deletes and by the medium: a deletion of
a gene with no reaction in yeast-GEM leaves the program unchanged, so its row is the
wild-type row, and two genotypes that delete the same GEM genes share a row. The count
that sizes the store is therefore the number of distinct (GEM gene subset, medium) keys
across the records, which is not the number of records and not the number of genotypes.

This reads every record of the 025 solid-growth build (13,525,071 aggregated records,
517 GB of LMDB) with the LMDB cursor over contiguous key ranges in parallel, extracts
the perturbed systematic gene names and the medium names by regular expression rather
than parsing the 39 KB records (the genotype and environment fields are the only ones
read, and both are single-line JSON strings), and counts:

* records, records by perturbation count, distinct genotypes (as gene sets);
* distinct media names;
* distinct GEM gene subsets by size, and the records whose subset is empty (the
  wild-type row);
* distinct (GEM gene subset, medium) keys, the store's row count;

and reports the store's size at 4,131 float32 columns per row and the CPU solve time at
the measured 0.15 s (interior point, dual simplex) and 1 s (pFBA) per program.
Writes ``results/store_key_count.json``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import os.path as osp
import re
import time
from collections import Counter
from typing import Any

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
LMDB_PATH = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build/processed/lmdb"
)
GENE_RE = re.compile(rb'"systematic_gene_name":\s*"([^"]+)"')
MEDIA_RE = re.compile(rb'"media":\s*\{"name":\s*"([^"]+)"')
N_REACTIONS = 4131
SECONDS_PER_SOLVE = {"interior_point_or_dual_simplex": 0.15, "pfba": 1.0}


class KeyCount(BaseModel):
    n_records: int
    records_by_perturbation_count: dict[str, int]
    n_distinct_genotypes: int
    n_distinct_media: int
    media: dict[str, int]
    n_records_empty_gem_subset: int
    n_distinct_gem_subsets: int
    gem_subsets_by_size: dict[str, int]
    n_distinct_keys: int
    n_gem_genes: int
    store_float32_gb: float
    store_float16_gb: float
    cpu_hours_per_core: dict[str, float]
    seconds: float


def _scan(args: tuple[int, int, frozenset[str]]) -> dict[str, Any]:
    start, stop, gem_genes = args
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, max_readers=64, readahead=True)
    n = 0
    by_count: Counter[int] = Counter()
    genotypes: set[int] = set()
    media: Counter[bytes] = Counter()
    empty = 0
    subsets: set[frozenset[str]] = set()
    keys: Counter[tuple[frozenset[str], bytes]] = Counter()
    with env.begin(buffers=True) as txn:
        cur = txn.cursor()
        for i in range(start, stop):
            v = cur.get(str(i).encode())
            if v is None:
                continue
            b = bytes(v)
            genes = frozenset(g.decode() for g in GENE_RE.findall(b))
            ms = set(MEDIA_RE.findall(b))
            n += 1
            by_count[len(genes)] += 1
            genotypes.add(hash(tuple(sorted(genes))))
            gem = frozenset(g for g in genes if g in gem_genes)
            if not gem:
                empty += 1
            subsets.add(gem)
            for m in ms:
                media[m] += 1
                keys[(gem, m)] += 1
    env.close()
    return {
        "n": n,
        "by_count": dict(by_count),
        "genotypes": genotypes,
        "media": media,
        "empty": empty,
        "subsets": subsets,
        "keys": keys,
    }


def main() -> None:
    from torchcell.metabolism.yeast_GEM import YeastGEM

    started = time.time()
    gem_genes = frozenset(g.id for g in YeastGEM().model.genes)
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, max_readers=64)
    with env.begin() as txn:
        n_entries = int(txn.stat()["entries"])
    env.close()
    workers = int(os.environ.get("TC_SCAN_WORKERS", "20"))
    step = -(-n_entries // workers)
    jobs = [(s, min(s + step, n_entries), gem_genes) for s in range(0, n_entries, step)]
    print(f"{n_entries} entries, {len(gem_genes)} GEM genes, {len(jobs)} ranges of {step}")
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_scan, jobs)
    print(f"scanned in {time.time() - started:.0f}s")

    n = sum(p["n"] for p in parts)
    by_count: Counter[int] = Counter()
    media: Counter[bytes] = Counter()
    genotypes: set[int] = set()
    subsets: set[frozenset[str]] = set()
    keys: Counter[tuple[frozenset[str], bytes]] = Counter()
    empty = 0
    for p in parts:
        by_count.update(p["by_count"])
        media.update(p["media"])
        genotypes |= p["genotypes"]
        subsets |= p["subsets"]
        keys.update(p["keys"])
        empty += p["empty"]
    # The keys themselves, with the number of records each serves, so the store can be
    # built in order of records served and a partial store covers the most records.
    import gzip

    keys_path = osp.join(RESULTS_DIR, "store_keys.json.gz")
    with gzip.open(keys_path, "wt") as fh:
        json.dump(
            [
                {"gem_genes": sorted(g), "medium": m.decode(), "n_records": c}
                for (g, m), c in keys.most_common()
            ],
            fh,
        )
    print(f"{len(keys)} keys written to {keys_path}")
    by_size: Counter[int] = Counter(len(s) for s in subsets)
    rows = len(keys)
    out = KeyCount(
        n_records=n,
        records_by_perturbation_count={str(k): v for k, v in sorted(by_count.items())},
        n_distinct_genotypes=len(genotypes),
        n_distinct_media=len(media),
        media={m.decode(): c for m, c in media.most_common()},
        n_records_empty_gem_subset=empty,
        n_distinct_gem_subsets=len(subsets),
        gem_subsets_by_size={str(k): v for k, v in sorted(by_size.items())},
        n_distinct_keys=rows,
        n_gem_genes=len(gem_genes),
        store_float32_gb=rows * N_REACTIONS * 4 / 1e9,
        store_float16_gb=rows * N_REACTIONS * 2 / 1e9,
        cpu_hours_per_core={k: rows * s / 3600 for k, s in SECONDS_PER_SOLVE.items()},
        seconds=time.time() - started,
    )
    path = osp.join(RESULTS_DIR, "store_key_count.json")
    with open(path, "w") as fh:
        json.dump(out.model_dump(), fh, indent=2)
    print(out.model_dump_json(indent=2))
    print(path)


if __name__ == "__main__":
    main()
