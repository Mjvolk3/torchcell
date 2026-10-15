# experiments/database/scripts/content_id_collision_scan.py
# [[experiments.database.scripts.content_id_collision_scan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/content_id_collision_scan
r"""Count the records of every dev store whose Experiment content id collides within it.

Lane Q of the ontology unification (2026-10-15). The adapter's ``Experiment`` node id is
``sha256(json.dumps(experiment.model_dump()))`` and ``FastCsvSink`` deduplicates node
ids, so two stored records with byte-identical experiment dumps become ONE node and the
second measurement is silently absent from the graph. Lane P found 31 such pairs in
Ozaydin 2013 (``prebuild_verify_stores.py``); this script asks the same question of
every store under ``$DATA_ROOT/data/torchcell`` (superseded and deprecated copies
skipped) and writes, per store, the record count, the distinct count, the number of
collision groups, their size histogram and up to five example groups (store indices
plus the gene names and phenotype label of the shared content).

Records are read raw (the shared tree was written by ``main`` under the pre-unification
schema, so a retired enum class unpickles as its plain value through the tolerant
unpickler, which leaves ``json.dumps`` of the dump unchanged) and interned pointers are
resolved before hashing, as the adapter does.

Usage (from the repo root, the torchcell environment)::

    python experiments/database/scripts/content_id_collision_scan.py \\
        --data-root /scratch/projects/torchcell-scratch \\
        --out experiments/database/results/content_id_collision_scan.json \\
        [--workers 4] [--only slug ...] [--skip-larger-than-gb 40]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import os.path as osp
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import lmdb

from torchcell.data.experiment_dataset import InternedDict, resolve_interned


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle a record whose enum classes the current schema has retired."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return lambda value: value


def _loads(raw: bytes) -> Any:
    return _TolerantUnpickler(io.BytesIO(raw)).load()


def _load_interned(processed_dir: str) -> dict[str, Any]:
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if not osp.isdir(interned_dir):
        return interned
    env = lmdb.open(interned_dir, readonly=True, lock=False, max_readers=256)
    with env.begin() as txn:
        for key, value in txn.cursor():
            ref = key.decode()
            loaded = _loads(value)
            interned[ref] = (
                InternedDict(loaded, ref) if isinstance(loaded, dict) else loaded
            )
    env.close()
    return interned


def _describe(experiment: dict[str, Any]) -> dict[str, Any]:
    genotype = experiment.get("genotype")
    perturbations: list[dict[str, Any]] = []
    if isinstance(genotype, dict):
        perturbations = genotype.get("perturbations") or []
    elif isinstance(genotype, list):
        for g in genotype:
            perturbations.extend(g.get("perturbations") or [])
    phenotype = experiment.get("phenotype") or {}
    label = phenotype.get("label_name")
    return {
        "experiment_type": experiment.get("experiment_type"),
        "genes": [p.get("systematic_gene_name") for p in perturbations][:8],
        "n_perturbations": len(perturbations),
        "label_name": label,
        "value": phenotype.get(label) if isinstance(label, str) else None,
    }


def scan_store(slug: str, processed_dir: str) -> dict[str, Any]:
    """Hash every record's experiment dump; return the collision accounting."""
    started = time.time()
    interned = _load_interned(processed_dir)
    env = lmdb.open(
        osp.join(processed_dir, "lmdb"), readonly=True, lock=False, max_readers=256
    )
    first_index: dict[bytes, int] = {}
    groups: dict[bytes, list[int]] = defaultdict(list)
    n = 0
    with env.begin() as txn:
        for key, raw in txn.cursor():
            index = int(key.decode())
            record = resolve_interned(_loads(raw), interned)
            digest = hashlib.sha256(
                json.dumps(record["experiment"]).encode("utf-8")
            ).digest()
            n += 1
            earlier = first_index.get(digest)
            if earlier is None:
                first_index[digest] = index
                continue
            if not groups[digest]:
                groups[digest].append(earlier)
            groups[digest].append(index)
    examples = []
    with env.begin() as txn:
        for digest, indices in list(groups.items())[:5]:
            raw = txn.get(str(indices[0]).encode())
            record = resolve_interned(_loads(raw), interned)
            examples.append({"indices": indices[:6], **_describe(record["experiment"])})
    env.close()
    sizes = Counter(len(v) for v in groups.values())
    return {
        "slug": slug,
        "n_records": n,
        "n_distinct": len(first_index),
        "n_collision_groups": len(groups),
        "n_records_lost_in_graph": sum(len(v) - 1 for v in groups.values()),
        "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "examples": examples,
        "wall_s": round(time.time() - started, 1),
    }


def _is_pickle_store(processed_dir: str) -> bool:
    """True when the first value of the records env is a pickle (protocol byte 0x80).

    The ``showcase_*`` directories under the same tree hold JSON lists of query results,
    not loader-built records, and are not stores this question applies to.
    """
    env = lmdb.open(
        osp.join(processed_dir, "lmdb"), readonly=True, lock=False, max_readers=256
    )
    with env.begin() as txn:
        cursor = txn.cursor()
        first = cursor.value()[:1] if cursor.first() else b""
    env.close()
    return first == b"\x80"


def _stores(
    data_root: str, only: list[str], skip_gb: float
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """The stores to scan, and the directories left out with the reason for each."""
    base = osp.join(data_root, "data", "torchcell")
    out = []
    skipped: dict[str, str] = {}
    for slug in sorted(os.listdir(base)):
        if "deprecated" in slug or "superseded" in slug:
            continue
        if only and slug not in only:
            continue
        processed = osp.join(base, slug, "processed")
        data_file = osp.join(processed, "lmdb", "data.mdb")
        if not osp.isfile(data_file):
            continue
        size_gb = osp.getsize(data_file) / 1e9
        if size_gb > skip_gb:
            skipped[slug] = f"{size_gb:.1f} GB, over --skip-larger-than-gb {skip_gb}"
        elif not _is_pickle_store(processed):
            skipped[slug] = "records are not pickles (a query-result dump, not a store)"
        else:
            out.append((slug, processed))
    for slug, why in skipped.items():
        print(f"skip {slug}: {why}", file=sys.stderr)
    return out, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip-larger-than-gb", type=float, default=1e9)
    args = parser.parse_args()
    stores, skipped = _stores(args.data_root, args.only, args.skip_larger_than_gb)
    # largest first so the long tail does not run alone at the end
    stores.sort(key=lambda s: -osp.getsize(osp.join(s[1], "lmdb", "data.mdb")))
    results: dict[str, Any] = {}
    if osp.exists(args.out):
        with open(args.out) as fh:
            results = {r["slug"]: r for r in json.load(fh)["stores"]}
    todo = [s for s in stores if s[0] not in results]
    print(f"{len(stores)} stores, {len(todo)} to scan", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_store, slug, p): slug for slug, p in todo}
        for future in as_completed(futures):
            result = future.result()
            results[result["slug"]] = result
            print(
                f"{result['slug']}: {result['n_records']} records, "
                f"{result['n_collision_groups']} collision groups, "
                f"{result['n_records_lost_in_graph']} lost, {result['wall_s']} s",
                file=sys.stderr,
            )
            with open(args.out, "w") as fh:
                json.dump(
                    {
                        "data_root": args.data_root,
                        "skipped": skipped,
                        "stores": sorted(results.values(), key=lambda r: r["slug"]),
                    },
                    fh,
                    indent=1,
                )


if __name__ == "__main__":
    main()
