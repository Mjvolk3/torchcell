# experiments/043-expression-proteome-morphology/scripts/compare_raw_stores_043.py
# [[experiments.043-expression-proteome-morphology.scripts.compare_raw_stores_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/compare_raw_stores_043
"""Record-level equivalence of two 043 raw stores (single-session vs partitioned).

Owner, 2026-10-23: "if we speed it up we must be confident that we are getting
equivalent query." The reference is ``raw_043_v2`` (``build_043_store.py``, one session,
job 6187); the candidate is ``raw_043_v2_partitioned`` (``build_043_raw_partitioned.py``,
``fetch_workers`` forked workers over ``e.id`` prefix partitions).

The test is ORDER-FREE by design: ``data_<i>`` keys are assigned in write order, and
nothing here assumes the two paths write in the same order. For each store, every record
of ``raw/lmdb`` is hashed as the sha256 of its canonical JSON (``json.loads`` of the
stored value, then ``json.dumps`` with sorted keys and no whitespace), and the two
MULTISETS of hashes are compared: the record count, the per-dataset counts, the set of
hashes, the multiplicity of every hash, and the hashes found on one side only (each must
be zero). The raw stage's other outputs are compared the same way:

* ``raw/gene_set.json``: equal as sets (and as sorted lists).
* ``raw/experiment_reference_index.json``: each entry becomes (sha256 of the canonical
  reference JSON, sorted sha256s of its member records), so the comparison does not
  depend on record indices; the two multisets of entries must be equal.

As information only (not part of the verdict), the report also counts the indices whose
record hash is the same at the same ``data_<i>`` in both stores (the partitioned writer is
designed to write in single-session order).

Writes ``results/compare_raw_stores_043.json`` with ``verdict`` ``identical`` or
``different`` and, when different, the counts and up to 20 example hashes per side with
their dataset and index.

    python experiments/043-expression-proteome-morphology/scripts/compare_raw_stores_043.py --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import socket
import time
from collections import Counter
from multiprocessing import Pool
from typing import Any

import lmdb
from dotenv import load_dotenv

EXPERIMENT = "043-expression-proteome-morphology"
REFERENCE_TAG = "raw_043_v2"
CANDIDATE_TAG = "raw_043_v2_partitioned"
CHUNK = 2000


def canonical_sha256(obj: Any) -> str:
    """sha256 of the canonical JSON of ``obj`` (sorted keys, no whitespace)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_range(task: tuple[str, int, int]) -> list[tuple[int, str, str]]:
    """(index, dataset_name, record sha256) for ``data_<start>`` .. ``data_<end - 1>``."""
    path, start, end = task
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=True)
    out = []
    with env.begin() as txn:
        for i in range(start, end):
            value = txn.get(f"data_{i}".encode())
            if value is None:
                raise KeyError(f"data_{i} missing from {path}")
            record = json.loads(value)
            out.append(
                (i, record["experiment"]["dataset_name"], canonical_sha256(record))
            )
    env.close()
    return out


def hash_store(lmdb_path: str, workers: int) -> list[tuple[str, str]]:
    """Per index, (dataset_name, record sha256) of one raw LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=True)
    n = int(env.stat()["entries"])
    env.close()
    tasks = [(lmdb_path, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    rows: list[tuple[str, str]] = [("", "")] * n
    t0 = time.time()
    with Pool(workers) as pool:
        for k, part in enumerate(pool.imap_unordered(_hash_range, tasks), start=1):
            for i, name, sha in part:
                rows[i] = (name, sha)
            if k % 10 == 0 or k == len(tasks):
                print(f"{lmdb_path}: {k}/{len(tasks)} chunks, {time.time() - t0:.0f} s")
    return rows


def reference_entries(path: str, record_sha: list[str]) -> Counter[str]:
    """Multiset of index-free reference entries: sha256 of (reference hash, sorted member
    record hashes).
    """
    with open(path) as f:
        entries = json.load(f)
    out: Counter[str] = Counter()
    for e in entries:
        members = sorted(record_sha[i] for i in e["member_indices"])
        out[canonical_sha256([canonical_sha256(e["reference"]), members])] += 1
    return out


def main() -> None:
    """Hash both stores, compare, write the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reference", default=REFERENCE_TAG)
    ap.add_argument("--candidate", default=CANDIDATE_TAG)
    ap.add_argument(
        "--out",
        default=None,
        help="report path (default results/compare_raw_stores_043.json)",
    )
    args = ap.parse_args()
    load_dotenv()
    base = osp.join(os.environ["DATA_ROOT"], "data/torchcell/experiments", EXPERIMENT)
    roots = {
        "reference": osp.join(base, args.reference),
        "candidate": osp.join(base, args.candidate),
    }
    t0 = time.time()
    rows = {
        k: hash_store(osp.join(r, "raw", "lmdb"), args.workers)
        for k, r in roots.items()
    }
    ms = {k: Counter(sha for _, sha in v) for k, v in rows.items()}
    by_dataset = {k: Counter(name for name, _ in v) for k, v in rows.items()}
    ref_only = ms["reference"] - ms["candidate"]
    cand_only = ms["candidate"] - ms["reference"]

    def examples(side: str, hashes: Counter[str]) -> list[dict[str, Any]]:
        out = []
        for i, (name, sha) in enumerate(rows[side]):
            if sha in hashes and len(out) < 20:
                out.append({"index": i, "dataset": name, "sha256": sha})
        return out

    genes = {}
    for k, r in roots.items():
        with open(osp.join(r, "raw", "gene_set.json")) as f:
            genes[k] = json.load(f)
    refs = {
        k: reference_entries(
            osp.join(r, "raw", "experiment_reference_index.json"),
            [sha for _, sha in rows[k]],
        )
        for k, r in roots.items()
    }
    n_same_position = sum(
        a[1] == b[1] for a, b in zip(rows["reference"], rows["candidate"], strict=False)
    )
    checks = {
        "record_count_equal": len(rows["reference"]) == len(rows["candidate"]),
        "per_dataset_counts_equal": by_dataset["reference"] == by_dataset["candidate"],
        "hash_set_equal": set(ms["reference"]) == set(ms["candidate"]),
        "hash_multiset_equal": ms["reference"] == ms["candidate"],
        "reference_only_zero": sum(ref_only.values()) == 0,
        "candidate_only_zero": sum(cand_only.values()) == 0,
        "gene_set_equal": set(genes["reference"]) == set(genes["candidate"])
        and sorted(genes["reference"]) == sorted(genes["candidate"]),
        "experiment_reference_index_equal": refs["reference"] == refs["candidate"],
    }
    report = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/compare_raw_stores_043.py",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": round(time.time() - t0, 1),
        "roots": roots,
        "method": "order-free: per record sha256 of canonical JSON (json.loads, then "
        "json.dumps sort_keys, no whitespace); multisets compared; data_<i> order "
        "is not assumed",
        "verdict": "identical" if all(checks.values()) else "different",
        "checks": checks,
        "n_records": {k: len(v) for k, v in rows.items()},
        "n_distinct_hashes": {k: len(v) for k, v in ms.items()},
        "n_duplicate_records": {
            k: sum(c - 1 for c in v.values() if c > 1) for k, v in ms.items()
        },
        "per_dataset": {
            name: {
                "reference": by_dataset["reference"].get(name, 0),
                "candidate": by_dataset["candidate"].get(name, 0),
            }
            for name in sorted(
                set(by_dataset["reference"]) | set(by_dataset["candidate"])
            )
        },
        "n_reference_only": sum(ref_only.values()),
        "n_candidate_only": sum(cand_only.values()),
        "reference_only_examples": examples("reference", ref_only),
        "candidate_only_examples": examples("candidate", cand_only),
        "gene_set_sizes": {k: len(v) for k, v in genes.items()},
        "experiment_reference_entries": {k: sum(v.values()) for k, v in refs.items()},
        "info_same_hash_at_same_index": n_same_position,
    }
    out = args.out or osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))),
        "results",
        "compare_raw_stores_043.json",
    )
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {k: v for k, v in report.items() if not k.endswith("examples")}, indent=1
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
