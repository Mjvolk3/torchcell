# experiments/025-solid-growth/scripts/subset_s3q_fractions.py
# [[experiments.025-solid-growth.scripts.subset_s3q_fractions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/subset_s3q_fractions
"""Nested training subsets of the strict closure pool S3Q, drawn by query pair, for the D sweep.

The data axis of a scaling curve is swept over DISTINCT training records against one fixed
evaluation set, and the subsets are nested and drawn by the unit that leaks. On the trigenic
build that unit is the query pair: a smaller subset removes whole query pairs with every
triple they reach, so it behaves like having run fewer screens rather than like thinning the
same screens. The evaluation set, the pinned Q val and test triples, is untouched.

Construction for fraction f in {1/4, 1/16, 1/64}:

1. The TRAIN query pairs of the Q split (``query_pair_disjoint_splits_025.json.gz``,
   ``pair_assignment == "train"``) are shuffled once with seed 42 and the first ceil(f * n)
   pairs of that order are kept, so the subsets nest (the 1/64 pairs are inside the 1/16
   pairs, inside the 1/4 pairs).
2. Kept triples: the training triples whose query pair is kept.
3. Kept doubles: the S3Q closure doubles whose gene pair lies inside a KEPT triple's gene
   set (the closure of the kept triples, which is what a screen contributes), read from the
   025 LMDB once for the 739,219 closure doubles.
4. Singles: all 5,694 (one record per gene; cheap, and present in every subset so the single
   fitness supervision does not vary with f).

The pool written for each f is kept singles + kept doubles + kept triples + ALL pinned val
and test triples (the data module pins those by index, so they must be in the pool). With
``unpinned_to_train`` every pool record outside val/test trains. The leak audit of the
strict rule carries over unchanged: S3Q already excludes every double whose pair is held
out, and a subset of S3Q cannot add one back.

Writes (``experiments/025-solid-growth/results/``):
- ``subset_S3Q_frac4_indices.json.gz``, ``subset_S3Q_frac16_indices.json.gz``,
  ``subset_S3Q_frac64_indices.json.gz``
- ``subset_s3q_fractions_summary.json``
- ``subset_S3Q_double_pairs.json.gz``: the gene pair of every S3Q closure double, read once
  from the LMDB and cached

    PYTHONPATH=$PWD python experiments/025-solid-growth/scripts/subset_s3q_fractions.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import os.path as osp
import random
import re
from itertools import combinations

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build/processed"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
GENE_RE = re.compile(rb'"systematic_gene_name": "([^"]+)"')
SEED = 42
FRACTIONS = (4, 16, 64)
QUERY_PAIR_MIN_COUNT = 5


class FractionSummary(BaseModel):
    """One nested subset of S3Q."""

    denominator: int
    n_train_pairs_kept: int
    n_train_triples: int
    n_doubles: int
    n_singles: int
    n_pool: int
    n_train_records: int


class S3QFractionsSummary(BaseModel):
    seed: int
    n_train_pairs: int
    n_s3q: int
    n_s3q_train_records: int
    fractions: list[FractionSummary]


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def dump_gz(name: str, obj) -> None:
    with gzip.open(osp.join(RESULTS_DIR, name), "wt") as f:
        json.dump(obj, f)


def triple_gene_sets() -> dict[int, tuple[str, str, str]]:
    path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
    )
    out: dict[int, tuple[str, str, str]] = {}
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split(",")
        col = {c: i for i, c in enumerate(header)}
        for line in f:
            v = line.rstrip("\n").split(",")
            out[int(v[col["idx_025"]])] = (v[col["gene_a"]], v[col["gene_b"]], v[col["gene_c"]])
    return out


def double_pairs(indices: list[int]) -> dict[int, frozenset[str]]:
    env = lmdb.open(osp.join(BUILD, "lmdb"), readonly=True, lock=False, readahead=False)
    out: dict[int, frozenset[str]] = {}
    with env.begin() as txn:
        for n, i in enumerate(indices):
            raw = txn.get(str(i).encode())
            out[i] = frozenset(m.decode() for m in set(GENE_RE.findall(raw)))
            if n and n % 200000 == 0:
                print(f"  doubles read: {n:,}", flush=True)
    env.close()
    return out


def query_pair_of(genes: tuple[str, ...], recurring: dict[frozenset[str], int]) -> frozenset[str]:
    rec = [frozenset(p) for p in combinations(genes, 2) if frozenset(p) in recurring]
    return max(rec, key=lambda p: recurring[p])


def main() -> None:
    with open(osp.join(BUILD, "perturbation_count_index.json")) as f:
        count_index = json.load(f)
    singles = [int(i) for i in count_index["1"]]
    s3q = [int(i) for i in load_gz("subset_S3Q_indices.json.gz")]
    s3q_set = set(s3q)
    q = load_gz("query_pair_disjoint_splits_025.json.gz")
    pinned = {k: [int(i) for i in v] for k, v in q["splits"].items()}
    held = set(pinned["val"]) | set(pinned["test"])
    assign: dict[str, str] = q["pair_assignment"]

    triples = triple_gene_sets()
    assert set(triples) <= s3q_set
    pair_counts: dict[frozenset[str], int] = {}
    for gs in triples.values():
        for p in combinations(gs, 2):
            pair_counts[frozenset(p)] = pair_counts.get(frozenset(p), 0) + 1
    recurring = {p: c for p, c in pair_counts.items() if c >= QUERY_PAIR_MIN_COUNT}
    triple_pair = {idx: query_pair_of(gs, recurring) for idx, gs in triples.items()}
    for idx, p in triple_pair.items():
        key = "+".join(sorted(p))
        expected = "train" if idx in set(pinned["train"]) else ("val" if idx in set(pinned["val"]) else "test")
        assert assign[key] == expected, f"triple {idx} pair {key} assigned {assign[key]} but pinned {expected}"

    train_pairs = sorted(k for k, s in assign.items() if s == "train")
    rng = random.Random(SEED)
    rng.shuffle(train_pairs)
    print(f"train query pairs: {len(train_pairs)}; S3Q {len(s3q):,} records")

    single_set = set(singles)
    closure_doubles = [i for i in s3q if i not in triples and i not in single_set]
    cache = osp.join(RESULTS_DIR, "subset_S3Q_double_pairs.json.gz")
    if osp.exists(cache):
        with gzip.open(cache, "rt") as f:
            dpair = {int(k): frozenset(v) for k, v in json.load(f).items()}
        print(f"gene pairs of {len(dpair):,} closure doubles read from {cache}")
    else:
        print(f"reading gene pairs of {len(closure_doubles):,} closure doubles from the 025 LMDB")
        dpair = double_pairs(closure_doubles)
        with gzip.open(cache, "wt") as f:
            json.dump({str(k): sorted(v) for k, v in dpair.items()}, f)
    assert set(dpair) == set(closure_doubles)

    s3q_train = sorted(s3q_set - held)
    out = S3QFractionsSummary(
        seed=SEED, n_train_pairs=len(train_pairs), n_s3q=len(s3q), n_s3q_train_records=len(s3q_train), fractions=[]
    )
    for denom in FRACTIONS:
        n_keep = math.ceil(len(train_pairs) / denom)
        kept_pairs = {frozenset(k.split("+")) for k in train_pairs[:n_keep]}
        kept_triples = [idx for idx, p in triple_pair.items() if p in kept_pairs]
        # every gene pair inside a kept triple, so the double test is one set lookup
        kept_pairs_in_triples = {
            frozenset(p) for idx in kept_triples for p in combinations(triples[idx], 2)
        }
        kept_doubles = [i for i, p in dpair.items() if p in kept_pairs_in_triples]
        pool = sorted(set(singles) | set(kept_doubles) | set(kept_triples) | held)
        train_records = sorted(set(pool) - held)
        dump_gz(f"subset_S3Q_frac{denom}_indices.json.gz", pool)
        fs = FractionSummary(
            denominator=denom,
            n_train_pairs_kept=n_keep,
            n_train_triples=len(kept_triples),
            n_doubles=len(kept_doubles),
            n_singles=len(singles),
            n_pool=len(pool),
            n_train_records=len(train_records),
        )
        out.fractions.append(fs)
        print(
            f"1/{denom}: pairs {n_keep}, triples {len(kept_triples):,}, doubles {len(kept_doubles):,}, "
            f"singles {len(singles):,} -> pool {len(pool):,}, training {len(train_records):,}"
        )
    with open(osp.join(RESULTS_DIR, "subset_s3q_fractions_summary.json"), "w") as f:
        f.write(out.model_dump_json(indent=2))
    print("wrote subset_S3Q_frac{4,16,64}_indices.json.gz and subset_s3q_fractions_summary.json")


if __name__ == "__main__":
    main()
