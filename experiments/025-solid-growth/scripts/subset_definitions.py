# experiments/025-solid-growth/scripts/subset_definitions.py
# [[experiments.025-solid-growth.training-plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/subset_definitions
"""Materialize the 025 training-campaign subsets and splits as index artifacts.

Arms (record pools; training data only -- evaluation is always the tmi triples):

- S0/S1: triples only (S0 vs S1 differ in which labels the trainer uses, not in records)
- S2: triples + all singles
- S3: S2 + closure doubles (every double whose gene pair lies inside some triple)
- S4: S2 + size-matched random non-closure doubles (|S4 doubles| = |S3 doubles|)
- S6: singles + all doubles-in-closure + triples (same pool as S3; the trainer masks
  trigenic gene_interaction supervision, so no separate pool is emitted)

Splits over the triples:

- R: 010's random-over-records split, already pinned
  (results/pinned_splits_from_010_seed_42.json).
- Q: query-pair-disjoint, ported from
  experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_split.py -- whole recurring
  query pairs are assigned to train/val/test by seeded shuffle + greedy fill toward
  80/10/10 record proportions. Non-triple records train only. Two double-exclusion
  variants are emitted for the leakage rule: "strict" drops training doubles whose gene
  pair IS a val/test query pair; "diagnostic" keeps them.

Outputs (experiments/025-solid-growth/results/):
- subset_S{0,2,3,4}_indices.json.gz  (record-index lists into the 025 build)
- query_pair_disjoint_splits_025.json.gz  (triple indices per split + per-pair assignment)
- subset_Q_excluded_doubles.json.gz  (train doubles removed under the strict rule)
- subset_definitions_summary.json
"""

import gzip
import json
import os
import os.path as osp
import random
import re
from collections import Counter
from itertools import combinations

import lmdb
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build/processed"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
GENE_RE = re.compile(rb'"systematic_gene_name": "([^"]+)"')

SEED = 42
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
QUERY_PAIR_MIN_COUNT = 5


def load_triple_gene_sets() -> dict[int, tuple]:
    """025 triple index -> sorted gene tuple, from the recapitulation per-triple table."""
    path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
    )
    out: dict[int, tuple] = {}
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split(",")
        col = {c: i for i, c in enumerate(header)}
        for line in f:
            v = line.rstrip("\n").split(",")
            out[int(v[col["idx_025"]])] = (
                v[col["gene_a"]], v[col["gene_b"]], v[col["gene_c"]]
            )
    return out


def scan_double_pairs(idx_double: list[int]) -> dict[int, frozenset]:
    """025 double index -> gene pair. Regex membership is exact: reference blocks carry
    no perturbations, so gene names in raw bytes come only from experiment genotypes."""
    env = lmdb.open(osp.join(BUILD, "lmdb"), readonly=True, lock=False)
    out: dict[int, frozenset] = {}
    with env.begin() as txn:
        for n, i in enumerate(idx_double):
            raw = txn.get(str(i).encode())
            out[i] = frozenset(m.decode() for m in set(GENE_RE.findall(raw)))
            if n % 1000000 == 0:
                print(f"doubles scanned: {n}", flush=True)
    env.close()
    return out


def query_pair_split(triple_genes: dict[int, tuple]) -> dict:
    pair_counts: Counter = Counter()
    for gs in triple_genes.values():
        for p in combinations(gs, 2):
            pair_counts[frozenset(p)] += 1
    recurring = {p for p, c in pair_counts.items() if c >= QUERY_PAIR_MIN_COUNT}

    by_pair: dict[frozenset, list[int]] = {p: [] for p in recurring}
    unassigned = []
    for idx, gs in triple_genes.items():
        rec = [frozenset(p) for p in combinations(gs, 2) if frozenset(p) in recurring]
        if not rec:
            unassigned.append(idx)
            continue
        # every record carries exactly one recurring pair (one 010 record carries two);
        # ties resolve to the most frequent, matching the 010 script
        best = max(rec, key=lambda p: pair_counts[p])
        by_pair[best].append(idx)
    assert len(unassigned) == 0, f"{len(unassigned)} triples carry no recurring pair"

    rng = random.Random(SEED)
    pairs = sorted(by_pair, key=lambda p: (-len(by_pair[p]), sorted(p)))
    rng.shuffle(pairs)
    n_total = len(triple_genes)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    pair_assignment: dict[str, str] = {}
    for p in pairs:
        deficits = {
            s: TARGET[s] - len(splits[s]) / n_total for s in ("train", "val", "test")
        }
        s = max(deficits, key=lambda k: deficits[k])
        splits[s].extend(by_pair[p])
        pair_assignment["+".join(sorted(p))] = s
    return {
        "seed": SEED,
        "query_pair_min_count": QUERY_PAIR_MIN_COUNT,
        "n_recurring_pairs": len(recurring),
        "splits": {s: sorted(v) for s, v in splits.items()},
        "pair_assignment": pair_assignment,
    }


def dump_gz(path: str, obj) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def main() -> None:
    with open(osp.join(BUILD, "perturbation_count_index.json")) as f:
        count_index = json.load(f)
    idx_single, idx_double, idx_triple = (
        count_index["1"], count_index["2"], count_index["3"]
    )

    triple_genes = load_triple_gene_sets()
    assert len(triple_genes) == len(idx_triple)
    closure_pairs = {
        frozenset(p) for gs in triple_genes.values() for p in combinations(gs, 2)
    }

    double_pairs = scan_double_pairs(idx_double)
    closure_doubles = sorted(i for i, p in double_pairs.items() if p in closure_pairs)
    non_closure = [i for i, p in double_pairs.items() if p not in closure_pairs]
    rng = random.Random(SEED)
    s4_doubles = sorted(rng.sample(non_closure, len(closure_doubles)))

    s0 = sorted(idx_triple)
    s2 = sorted(idx_triple + idx_single)
    s3 = sorted(s2 + closure_doubles)
    s4 = sorted(s2 + s4_doubles)
    dump_gz(osp.join(RESULTS_DIR, "subset_S0_indices.json.gz"), s0)
    dump_gz(osp.join(RESULTS_DIR, "subset_S2_indices.json.gz"), s2)
    dump_gz(osp.join(RESULTS_DIR, "subset_S3_indices.json.gz"), s3)
    dump_gz(osp.join(RESULTS_DIR, "subset_S4_indices.json.gz"), s4)

    q = query_pair_split(triple_genes)
    dump_gz(osp.join(RESULTS_DIR, "query_pair_disjoint_splits_025.json.gz"), q)

    heldout_pairs = {
        frozenset(k.split("+"))
        for k, s in q["pair_assignment"].items()
        if s in ("val", "test")
    }
    q_excluded = sorted(
        i for i in closure_doubles if double_pairs[i] in heldout_pairs
    )
    dump_gz(osp.join(RESULTS_DIR, "subset_Q_excluded_doubles.json.gz"), q_excluded)

    summary = {
        "n_singles": len(idx_single),
        "n_doubles": len(idx_double),
        "n_triples": len(idx_triple),
        "n_closure_pairs": len(closure_pairs),
        "n_closure_doubles_found": len(closure_doubles),
        "subset_sizes": {"S0": len(s0), "S2": len(s2), "S3": len(s3), "S4": len(s4)},
        "q_split": {
            "n_recurring_pairs": q["n_recurring_pairs"],
            "records": {s: len(v) for s, v in q["splits"].items()},
            "pairs": dict(Counter(q["pair_assignment"].values())),
        },
        "q_strict_excluded_train_doubles": len(q_excluded),
    }
    with open(osp.join(RESULTS_DIR, "subset_definitions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
