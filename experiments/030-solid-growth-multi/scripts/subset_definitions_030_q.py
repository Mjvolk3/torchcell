# experiments/030-solid-growth-multi/scripts/subset_definitions_030_q.py
# [[experiments.030-solid-growth-multi.scripts.subset_definitions_030_q]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/subset_definitions_030_q
"""The query-pair-disjoint (Q) split and the strict closure pool S3Q on the 030 build.

025 defined Q (``experiments/025-solid-growth/scripts/subset_definitions.py``): every
gene pair that recurs in at least 5 triples is a candidate query pair; each triple is
assigned to its most frequent recurring pair; the pairs are shuffled with seed 42 and
greedily filled toward 80/10/10 of the records. A triple is held out WITH its query pair,
so no training triple shares a held-out pair. 025 also wrote the strict rule for the
closure pool: a double whose gene pair IS a held-out query pair carries that pair's
double-mutant fitness and interaction, which is exactly what the split withholds, so it
leaves the training pool. 85 such doubles on 025. The 025 arm that would have trained on
it (``cgt_s3_q_kl_fit_036``) was never run.

This script carries the split to the 030 build and audits the pool for every channel a
held-out pair could reach training through on a build that merges nothing:

1. Transfer by PAIR, not by index: 025's ``pair_assignment`` maps each query pair to a
   split; every 030 triple's query pair is derived the same way and looked up. The split
   is also re-derived from scratch on the 030 triples with the same algorithm and seed,
   and the two assignments must agree pair for pair.
2. Excluded doubles: S3 closure doubles whose gene pair is a held-out pair. On 030 such
   a record carries its Costanzo entries at both temperatures AND the Kuzmin
   query-strain double fitness that 030 added, so the whole record leaves.
3. Training triples that contain a held-out pair as a secondary pair (a triple is
   assigned by its most frequent recurring pair; a second recurring pair in the same
   triple can be held out). Counted and, under the strict rule, excluded.
4. Held-out pairs that appear in NO training record afterwards: asserted.

Writes (``experiments/030-solid-growth-multi/results/``):
- ``query_pair_disjoint_splits_030.json.gz``: ``{"report", "pinned": {train, val,
  test}, "pair_assignment"}`` in the shape ``arm_030.resolve_arm`` reads with
  ``split_key: pinned``.
- ``subset_S3Q_indices_030.json.gz``: S3 minus the excluded doubles and triples.
- ``subset_definitions_030_q_summary.json``: ``S3QSummary``.

    PYTHONPATH=$PWD EXPERIMENT_ROOT=$PWD/experiments python \\
        experiments/030-solid-growth-multi/scripts/subset_definitions_030_q.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
import random
import sys
from collections import Counter
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import load_json_artifact, results_dir  # noqa: E402

SEED = 42
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
QUERY_PAIR_MIN_COUNT = 5
SPLITS = ("train", "val", "test")
EXPECTED_025 = {"train": 301_236, "val": 37_705, "test": 37_791}


class S3QSummary(BaseModel):
    """What the Q split and the strict closure pool contain on the 030 build."""

    seed: int
    query_pair_min_count: int
    n_triples: int
    n_recurring_pairs: int
    n_pairs_025: int
    pairs_agree_with_025: bool
    rederived_assignment_agrees_with_025: bool
    pinned_counts: dict[str, int]
    pinned_counts_025: dict[str, int]
    n_heldout_pairs: int
    n_s3: int
    n_excluded_doubles: int
    excluded_double_entries_by_dataset: dict[str, int]
    n_training_triples_with_heldout_secondary_pair: int
    n_s3q: int
    n_train_records: int
    heldout_pairs_in_training_after_exclusion: int


def query_pairs(
    triple_genes: dict[int, tuple[str, ...]],
) -> tuple[dict[int, frozenset[str]], Counter]:
    """Each triple's query pair under 025's rule: its most frequent recurring pair."""
    pair_counts: Counter = Counter()
    for gs in triple_genes.values():
        for p in combinations(gs, 2):
            pair_counts[frozenset(p)] += 1
    recurring = {p for p, c in pair_counts.items() if c >= QUERY_PAIR_MIN_COUNT}
    out: dict[int, frozenset[str]] = {}
    for idx, gs in triple_genes.items():
        rec = [frozenset(p) for p in combinations(gs, 2) if frozenset(p) in recurring]
        if not rec:
            raise ValueError(f"triple {idx} carries no recurring pair")
        out[idx] = max(rec, key=lambda p: pair_counts[p])
    return out, pair_counts


def rederive_assignment(
    triple_pair: dict[int, frozenset[str]], pair_counts: Counter, n_total: int
) -> dict[str, str]:
    """025's seeded greedy fill, run again on the 030 triples."""
    by_pair: dict[frozenset[str], list[int]] = {}
    for idx, p in triple_pair.items():
        by_pair.setdefault(p, []).append(idx)
    rng = random.Random(SEED)
    pairs = sorted(by_pair, key=lambda p: (-len(by_pair[p]), sorted(p)))
    rng.shuffle(pairs)
    filled = {s: 0 for s in SPLITS}
    assignment: dict[str, str] = {}
    for p in pairs:
        deficits = {s: TARGET[s] - filled[s] / n_total for s in SPLITS}
        s = max(deficits, key=lambda k: deficits[k])
        filled[s] += len(by_pair[p])
        assignment["+".join(sorted(p))] = s
    return assignment


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    entries = pd.read_parquet(
        osp.join(
            data_root,
            "data/torchcell/experiments/030-solid-growth-multi/closure/entries.parquet",
        ),
        columns=["idx", "order", "genes", "dataset"],
    )
    rec = entries.drop_duplicates("idx").set_index("idx")
    triple_genes = {
        int(i): tuple(sorted(g.split("|")))
        for i, g in rec.loc[rec["order"] == 3, "genes"].items()
    }
    double_pair = {
        int(i): frozenset(g.split("|"))
        for i, g in rec.loc[rec["order"] == 2, "genes"].items()
    }
    s3 = [int(i) for i in load_json_artifact("subset_S3_indices.json.gz")]
    s3_set = set(s3)
    assert set(triple_genes) <= s3_set and set(double_pair) <= s3_set

    # 1. the query pair of every 030 triple, and 025's assignment of that pair
    q025 = json.load(
        gzip.open(
            osp.join(
                osp.dirname(results_dir()),
                "..",
                "025-solid-growth",
                "results",
                "query_pair_disjoint_splits_025.json.gz",
            ),
            "rt",
        )
    )
    assign_025: dict[str, str] = q025["pair_assignment"]
    triple_pair, pair_counts = query_pairs(triple_genes)
    pairs_030 = {"+".join(sorted(p)) for p in set(triple_pair.values())}
    pairs_agree = pairs_030 == set(assign_025)
    print(f"recurring query pairs: 030 {len(pairs_030)}, 025 {len(assign_025)}, agree {pairs_agree}")
    rederived = rederive_assignment(triple_pair, pair_counts, len(triple_genes))
    rederived_agrees = rederived == assign_025
    print(f"re-derived assignment agrees with 025 pair for pair: {rederived_agrees}")
    if not pairs_agree:
        raise ValueError("the 030 triples do not carry 025's query pairs")

    pinned: dict[str, list[int]] = {s: [] for s in SPLITS}
    for idx, p in triple_pair.items():
        pinned[assign_025["+".join(sorted(p))]].append(idx)
    pinned = {s: sorted(v) for s, v in pinned.items()}
    counts = {s: len(v) for s, v in pinned.items()}
    print(f"pinned counts 030 {counts}; 025 {EXPECTED_025}")
    heldout = {
        frozenset(k.split("+")) for k, s in assign_025.items() if s in ("val", "test")
    }

    # 2. closure doubles that ARE a held-out query pair
    excluded_doubles = sorted(i for i, p in double_pair.items() if p in heldout)
    ex_entries = entries[entries["idx"].isin(excluded_doubles)]["dataset"].value_counts()
    print(f"excluded doubles: {len(excluded_doubles)}; their entries by dataset:")
    for k, v in ex_entries.items():
        print(f"    {k:42s} {v}")

    # 3. training triples that contain a held-out pair as a secondary pair
    leaky_triples = sorted(
        idx
        for idx in pinned["train"]
        if any(frozenset(p) in heldout for p in combinations(triple_genes[idx], 2))
    )
    print(f"training triples containing a held-out pair: {len(leaky_triples)}")

    # 4. the strict pool, and the assertion that no training record carries a held-out pair
    excluded = set(excluded_doubles) | set(leaky_triples)
    s3q = sorted(s3_set - excluded)
    held_records = set(pinned["val"]) | set(pinned["test"])
    train_records = sorted(set(s3q) - held_records)
    leaks = 0
    for i in train_records:
        if i in double_pair and double_pair[i] in heldout:
            leaks += 1
        elif i in triple_genes and any(
            frozenset(p) in heldout for p in combinations(triple_genes[i], 2)
        ):
            leaks += 1
    print(f"S3Q {len(s3q)} records; training {len(train_records)}; held-out pairs reachable from training: {leaks}")
    assert leaks == 0

    out_dir = results_dir()
    with gzip.open(osp.join(out_dir, "query_pair_disjoint_splits_030.json.gz"), "wt") as f:
        json.dump(
            {
                "report": {
                    "source": "experiments/025-solid-growth/results/query_pair_disjoint_splits_025.json.gz pair_assignment, transferred by query pair",
                    "seed": SEED,
                    "query_pair_min_count": QUERY_PAIR_MIN_COUNT,
                    "pinned_counts": counts,
                    "rederived_assignment_agrees_with_025": rederived_agrees,
                },
                "pinned": pinned,
                "pair_assignment": assign_025,
            },
            f,
        )
    with gzip.open(osp.join(out_dir, "subset_S3Q_indices_030.json.gz"), "wt") as f:
        json.dump(s3q, f)
    summary = S3QSummary(
        seed=SEED,
        query_pair_min_count=QUERY_PAIR_MIN_COUNT,
        n_triples=len(triple_genes),
        n_recurring_pairs=len(pairs_030),
        n_pairs_025=len(assign_025),
        pairs_agree_with_025=pairs_agree,
        rederived_assignment_agrees_with_025=rederived_agrees,
        pinned_counts=counts,
        pinned_counts_025=EXPECTED_025,
        n_heldout_pairs=len(heldout),
        n_s3=len(s3),
        n_excluded_doubles=len(excluded_doubles),
        excluded_double_entries_by_dataset={str(k): int(v) for k, v in ex_entries.items()},
        n_training_triples_with_heldout_secondary_pair=len(leaky_triples),
        n_s3q=len(s3q),
        n_train_records=len(train_records),
        heldout_pairs_in_training_after_exclusion=leaks,
    )
    with open(osp.join(out_dir, "subset_definitions_030_q_summary.json"), "w") as f:
        f.write(summary.model_dump_json(indent=2))
    print(f"wrote {out_dir}/query_pair_disjoint_splits_030.json.gz, subset_S3Q_indices_030.json.gz, subset_definitions_030_q_summary.json")


if __name__ == "__main__":
    main()
