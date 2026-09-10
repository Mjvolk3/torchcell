#!/usr/bin/env python
# experiments/025-solid-growth/scripts/make_010build_index_artifacts.py
# [[experiments.025-solid-growth.scripts.make_010build_index_artifacts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/make_010build_index_artifacts
"""Index artifacts that let the 025 training script run on the 010 build.

WHY. The 025 full build is 3.2 TB (517 GB of LMDB alone), so it cannot follow the
graph-regularization sweep to Delta. The S0 subset is a list of indices INTO that LMDB, so
it does not help on its own. But S0 is, record for record, the 010 build: 376,732 triples,
labels bit-identical on 376,526 of them and within 6e-17 on the rest
(results/label_parity_010_vs_025.json). The migrated 010 build (001-small-build-schema-v2, the key-preserving copy that validates under the current schema) is 1.5 GB. So the sweep runs the 025
script on the 010 build, and this writes the three index artifacts that script expects,
in 010's own record-index space (0 to 376,731):

  subset_010build_all_indices.json.gz              every record; the "subset" is the whole build
  pinned_splits_010build_seed_42.json.gz           {pinned: {train, val, test}}, 010's random split
  query_pair_disjoint_splits_010build_armq.json.gz {splits: {train, val, test}}, ARM Q's
                                                   query-pair-disjoint partition, carried into
                                                   010-build index space

WHICH DISJOINT SPLIT, AND WHY IT IS NOT 010'S OWN. Two query-pair-disjoint partitions of these
376,732 records exist. 010's own (experiments/010-kuzmin-tmi/results/index_query_pair_disjoint_seed_42.json,
285/67/68 pairs, 301,483/37,569/37,680 records) is the single split behind the cross-validated
table of the additive-baselines report. Arm Q's (results/query_pair_disjoint_splits_025.json.gz,
331/43/46 pairs, 301,236/37,705/37,791 records) is the one the six additive nulls were refit on
and the one the GilaHyper disjoint run trains on. They assign the same 420 recurring pairs to
different folds, so a Delta result on 010's split would compare to neither the arm Q baselines
(B1 0.185, B5 0.141) nor the GilaHyper run. This writes arm Q's partition, so every disjoint
number lands on one set of held-out screens.

HOW IT IS CARRIED. Record indices do not survive across builds, but the arm Q artifact stores
``pair_assignment``, a query pair name to split map that is index-free. So the transfer regroups
the 010 records by query pair with the rule ``subset_definitions.query_pair_split`` used, then
reads each pair's fold from that map. Nothing is reshuffled and the seed plays no part here. The
run asserts the recurring pair set matches arm Q's 420 pairs and that the resulting fold sizes
equal arm Q's exactly, which they must, the two record populations being genotype-identical.

The random split is read from the 010 build's own data_module_cache, a frozen artifact.

Run from repo root:
  python experiments/025-solid-growth/scripts/make_010build_index_artifacts.py
"""

import gzip
import json
import os
import os.path as osp
from collections import Counter
from itertools import combinations

import lmdb
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_010 = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2")
RESULTS_010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
RESULTS_025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
N_RECORDS = 376_732
ARM_Q = "query_pair_disjoint_splits_025.json.gz"
QUERY_PAIR_MIN_COUNT = 5
SPLITS = ("train", "val", "test")


def check_partition(splits: dict[str, list[int]], name: str) -> None:
    parts = [set(v) for v in splits.values()]
    n = sum(len(p) for p in parts)
    if n != N_RECORDS:
        raise SystemExit(f"{name}: {n:,} indices, expected {N_RECORDS:,}")
    if set.union(*parts) != set(range(N_RECORDS)):
        raise SystemExit(f"{name}: indices are not exactly 0..{N_RECORDS - 1}")
    for a in parts:
        for b in parts:
            if a is not b and a & b:
                raise SystemExit(f"{name}: splits overlap")


def write(name: str, payload: dict) -> None:
    path = osp.join(RESULTS_025, name)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)
    print(f"wrote {path}")


def read_010_gene_sets() -> dict[int, tuple[str, ...]]:
    """010 record index -> its sorted perturbed-gene tuple.

    The genotype path is the one ``transfer_010_tmi_splits.record_identity`` uses: a
    post-aggregation record is a list of entries sharing one aggregation key, and the first
    entry's genotype carries the gene set. Sorting matters, because the pair-selection rule
    below enumerates ``combinations`` in gene order and arm Q enumerated a sorted tuple.
    """
    env = lmdb.open(osp.join(BUILD_010, "processed/lmdb"), readonly=True, lock=False, readahead=False)
    out: dict[int, tuple[str, ...]] = {}
    with env.begin() as txn:
        n = txn.stat()["entries"]
        if n != N_RECORDS:
            raise SystemExit(f"010 build holds {n:,} records, expected {N_RECORDS:,}")
        for i in tqdm(range(n), desc="reading 010 genotypes"):
            record = json.loads(txn.get(str(i).encode()))
            perturbations = record[0]["experiment"]["genotype"]["perturbations"]
            out[i] = tuple(sorted({p["systematic_gene_name"] for p in perturbations}))
    env.close()
    return out


def carry_arm_q_split(gene_sets: dict[int, tuple[str, ...]]) -> tuple[dict, dict]:
    """Assign each 010 record the fold arm Q gave its query pair."""
    with gzip.open(osp.join(RESULTS_025, ARM_Q), "rt") as f:
        arm_q = json.load(f)
    pair_assignment: dict[str, str] = arm_q["pair_assignment"]
    arm_q_sizes = {k: len(v) for k, v in arm_q["splits"].items()}

    pair_counts: Counter = Counter()
    for gs in gene_sets.values():
        for p in combinations(gs, 2):
            pair_counts[frozenset(p)] += 1
    recurring = {p for p, c in pair_counts.items() if c >= QUERY_PAIR_MIN_COUNT}
    names = {"+".join(sorted(p)) for p in recurring}
    if names != set(pair_assignment):
        raise SystemExit(
            f"recurring pairs differ from arm Q: {len(names)} here, "
            f"{len(pair_assignment)} there, {len(names ^ set(pair_assignment))} symmetric difference"
        )

    splits: dict[str, list[int]] = {s: [] for s in SPLITS}
    for idx, gs in gene_sets.items():
        rec = [frozenset(p) for p in combinations(gs, 2) if frozenset(p) in recurring]
        if not rec:
            raise SystemExit(f"record {idx} carries no recurring query pair")
        # Ties resolve to the most frequent pair, matching subset_definitions and the 010 script.
        best = max(rec, key=lambda p: pair_counts[p])
        splits[pair_assignment["+".join(sorted(best))]].append(idx)

    sizes = {k: len(v) for k, v in splits.items()}
    if sizes != arm_q_sizes:
        raise SystemExit(f"carried sizes {sizes} do not match arm Q {arm_q_sizes}")
    return {s: sorted(v) for s, v in splits.items()}, arm_q_sizes


def main():
    with open(osp.join(BUILD_010, "data_module_cache", "index_seed_42.json")) as f:
        raw = json.load(f)
    random_split = {k: list(raw[k]) for k in SPLITS}
    check_partition(random_split, "010 random split")

    disjoint_split, arm_q_sizes = carry_arm_q_split(read_010_gene_sets())
    check_partition(disjoint_split, "arm Q query-pair-disjoint split")
    print(f"arm Q sizes reproduced exactly: {arm_q_sizes}")

    write("subset_010build_all_indices.json.gz", list(range(N_RECORDS)))
    write("pinned_splits_010build_seed_42.json.gz",
          {"source": osp.join(BUILD_010, "data_module_cache/index_seed_42.json"),
           "seed": 42, "pinned": random_split})
    write("query_pair_disjoint_splits_010build_armq.json.gz",
          {"source": osp.join(RESULTS_025, ARM_Q),
           "carried_by": "query pair name, via pair_assignment",
           "splits": disjoint_split})
    for name, sp in (("random", random_split), ("disjoint", disjoint_split)):
        print(f"{name}: " + ", ".join(f"{k} {len(v):,}" for k, v in sp.items()))


if __name__ == "__main__":
    main()
