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
  query_pair_disjoint_splits_010build_seed_42.json.gz
                                                   {splits: {train, val, test}}, 010's disjoint split,
                                                   i.e. the partition behind Table 10 of the
                                                   additive-baselines report

The random split is read from the 010 build's own data_module_cache, the disjoint split from
experiments/010-kuzmin-tmi/results, both frozen artifacts. Nothing is recomputed.

Run from repo root:
  python experiments/025-solid-growth/scripts/make_010build_index_artifacts.py
"""

import gzip
import json
import os
import os.path as osp

from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_010 = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2")
RESULTS_010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
RESULTS_025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
N_RECORDS = 376_732


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


def main():
    with open(osp.join(BUILD_010, "data_module_cache", "index_seed_42.json")) as f:
        raw = json.load(f)
    random_split = {k: list(raw[k]) for k in ("train", "val", "test")}
    check_partition(random_split, "010 random split")

    with open(osp.join(RESULTS_010, "index_query_pair_disjoint_seed_42.json")) as f:
        raw = json.load(f)
    disjoint_split = {k: list(raw[k]) for k in ("train", "val", "test")}
    check_partition(disjoint_split, "010 query-pair-disjoint split")

    write("subset_010build_all_indices.json.gz", list(range(N_RECORDS)))
    write("pinned_splits_010build_seed_42.json.gz",
          {"source": osp.join(BUILD_010, "data_module_cache/index_seed_42.json"),
           "seed": 42, "pinned": random_split})
    write("query_pair_disjoint_splits_010build_seed_42.json.gz",
          {"source": osp.join(RESULTS_010, "index_query_pair_disjoint_seed_42.json"),
           "seed": 42, "splits": disjoint_split})
    for name, sp in (("random", random_split), ("disjoint", disjoint_split)):
        print(f"{name}: " + ", ".join(f"{k} {len(v):,}" for k, v in sp.items()))


if __name__ == "__main__":
    main()
