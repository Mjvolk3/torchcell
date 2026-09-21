# experiments/025-solid-growth/scripts/subset_s3_query_strict.py
# [[experiments.025-solid-growth.scripts.subset_s3_query_strict]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/subset_s3_query_strict
"""The closure subset S3 for the query-pair-disjoint split: S3 minus the held-out pairs.

On the Q split a triple is held out by its query pair. 85 closure doubles ARE such a pair
(`subset_Q_excluded_doubles.json.gz`, written by `subset_definitions.py`), and an arm that
sends every unpinned record to train would fit the double-mutant fitness of exactly the
pairs the split withholds. This writes S3 without them, from the two committed artifacts,
so it needs no scan of the build.

    python experiments/025-solid-growth/scripts/subset_s3_query_strict.py

Writes results/subset_S3Q_indices.json.gz.
"""

import gzip
import json
import os
import os.path as osp

from dotenv import load_dotenv

load_dotenv()
RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "025-solid-growth", "results")


def load(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def main() -> None:
    s3 = load("subset_S3_indices.json.gz")
    excluded = set(load("subset_Q_excluded_doubles.json.gz"))
    split = load("query_pair_disjoint_splits_025.json.gz")["splits"]
    pinned = {i for v in split.values() for i in v}
    assert excluded <= set(s3), "an excluded double is not in S3"
    assert not excluded & pinned, "an excluded double is a pinned triple"
    s3q = [i for i in s3 if i not in excluded]
    assert pinned <= set(s3q), "a pinned triple is missing from the strict subset"
    with gzip.open(osp.join(RESULTS_DIR, "subset_S3Q_indices.json.gz"), "wt") as f:
        json.dump(s3q, f)
    print(f"S3 {len(s3):,} - excluded {len(excluded)} = S3Q {len(s3q):,}; pinned triples {len(pinned):,}")


if __name__ == "__main__":
    main()
