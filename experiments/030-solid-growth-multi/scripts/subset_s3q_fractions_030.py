# experiments/030-solid-growth-multi/scripts/subset_s3q_fractions_030.py
# [[experiments.030-solid-growth-multi.scripts.subset_s3q_fractions_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/subset_s3q_fractions_030
"""Nested training subsets of the 030 strict closure pool S3Q, drawn by the SAME query
pairs as 025's subsets, for the data axis of the scaling sweep.

The 025 script (experiments/025-solid-growth/scripts/subset_s3q_fractions.py) shuffles the
331 training query pairs of the Q split once with seed 42 and keeps the first ceil(n/f) of
that order for f in {4, 16, 64}. The pair assignment of the 030 Q split is 025's, pair for
pair (subset_definitions_030_q.py), so the same shuffle on the same sorted pair list gives
the same kept pairs here, and a 030 subset holds the same genotypes as its 025 twin: the
same training triples, the same closure doubles (every closure double whose gene pair lies
inside a kept triple's gene set), all singles, and the pinned val and test triples. The
two builds then differ at every point of the curve only in what they keep per genotype
(every stored entry under its source token on 030, one merged label on 025), which is the
question the paired curves ask.

D is reported in both units: training records (the 025 unit) and gene_interaction entry
rows (the 030 unit, what the per-entry loss averages over), read from the closure
entries table.

Writes (``experiments/030-solid-growth-multi/results/``):
- ``subset_S3Q_frac{4,16,64}_indices_030.json.gz``
- ``subset_s3q_fractions_030_summary.json``

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/subset_s3q_fractions_030.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import os.path as osp
import random
import sys
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import load_json_artifact, results_dir  # noqa: E402
from subset_definitions_030_q import query_pairs  # noqa: E402

SEED = 42
FRACTIONS = (4, 16, 64)


class FractionSummary030(BaseModel):
    """One nested subset of S3Q on the 030 build."""

    denominator: int
    n_train_pairs_kept: int
    kept_pairs: list[str]
    n_train_triples: int
    n_doubles: int
    n_singles: int
    n_pool: int
    n_train_records: int
    n_train_gi_entry_rows: int
    n_train_fitness_entry_rows: int
    n_train_gi_entry_rows_by_dataset: dict[str, int]


class S3QFractions030Summary(BaseModel):
    """The three nested subsets and the full training set they are drawn from."""

    seed: int
    n_train_pairs: int
    n_s3q: int
    n_s3q_train_records: int
    n_s3q_train_gi_entry_rows: int
    fractions: list[FractionSummary030]


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    out_dir = results_dir()
    entries = pd.read_parquet(
        osp.join(
            data_root,
            "data/torchcell/experiments/030-solid-growth-multi/closure/entries.parquet",
        ),
        columns=["idx", "order", "genes", "dataset", "exp_type"],
    )
    rec = entries.drop_duplicates("idx").set_index("idx")
    s3q = [int(i) for i in load_json_artifact("subset_S3Q_indices_030.json.gz")]
    s3q_set = set(s3q)
    q = load_json_artifact("query_pair_disjoint_splits_030.json.gz")
    pinned = {k: [int(i) for i in v] for k, v in q["pinned"].items()}
    held = set(pinned["val"]) | set(pinned["test"])
    assign: dict[str, str] = q["pair_assignment"]

    triple_genes = {
        int(str(i)): tuple(sorted(g.split("|")))
        for i, g in rec.loc[rec["order"] == 3, "genes"].items()
        if int(str(i)) in s3q_set
    }
    double_pair = {
        int(str(i)): frozenset(g.split("|"))
        for i, g in rec.loc[rec["order"] == 2, "genes"].items()
        if int(str(i)) in s3q_set
    }
    singles = sorted(
        int(str(i)) for i in rec.index[rec["order"] == 1] if int(str(i)) in s3q_set
    )
    triple_pair, _ = query_pairs(triple_genes)
    train_triples = set(pinned["train"])
    for idx, p in triple_pair.items():
        key = "+".join(sorted(p))
        expected = (
            "train"
            if idx in train_triples
            else ("val" if idx in set(pinned["val"]) else "test")
        )
        assert assign[key] == expected, (
            f"triple {idx} pair {key}: {assign[key]} vs pinned {expected}"
        )

    train_pairs = sorted(k for k, s in assign.items() if s == "train")
    rng = random.Random(SEED)
    rng.shuffle(train_pairs)
    print(
        f"train query pairs: {len(train_pairs)}; S3Q {len(s3q):,} records; singles {len(singles):,}"
    )

    gi_rows = entries[entries["exp_type"] == "gene interaction"]
    fit_rows = entries[entries["exp_type"] == "fitness"]
    s3q_train = sorted(s3q_set - held)
    n_gi_full = int(gi_rows["idx"].isin(s3q_train).sum())
    out = S3QFractions030Summary(
        seed=SEED,
        n_train_pairs=len(train_pairs),
        n_s3q=len(s3q),
        n_s3q_train_records=len(s3q_train),
        n_s3q_train_gi_entry_rows=n_gi_full,
        fractions=[],
    )
    for denom in FRACTIONS:
        n_keep = math.ceil(len(train_pairs) / denom)
        kept_keys = train_pairs[:n_keep]
        kept_pairs = {frozenset(k.split("+")) for k in kept_keys}
        kept_triples = sorted(
            idx
            for idx, p in triple_pair.items()
            if p in kept_pairs and idx in train_triples
        )
        pairs_in_triples = {
            frozenset(p)
            for idx in kept_triples
            for p in combinations(triple_genes[idx], 2)
        }
        kept_doubles = sorted(
            i for i, p in double_pair.items() if p in pairs_in_triples
        )
        pool = sorted(set(singles) | set(kept_doubles) | set(kept_triples) | held)
        train_records = sorted(set(pool) - held)
        tr = set(train_records)
        gi_tr = gi_rows[gi_rows["idx"].isin(tr)]
        with gzip.open(
            osp.join(out_dir, f"subset_S3Q_frac{denom}_indices_030.json.gz"), "wt"
        ) as f:
            json.dump(pool, f)
        fs = FractionSummary030(
            denominator=denom,
            n_train_pairs_kept=n_keep,
            kept_pairs=sorted(kept_keys),
            n_train_triples=len(kept_triples),
            n_doubles=len(kept_doubles),
            n_singles=len(singles),
            n_pool=len(pool),
            n_train_records=len(train_records),
            n_train_gi_entry_rows=int(len(gi_tr)),
            n_train_fitness_entry_rows=int(fit_rows["idx"].isin(tr).sum()),
            n_train_gi_entry_rows_by_dataset={
                str(k): int(v) for k, v in gi_tr["dataset"].value_counts().items()
            },
        )
        out.fractions.append(fs)
        print(
            f"1/{denom}: pairs {n_keep}, triples {len(kept_triples):,}, doubles {len(kept_doubles):,}, "
            f"singles {len(singles):,} -> pool {len(pool):,}, training {len(train_records):,} records, "
            f"{fs.n_train_gi_entry_rows:,} gi rows"
        )
    with open(osp.join(out_dir, "subset_s3q_fractions_030_summary.json"), "w") as f:
        f.write(out.model_dump_json(indent=2))
    print(
        "wrote subset_S3Q_frac{4,16,64}_indices_030.json.gz and subset_s3q_fractions_030_summary.json"
    )


if __name__ == "__main__":
    main()
