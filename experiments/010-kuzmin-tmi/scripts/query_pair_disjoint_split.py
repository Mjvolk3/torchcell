# experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_split.py
# [[experiments.010-kuzmin-tmi.scripts.query_pair_disjoint_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_split
"""Build a query-pair-disjoint split of the 010 trigenic build.

Kuzmin's trigenic screen crosses a query DOUBLE mutant against an array of
single mutants, so every triple is a query pair plus one array gene and the
same query pair recurs across hundreds or thousands of records. The published
010 split (``index_seed_42.json``) is random over records, so every query pair
appears in train, val and test at once. A model that can represent a per-pair
offset therefore scores partly on knowing which screen a record came from,
which is not trigenic biology.

The query-pair structure of this build is exact rather than approximate:

    376,732 records, 739,315 distinct unordered gene pairs
    420 pairs occur 5 or more times, covering 376,733 pair-instances
    every record's most frequent pair occurs at least 200 times

376,733 is 376,732 plus one, so each record carries exactly one of the 420
recurring pairs, with a single record carrying two. Those 420 pairs are the
Kuzmin query doubles, and grouping records by them recovers the screen.

This script assigns whole query pairs to train, val and test, so no query
double is ever seen in more than one split. Group sizes are unequal, ranging
into the thousands, so the assignment is a seeded shuffle followed by a
greedy fill toward the target record proportions rather than a shuffle of
group labels, which would miss the target badly.

Output is written in the same schema as ``index_seed_42.json``, a dict of
split name to record index list, so the data module and the baseline scripts
can consume it unchanged.
"""

import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_DIR = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/010-kuzmin-tmi/001-small-build",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

# A pair must recur at least this often to be read as a Kuzmin query double.
# The count distribution is bimodal with nothing between 2 and 200, so the
# result is insensitive to the exact value anywhere in that gap.
QUERY_PAIR_MIN_COUNT = 5

SEED = 42
TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}


def load_records() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (record_ids, row_genes, y, gene_names) on a dense row index.

    ``row_genes`` holds the three perturbed gene columns per record, sorted, so
    pair keys are canonical.
    """
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(
        osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")
    ) as f:
        gene_index = json.load(f)

    record_ids = label_df["index"].to_numpy()
    order = np.argsort(record_ids)
    record_ids = record_ids[order]
    y = label_df["gene_interaction"].to_numpy()[order]
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}

    gene_names = sorted(gene_index.keys())
    gene_to_col = {g: j for j, g in enumerate(gene_names)}

    row_genes = np.full((len(record_ids), 3), -1, dtype=np.int32)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for gene, ids in gene_index.items():
        col = gene_to_col[gene]
        for rid in ids:
            row = id_to_row[int(rid)]
            row_genes[row, fill[row]] = col
            fill[row] += 1
    assert (fill == 3).all(), "every 010 record must carry exactly 3 perturbed genes"

    return record_ids, np.sort(row_genes, axis=1), y, gene_names


def pair_keys(row_genes: np.ndarray) -> tuple[np.ndarray, int]:
    """Canonical unordered pair keys, shape (n, 3), encoded as lo * base + hi."""
    base = int(row_genes.max()) + 1
    a, b, c = row_genes[:, 0], row_genes[:, 1], row_genes[:, 2]
    pairs = np.stack(
        [
            a.astype(np.int64) * base + b,
            a.astype(np.int64) * base + c,
            b.astype(np.int64) * base + c,
        ],
        axis=1,
    )
    return pairs, base


def assign_query_pair(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each record to its query pair.

    The query pair is the record's most frequent pair. One record carries two
    recurring pairs; taking the argmax of the count resolves it deterministically,
    and ties break on the smaller key so the result does not depend on ordering.
    """
    keys, counts = np.unique(pairs.reshape(-1), return_counts=True)
    count_of = dict(zip(keys.tolist(), counts.tolist()))

    recurring = {int(k) for k, c in count_of.items() if c >= QUERY_PAIR_MIN_COUNT}
    print(
        f"distinct pairs {keys.size}  "
        f"recurring (>={QUERY_PAIR_MIN_COUNT}) {len(recurring)}  "
        f"pair-instances covered {sum(count_of[k] for k in recurring)}"
    )

    query = np.empty(pairs.shape[0], dtype=np.int64)
    n_multi = 0
    for i, row in enumerate(pairs):
        hits = [int(p) for p in row if int(p) in recurring]
        if len(hits) > 1:
            n_multi += 1
        if not hits:
            raise AssertionError(
                f"record {i} carries no recurring pair; the query-pair reading "
                "of this build is wrong and the split must not be used"
            )
        # Most frequent first, then smaller key, so the choice is total.
        query[i] = min(hits, key=lambda k: (-count_of[k], k))
    print(f"records carrying more than one recurring pair: {n_multi}")
    return query, np.array(sorted(recurring), dtype=np.int64)


def split_groups(
    query: np.ndarray, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Assign whole query pairs to splits, greedily approaching TARGET.

    Groups are shuffled once, then walked largest-deficit-first: each group goes
    to whichever split is furthest below its target share of records. With 420
    groups of very unequal size this lands within a fraction of a percent of the
    target, which shuffling group labels does not.
    """
    rng = np.random.default_rng(seed)
    sizes = Counter(query.tolist())
    groups = np.array(sorted(sizes.keys()), dtype=np.int64)
    rng.shuffle(groups)
    # Large groups first, so the biggest indivisible chunks are placed while
    # every split still has room; otherwise one 4,119-record group lands last
    # and blows a 10 percent target.
    groups = sorted(groups.tolist(), key=lambda g: (-sizes[g], g))

    total = int(query.size)
    placed = {k: 0 for k in TARGET}
    assign: dict[str, list[int]] = {k: [] for k in TARGET}
    for g in groups:
        deficit = {k: TARGET[k] - placed[k] / total for k in TARGET}
        pick = max(deficit, key=lambda k: (deficit[k], k))
        assign[pick].append(g)
        placed[pick] += sizes[g]

    splits = {
        name: np.nonzero(np.isin(query, np.array(gs, dtype=np.int64)))[0]
        for name, gs in assign.items()
    }
    return splits, assign


def report(
    splits: dict[str, np.ndarray],
    assign: dict[str, list[int]],
    query: np.ndarray,
    row_genes: np.ndarray,
    y: np.ndarray,
    base: int,
    gene_names: list[str],
) -> dict[str, object]:
    total = int(query.size)
    print("\nsplit sizes")
    stats: dict[str, object] = {}
    for name in ("train", "val", "test"):
        idx = splits[name]
        print(
            f"  {name:<6s} records {idx.size:>7d} ({idx.size / total:.4%})  "
            f"query pairs {len(assign[name]):>4d}  label sd {y[idx].std(ddof=0):.6f}"
        )
        stats[name] = {
            "records": int(idx.size),
            "query_pairs": len(assign[name]),
            "label_sd": float(y[idx].std(ddof=0)),
        }

    # The invariant the split exists to enforce.
    sets = {n: set(assign[n]) for n in assign}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                shared = sets[a] & sets[b]
                print(f"  query pairs shared {a}/{b}: {len(shared)}")
                assert not shared, "query-pair disjointness violated"

    # Gene-level overlap is expected and is not what this split controls: array
    # genes are shared by design. Reporting it keeps the claim honest.
    tr_genes = set(np.unique(row_genes[splits["train"]]).tolist())
    print("\ngene coverage (array genes are shared by design)")
    for name in ("val", "test"):
        g = set(np.unique(row_genes[splits[name]]).tolist())
        unseen = g - tr_genes
        print(
            f"  {name:<6s} distinct genes {len(g):>5d}  "
            f"unseen in train {len(unseen):>5d} ({len(unseen) / len(g):.2%})"
        )
        stats[f"{name}_genes"] = {
            "distinct": len(g),
            "unseen_in_train": len(unseen),
        }
        rec_unseen = np.isin(row_genes[splits[name]], list(unseen)).any(axis=1)
        print(
            f"         records containing a gene unseen in train: "
            f"{int(rec_unseen.sum())} ({rec_unseen.mean():.2%})"
        )
        stats[f"{name}_genes"]["records_with_unseen_gene"] = int(rec_unseen.sum())

    # The two genes of a held-out query pair may still appear in train as array
    # genes; that is the residual leakage this split does NOT remove.
    print("\nquery-pair gene leakage into train")
    for name in ("val", "test"):
        qg = set()
        for k in assign[name]:
            qg.add(int(k) // base)
            qg.add(int(k) % base)
        seen_as_any = qg & tr_genes
        print(
            f"  {name:<6s} genes forming held-out query pairs {len(qg):>4d}  "
            f"of which appear somewhere in train {len(seen_as_any):>4d} "
            f"({len(seen_as_any) / len(qg):.1%})"
        )
        stats[f"{name}_query_genes"] = {
            "distinct": len(qg),
            "appear_in_train": len(seen_as_any),
        }
    return stats


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record_ids, row_genes, y, gene_names = load_records()
    print(f"records {record_ids.size}  genes {len(gene_names)}")

    pairs, base = pair_keys(row_genes)
    query, recurring = assign_query_pair(pairs)
    splits, assign = split_groups(query, SEED)
    stats = report(splits, assign, query, row_genes, y, base, gene_names)

    # Write in the schema of index_seed_42.json: record ids, not dense rows.
    out = {
        name: [int(record_ids[i]) for i in np.sort(splits[name])]
        for name in ("train", "val", "test")
    }
    assert sum(len(v) for v in out.values()) == record_ids.size
    assert len(set().union(*(set(v) for v in out.values()))) == record_ids.size

    split_path = osp.join(RESULTS_DIR, "index_query_pair_disjoint_seed_42.json")
    with open(split_path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {split_path}")

    detail_path = osp.join(RESULTS_DIR, "query_pair_disjoint_split_summary.json")
    with open(detail_path, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "query_pair_min_count": QUERY_PAIR_MIN_COUNT,
                "n_query_pairs": int(recurring.size),
                "target": TARGET,
                "stats": stats,
                "query_pairs_by_split": {
                    name: [
                        f"{gene_names[int(k) // base]}_{gene_names[int(k) % base]}"
                        for k in sorted(gs)
                    ]
                    for name, gs in assign.items()
                },
            },
            f,
            indent=2,
        )
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
