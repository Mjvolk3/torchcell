# experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover.py
# [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover
"""Minimum set of double mutants needed to construct the panel-12 triples.

In SGA trigenic construction a triple {A,B,C} is built by crossing ONE of its
three doubles (AB, AC, or BC) with the remaining single. So the doubles you must
physically build are not all C(12,2)=66 — only a set that "covers" every
constructible triple (each triple shares >=1 double with the set). This is a
classic minimum set-cover, solved greedily here (take the double covering the
most still-uncovered triples each round).

Input : results/inference_3/top_k_constructible_panel12_k200.csv  (52 triples)
Output: results/optimized_doubles_setcover_panel12.csv

Each output double has its genes ordered (gene1 < gene2, systematic sort).

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover.py
"""
import os
import os.path as osp
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
TRIPLES_CSV = osp.join(
    RESULTS_DIR, "inference_3", "top_k_constructible_panel12_k200.csv"
)
FULL_DOUBLES = 66  # C(12,2)


def subpairs(triple: frozenset) -> set:
    """The three doubles (as ordered gene tuples) contained in a triple."""
    return {tuple(sorted(p)) for p in combinations(sorted(triple), 2)}


def greedy_set_cover(triples: list[frozenset]) -> list[tuple]:
    """Fewest doubles such that every triple contains >=1 chosen double."""
    candidates = set().union(*[subpairs(t) for t in triples])
    covers = {d: {i for i, t in enumerate(triples) if set(d).issubset(t)}
              for d in candidates}
    uncovered = set(range(len(triples)))
    chosen: list[tuple] = []
    while uncovered:
        best = max(candidates, key=lambda d: len(covers[d] & uncovered))
        chosen.append(best)
        uncovered -= covers[best]
    # order the final list by triple-yield (most-enabling doubles first)
    return sorted(chosen, key=lambda d: (-len(covers[d]), d))


def main() -> None:
    tri = pd.read_csv(TRIPLES_CSV)
    triples = [frozenset([r.gene1, r.gene2, r.gene3]) for _, r in tri.iterrows()]
    covers = {}  # recompute yield/members for the chosen doubles
    chosen = greedy_set_cover(triples)

    rows = []
    for rank, (g1, g2) in enumerate(chosen, start=1):
        enabled = [
            "+".join(sorted(t)) for t in triples if {g1, g2}.issubset(t)
        ]
        rows.append({
            "rank": rank,
            "gene1": g1,          # ordered: gene1 < gene2
            "gene2": g2,
            "n_triples_enabled": len(enabled),
            "triples_enabled": "; ".join(sorted(enabled)),
        })
    out_df = pd.DataFrame(rows)

    out = osp.join(RESULTS_DIR, "optimized_doubles_setcover_panel12.csv")
    out_df.to_csv(out, index=False)

    n_cover = len(chosen)
    print(f"constructible triples          : {len(triples)}")
    print(f"full C(12,2)                    : {FULL_DOUBLES}")
    print(f"optimized doubles (set-cover)   : {n_cover}")
    print(f"savings vs full                 : {FULL_DOUBLES - n_cover} "
          f"({100 * (FULL_DOUBLES - n_cover) / FULL_DOUBLES:.0f}% fewer builds)")
    print(f"\nSaved: {out}\n")
    print(out_df[["rank", "gene1", "gene2", "n_triples_enabled"]].to_string(index=False))


if __name__ == "__main__":
    main()
