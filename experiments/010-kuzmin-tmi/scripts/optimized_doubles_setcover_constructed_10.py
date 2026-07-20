# experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover_constructed_10.py
# [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover_constructed_10
"""Minimum doubles set-cover restricted to the 10 properly-constructed genes.

Same greedy set-cover as optimized_doubles_setcover.py, but over ONLY the top-k
constructible triples whose three genes are all among the 10 genes the wet-lab
plate actually built (inference_3 panel-12 minus the dropped YIL174W and
LCL2/YLR104W). Answers: which few doubles, from the strains we now have, enable
the most high-ranking triples.

Input : results/inference_3/top_k_constructible_panel12_k200.csv (52; 31 within-10)
Output: results/optimized_doubles_setcover_constructed_10.csv

Genes within each double are ordered (gene1 < gene2, systematic sort).

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover_constructed_10.py
"""
import os
import os.path as osp
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
TRIPLES_CSV = osp.join(RESULTS_DIR, "inference_3", "top_k_constructible_panel12_k200.csv")

# 10 properly-constructed genes = inference_3 panel-12 minus YIL174W and YLR104W.
TEN = {
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
    "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W",
}
FULL_DOUBLES = len(TEN) * (len(TEN) - 1) // 2  # C(10,2) = 45


def greedy_set_cover(triples: list[frozenset]) -> tuple[list[tuple], dict]:
    candidates = set().union(
        *[{tuple(sorted(p)) for p in combinations(sorted(t), 2)} for t in triples]
    )
    covers = {d: {i for i, t in enumerate(triples) if set(d).issubset(t)}
              for d in candidates}
    uncovered = set(range(len(triples)))
    chosen: list[tuple] = []
    while uncovered:
        best = max(candidates, key=lambda d: len(covers[d] & uncovered))
        chosen.append(best)
        uncovered -= covers[best]
    return sorted(chosen, key=lambda d: (-len(covers[d]), d)), covers


def main() -> None:
    tri = pd.read_csv(TRIPLES_CSV)
    triples = [
        frozenset([r.gene1, r.gene2, r.gene3])
        for _, r in tri.iterrows()
        if {r.gene1, r.gene2, r.gene3}.issubset(TEN)
    ]
    chosen, _ = greedy_set_cover(triples)

    rows = []
    for rank, (g1, g2) in enumerate(chosen, start=1):
        enabled = ["+".join(sorted(t)) for t in triples if {g1, g2}.issubset(t)]
        rows.append({
            "rank": rank,
            "gene1": g1,
            "gene2": g2,
            "n_triples_enabled": len(enabled),
            "triples_enabled": "; ".join(sorted(enabled)),
        })
    out_df = pd.DataFrame(rows)
    out = osp.join(RESULTS_DIR, "optimized_doubles_setcover_constructed_10.csv")
    out_df.to_csv(out, index=False)

    n = len(chosen)
    print(f"within-10 top-k triples        : {len(triples)}")
    print(f"full C(10,2)                    : {FULL_DOUBLES}")
    print(f"optimized doubles (set-cover)   : {n}")
    print(f"savings vs full                 : {FULL_DOUBLES - n} "
          f"({100 * (FULL_DOUBLES - n) / FULL_DOUBLES:.0f}% fewer)")
    print(f"\nSaved: {out}\n")
    print(out_df[["rank", "gene1", "gene2", "n_triples_enabled"]].to_string(index=False))


if __name__ == "__main__":
    main()
