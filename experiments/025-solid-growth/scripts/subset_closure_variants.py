# experiments/025-solid-growth/scripts/subset_closure_variants.py
# [[experiments.025-solid-growth.training-plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/subset_closure_variants

"""Size the three ways a double can be "inside the triples", so the pool can be chosen.

S3 (``subset_definitions.py``) keeps a double when its gene PAIR lies inside some triple.
Two looser rules were asked for on 2026-09-16: keep a double when at least one of its genes
appears in any triple, or when both do (in any triples, not necessarily the same one). This
scans the 13,142,648 doubles once and counts all three, plus how many of each carry a
SynthLethDB record, and writes the index lists for the two new rules so a config can point
at them.

Outputs (experiments/025-solid-growth/results/):
- subset_closure_variants_summary.json
- subset_S3any_indices.json.gz   (singles + triples + doubles with >= 1 gene in the triple gene set)
- subset_S3both_indices.json.gz  (singles + triples + doubles with both genes in the triple gene set)
"""

import gzip
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from subset_definitions import (  # noqa: E402
    BUILD,
    RESULTS_DIR,
    load_triple_gene_sets,
    scan_double_pairs,
)


def main() -> None:
    with open(osp.join(BUILD, "perturbation_count_index.json")) as f:
        count_index = json.load(f)
    idx_single, idx_double, idx_triple = (
        count_index["1"],
        count_index["2"],
        count_index["3"],
    )
    triple_genes = load_triple_gene_sets()
    universe = {g for gs in triple_genes.values() for g in gs}
    closure_pairs = set()
    for gs in triple_genes.values():
        a, b, c = gs
        closure_pairs |= {frozenset((a, b)), frozenset((a, c)), frozenset((b, c))}
    double_pairs = scan_double_pairs(idx_double)
    any_gene = sorted(i for i, p in double_pairs.items() if p & universe)
    both_genes = sorted(i for i, p in double_pairs.items() if p <= universe)
    pair_in_triple = sorted(i for i, p in double_pairs.items() if p in closure_pairs)
    base = sorted(idx_single + idx_triple)
    summary = {
        "n_triple_genes": len(universe),
        "n_singles": len(idx_single),
        "n_triples": len(idx_triple),
        "n_doubles": len(idx_double),
        "doubles_any_gene_in_triples": len(any_gene),
        "doubles_both_genes_in_triples": len(both_genes),
        "doubles_pair_in_some_triple": len(pair_in_triple),
        "pool_S3any": len(base) + len(any_gene),
        "pool_S3both": len(base) + len(both_genes),
        "pool_S3": len(base) + len(pair_in_triple),
        "note": (
            "S3 (pair inside some triple) is the committed closure arm; S3both and S3any "
            "are the looser rules asked for 2026-09-16. Epoch cost scales with the pool; "
            "an S0 epoch (376,732 records) is ~12 min on four A100s."
        ),
    }
    for name, doubles in (("S3any", any_gene), ("S3both", both_genes)):
        with gzip.open(osp.join(RESULTS_DIR, f"subset_{name}_indices.json.gz"), "wt") as f:
            json.dump(sorted(base + doubles), f)
    with open(osp.join(RESULTS_DIR, "subset_closure_variants_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
