# experiments/028-gene-essentiality/scripts/fba_ratio_all_gem_genes.py
# [[experiments.028-gene-essentiality.scripts.fba_ratio_all_gem_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/fba_ratio_all_gem_genes

r"""Four-media FBA growth ratios for EVERY yeast-GEM gene, not only the labeled ones.

``fba_essentiality_baseline.py`` writes the four-media table the essentiality incumbent
reads (shipped, SM, SC, YPD), but only for the 1,121 genes with an essentiality label.
The trainer reads 1.0 (no growth defect) for any gene a table does not cover, so the
betaxanthin and amino-acid cells that carried "four media" saw wild-type columns for
every gene outside the essentiality labels (785 of the pool's 4,678 strains covered).
This script runs the same single-gene deletion on the same medium models, same growth
floor and same 1e-3 rounding, over every gene in the model, so the incumbent's recipe
can be carried to the two targets with its ratio columns intact. A gene outside the GEM
has no reaction to remove and keeps 1.0, which is the FBA answer, not a fill.

The labeled genes are checked against the corrected-YPD table
(``fba_essentiality_per_gene_nuc.csv``). A cell that disagrees by more than the
rounding step is re-solved alone on a freshly built medium model with presolve on, and
the fresh solve must equal the new value: on 2026-09-27 two thiamine genes on the
shipped medium (THI6 YPL214C lethal, THI4 YGR144W wild type) disagreed with both older
tables, and fresh solves reproduced the new values twice, so the older tables carried
solver state from their deletion scans. Writes ``results/fba_ratio_all_gem_genes.csv`` (gene, shipped, SM, SC,
YPD) and a JSON summary. Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/fba_ratio_all_gem_genes.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd
from cobra.flux_analysis import single_gene_deletion
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
from fba_essentiality_baseline import GROWTH_FLOOR, medium_model  # noqa: E402

RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
MEDIA = ("shipped", "SM", "SC", "YPD")
#: The corrected-YPD table (round 112: YPD-approx carries adenine and uracil). The
#: original ``fba_essentiality_per_gene.csv`` predates that recipe and differs on 36
#: labeled genes' YPD column, all from the nucleobases, so it is not the reference.
EXISTING = osp.join(RESULTS, "fba_essentiality_per_gene_nuc.csv")


def main() -> None:
    started = time.time()
    per_gene: dict[str, dict[str, float]] = {}
    summary: dict[str, dict[str, float]] = {}
    for key in MEDIA:
        model, name, n_bounds = medium_model(key)
        t0 = time.time()
        wt = float(model.slim_optimize())
        assert wt > GROWTH_FLOOR, f"wild type does not grow on {key}: {wt}"
        genes = sorted(g.id for g in model.genes)
        deletions = single_gene_deletion(model, gene_list=genes, processes=8)
        n_zero = 0
        for ids, growth in zip(deletions["ids"], deletions["growth"], strict=True):
            (gene,) = tuple(ids)
            g = (
                0.0
                if (growth is None or np.isnan(growth) or growth < GROWTH_FLOOR)
                else float(growth)
            )
            per_gene.setdefault(gene, {})[key] = round(g / wt, 3)
            n_zero += int(g == 0.0)
        summary[key] = {
            "wild_type_growth": wt,
            "n_genes": len(genes),
            "n_lethal": n_zero,
            "n_exchange_bounds": n_bounds,
            "seconds": time.time() - t0,
        }
        print(f"[{key}] {name}: WT {wt:.4f}/h, {len(genes)} deletions, {n_zero} lethal", flush=True)
    table = pd.DataFrame.from_dict(per_gene, orient="index").sort_index()[list(MEDIA)]
    table.index.name = "gene"
    old = pd.read_csv(EXISTING).set_index("gene")[list(MEDIA)]
    shared = sorted(set(old.index) & set(table.index))
    delta = (old.loc[shared] - table.loc[shared]).abs()
    diff = float(delta.to_numpy().max())
    mismatch = {
        m: sorted(delta.index[delta[m] > 1e-3 + 1e-9].tolist()) for m in MEDIA
    }
    out_csv = osp.join(RESULTS, "fba_ratio_all_gem_genes.csv")
    table.to_csv(out_csv)
    if diff > 1e-3 + 1e-9:
        for m, genes in mismatch.items():
            for g in genes[:10]:
                print(f"  MISMATCH {m} {g}: existing {old.loc[g, m]} new {table.loc[g, m]}")
            print(f"  {m}: {len(genes)} labeled genes disagree", flush=True)
    confirmed: list[dict[str, object]] = []
    for m, genes_m in mismatch.items():
        for g in genes_m:
            fresh_model, _, _ = medium_model(m)
            fresh_model.solver.configuration.presolve = True
            wt_fresh = float(fresh_model.slim_optimize())
            with fresh_model:
                fresh_model.genes.get_by_id(g).knock_out()
                v = fresh_model.slim_optimize(error_value=0.0)
            fresh = round((0.0 if v < GROWTH_FLOOR else v) / wt_fresh, 3)
            assert abs(fresh - float(table.loc[g, m])) <= 1e-3 + 1e-9, (
                f"{m} {g}: scan {table.loc[g, m]}, fresh solve {fresh}, "
                f"existing {old.loc[g, m]}"
            )
            confirmed.append(
                {"medium": m, "gene": g, "new": float(table.loc[g, m]), "fresh": fresh,
                 "existing": float(old.loc[g, m])}
            )
    print(f"  {len(confirmed)} disagreeing cells confirmed by fresh solves", flush=True)
    out = {
        "generated_by": "experiments/028-gene-essentiality/scripts/fba_ratio_all_gem_genes.py",
        "csv": out_csv,
        "n_genes": int(len(table)),
        "media": summary,
        "n_checked_against_existing": len(shared),
        "max_abs_difference_on_checked": diff,
        "disagreements_confirmed_by_fresh_solve": confirmed,
        "n_below_0_99": {m: int((table[m] < 0.99).sum()) for m in MEDIA},
        "seconds": time.time() - started,
    }
    with open(osp.join(RESULTS, "fba_ratio_all_gem_genes.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "media"}, indent=1))


if __name__ == "__main__":
    main()
