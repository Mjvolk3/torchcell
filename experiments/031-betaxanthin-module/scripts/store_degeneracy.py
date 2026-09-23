# experiments/031-betaxanthin-module/scripts/store_degeneracy.py
# [[experiments.031-betaxanthin-module.scripts.store_degeneracy]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/store_degeneracy

r"""How many no-effect deletions carry a vector that differs from the wild type's, per store.

A deletion that leaves growth unchanged and closes no reaction the wild type uses has the
wild-type flux among its parsimonious optima; a store built by re-solving every gene
lands on whatever vertex of that degenerate optimum the solver returns, so its vector can
differ from the wild type's for a gene that does nothing. This measures that per store,
in the encoding the affine reads (``log(1+|v|)``): over the genes whose growth ratio is 1,
the count whose vector differs from the wild-type row by more than 1e-6 in relative
$\ell_2$, with the median and ninetieth percentile of that distance. Writes
``results/store_degeneracy.json`` in the betaxanthin study (both targets' stores are
listed). Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/store_degeneracy.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
RESULTS = osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "results")
#: (store stem, flux columns to read, growth-ratio source, label)
STORES = [
    ("YPD_bx", 4177, "stats", "betaxanthin, parsimonious, cobra pFBA through GLPK"),
    ("YPD_bx_pfba", None, "growth_ratio", "betaxanthin, parsimonious, HiGHS two-stage"),
    ("YPD_bx_moma", None, "growth_ratio", "betaxanthin, linear MOMA"),
    ("SM_gem", None, "stats", "amino acids, parsimonious, HiGHS two-stage"),
    ("SM_gem_moma", None, "growth_ratio", "amino acids, linear MOMA"),
]


class StoreDegeneracy(BaseModel):
    stem: str
    label: str
    n_genes: int
    n_no_effect: int
    n_no_effect_vector_differs: int
    median_rel_l2_log1p: float
    p90_rel_l2_log1p: float


def main() -> None:
    rows: list[StoreDegeneracy] = []
    for stem, n_cols, ratio_source, label in STORES:
        z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{stem}.npz"), allow_pickle=False)
        v = z["v_deletion"].astype(float)
        wt = z["v_wild_type"].astype(float)
        if n_cols is not None:
            v, wt = v[:, :n_cols], wt[:n_cols]
        if ratio_source == "stats":
            names = [str(s) for s in z["stat_names"]]
            ratio = z["gene_flux_stats"][:, names.index("deletion_growth_ratio")].astype(float)
        else:
            ratio = z["growth_ratio"].astype(float)
        enc, ew = np.log1p(np.abs(v)), np.log1p(np.abs(wt))
        rel = np.linalg.norm(enc - ew, axis=1) / max(float(np.linalg.norm(ew)), 1e-12)
        same = np.abs(ratio - 1.0) < 1e-6
        rows.append(
            StoreDegeneracy(
                stem=stem,
                label=label,
                n_genes=int(v.shape[0]),
                n_no_effect=int(same.sum()),
                n_no_effect_vector_differs=int((rel[same] > 1e-6).sum()),
                median_rel_l2_log1p=float(np.median(rel[same])),
                p90_rel_l2_log1p=float(np.quantile(rel[same], 0.9)),
            )
        )
        print(rows[-1].model_dump_json())
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "store_degeneracy.json")
    with open(out, "w") as fh:
        json.dump([r.model_dump() for r in rows], fh, indent=2)
    print(out)


if __name__ == "__main__":
    main()
