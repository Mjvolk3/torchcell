# experiments/028-gene-essentiality/scripts/store_stats.py
# [[experiments.028-gene-essentiality.scripts.store_stats]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/store_stats

"""What the four solve stores carry: the flux vectors' size, support and clamp use.

Rounds 36 and 38 trained the layer-free cell over four stores of the same growth
program's optimal face and read one number from three of them (pFBA 0.912, interior
point 0.911, PDLP 0.907) and a lower one from the growth program's dual-simplex vertex
(0.870, two early peaks). This script describes the vectors themselves, per store, over
the 1,121 labeled genes: the total absolute flux, the number of reactions carrying flux,
the largest flux, the number of reactions at the layer's clamp (|v| = 100 mmol/gDW/h,
where a loop can spin), the growth, and the distance from the pFBA vector in the head's
encoding. It is a description, not a mechanism: whether extremal loop flux is what the
head cannot read is the hypothesis the numbers are read against.

Writes ``results/store_stats.json``.
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
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
FLUX_SCALE = 100.0

STORES = {
    "pFBA (two-stage, scipy HiGHS)": "ess_solve_encodings_YPD.npz",
    "dual simplex (vertex)": "ess_solve_encodings_YPD_simplex.npz",
    "interior point, no crossover": "ess_solve_encodings_YPD_ipm.npz",
    "PDLP, no crossover": "ess_solve_encodings_YPD_pdlp.npz",
}
REFERENCE = "pFBA (two-stage, scipy HiGHS)"


class StoreStats(BaseModel):
    """One store's vectors over the labeled genes."""

    store: str
    file: str
    n_genes: int
    n_zero_rows: int
    sum_abs_flux_median: float
    sum_abs_flux_mean: float
    n_carrying_median: float
    max_abs_flux_median: float
    n_at_clamp_median: float
    n_at_clamp_mean: float
    frac_genes_with_any_clamp: float
    growth_median: float
    log1p_rel_l2_to_pfba_median: float
    frac_moved_gt_0_01_vs_pfba_mean: float
    wild_type_sum_abs_flux: float
    wild_type_n_at_clamp: int


def main() -> None:
    payloads = {
        name: np.load(osp.join(TABLE_DIR, f), allow_pickle=False)
        for name, f in STORES.items()
    }
    genes = [str(g) for g in payloads[REFERENCE]["genes"]]
    for name, p in payloads.items():
        assert [str(g) for g in p["genes"]] == genes, name
    rxn_ids = [str(r) for r in payloads[REFERENCE]["rxn_ids"]]
    biomass = rxn_ids.index("r_2111")
    ref = payloads[REFERENCE]["v_deletion"].astype(np.float64)
    ref_enc = np.log1p(np.abs(ref))
    out = []
    for name, p in payloads.items():
        v = p["v_deletion"].astype(np.float64)
        a = np.abs(v)
        enc = np.log1p(a)
        zero_rows = int((a.sum(axis=1) == 0).sum())
        at_clamp = (a >= FLUX_SCALE - 1e-6).sum(axis=1)
        d = np.linalg.norm(enc - ref_enc, axis=1) / np.maximum(
            np.linalg.norm(ref_enc, axis=1), 1e-12
        )
        wild = np.abs(p["v_wild_type"].astype(np.float64))
        out.append(
            StoreStats(
                store=name,
                file=osp.basename(STORES[name]),
                n_genes=len(genes),
                n_zero_rows=zero_rows,
                sum_abs_flux_median=float(np.median(a.sum(axis=1))),
                sum_abs_flux_mean=float(a.sum(axis=1).mean()),
                n_carrying_median=float(np.median((a > 1e-6).sum(axis=1))),
                max_abs_flux_median=float(np.median(a.max(axis=1))),
                n_at_clamp_median=float(np.median(at_clamp)),
                n_at_clamp_mean=float(at_clamp.mean()),
                frac_genes_with_any_clamp=float((at_clamp > 0).mean()),
                growth_median=float(np.median(v[:, biomass])),
                log1p_rel_l2_to_pfba_median=float(np.median(d)),
                frac_moved_gt_0_01_vs_pfba_mean=float(
                    (np.abs(enc - ref_enc) > 0.01).mean()
                ),
                wild_type_sum_abs_flux=float(wild.sum()),
                wild_type_n_at_clamp=int((wild >= FLUX_SCALE - 1e-6).sum()),
            )
        )
        print(out[-1].model_dump_json())
    path = osp.join(RESULTS_DIR, "store_stats.json")
    with open(path, "w") as fh:
        json.dump(
            {
                "stores": [s.model_dump() for s in out],
                "flux_scale": FLUX_SCALE,
                "biomass": "r_2111",
            },
            fh,
            indent=2,
        )
    print(path)


if __name__ == "__main__":
    main()
