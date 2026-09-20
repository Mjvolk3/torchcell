# experiments/028-gene-essentiality/scripts/declamp_store.py
# [[experiments.028-gene-essentiality.scripts.declamp_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/declamp_store

r"""A store with its clamp-carrying columns dropped, to test what costs the vertex 0.07.

Section \ref{sec:store-discriminability} measured that the dual-simplex vertex store is
the only one of the four whose ten widest columns all touch the layer's $\pm100$ clamp,
where a loop can spin, and that it has the fewest columns which rank genes on their own.
The reading was left as a hypothesis, because four stores are an association: what costs
the vertex store 0.07 AUROC is loop-carrying columns at the clamp.

This writes the intervention that tests it. Every reaction whose flux reaches the clamp
on any genotype is zeroed in the copy, for one store and for the pFBA store as the
control, keeping the shape and the row order so ``ESS_SOLVE_TABLE`` reads it unchanged.
If the hypothesis holds, the vertex store recovers toward the other three and pFBA, which
has few such columns, does not move.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/declamp_store.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
FLUX_SCALE = 100.0
#: source -> the copy written beside it. The vertex store is the case; pFBA is the control.
SOURCES = {
    "ess_solve_encodings_YPD_simplex.npz": "ess_solve_encodings_YPD_simplexdeclamp.npz",
    "ess_solve_encodings_YPD.npz": "ess_solve_encodings_YPD_declamp.npz",
}


#: The specificity control. Round 68 showed that dropping the vertex store's 429
#: clamp-carrying columns recovers it; that could be the clamp, or it could be that
#: dropping 429 wide columns of any kind helps. This drops the same NUMBER of columns
#: from the same store, taken in order of variance among the columns that never clamp,
#: so the only difference from the case is which columns go.
CONTROL_SOURCE = "ess_solve_encodings_YPD_simplex.npz"
CONTROL_DEST = "ess_solve_encodings_YPD_simplextopvar.npz"


def write_topvar() -> dict[str, object]:
    """The same store with as many non-clamp columns dropped as the clamp rule drops."""
    p = np.load(osp.join(TABLE_DIR, CONTROL_SOURCE), allow_pickle=False)
    arrays = {k: p[k] for k in p.files}
    v = arrays["v_deletion"].astype(np.float64)
    wt = arrays["v_wild_type"].astype(np.float64)
    clamp = (np.abs(v) >= FLUX_SCALE - 1e-6).any(axis=0) | (
        np.abs(wt) >= FLUX_SCALE - 1e-6
    )
    n_drop = int(clamp.sum())
    var = np.log1p(np.abs(v)).var(axis=0)
    var[clamp] = -1.0  # never take a clamp column; those are the case's cut
    drop = np.argsort(var)[::-1][:n_drop]
    v[:, drop] = 0.0
    wt[drop] = 0.0
    arrays["v_deletion"] = v.astype(arrays["v_deletion"].dtype)
    arrays["v_wild_type"] = wt.astype(arrays["v_wild_type"].dtype)
    np.savez_compressed(osp.join(TABLE_DIR, CONTROL_DEST), **arrays)
    enc = np.log1p(np.abs(v))
    return {
        "source": CONTROL_SOURCE,
        "written": CONTROL_DEST,
        "rule": "the n widest columns that never reach the clamp, n = the clamp count",
        "n_reactions": int(v.shape[1]),
        "n_columns_dropped": n_drop,
        "n_columns_varying_after": int((enc.var(axis=0) > 1e-12).sum()),
        "sum_abs_flux_median_after": float(np.median(np.abs(v).sum(axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="clamp",
        choices=("clamp", "topvar"),
        help=(
            "'clamp' writes the case and its pFBA control; 'topvar' writes the "
            "specificity control, the same number of non-clamp columns dropped."
        ),
    )
    args = parser.parse_args()
    if args.mode == "topvar":
        row = write_topvar()
        print(json.dumps(row))
        path = osp.join(RESULTS_DIR, "declamp_store_topvar.json")
        with open(path, "w") as fh:
            json.dump({"flux_scale": FLUX_SCALE, "stores": [row]}, fh, indent=2)
        print(path)
        return
    report = []
    for src, dst in SOURCES.items():
        p = np.load(osp.join(TABLE_DIR, src), allow_pickle=False)
        arrays = {k: p[k] for k in p.files}
        v = arrays["v_deletion"].astype(np.float64)
        wt = arrays["v_wild_type"].astype(np.float64)
        # A column is dropped when ANY genotype drives it to the clamp, the wild type
        # included, so the same columns vanish from every row and from the reference.
        clamp = (np.abs(v) >= FLUX_SCALE - 1e-6).any(axis=0) | (
            np.abs(wt) >= FLUX_SCALE - 1e-6
        )
        v[:, clamp] = 0.0
        wt[clamp] = 0.0
        arrays["v_deletion"] = v.astype(arrays["v_deletion"].dtype)
        arrays["v_wild_type"] = wt.astype(arrays["v_wild_type"].dtype)
        out = osp.join(TABLE_DIR, dst)
        np.savez_compressed(out, **arrays)
        enc = np.log1p(np.abs(v))
        report.append(
            {
                "source": src,
                "written": dst,
                "n_reactions": int(v.shape[1]),
                "n_columns_dropped": int(clamp.sum()),
                "n_columns_varying_after": int((enc.var(axis=0) > 1e-12).sum()),
                "sum_abs_flux_median_after": float(np.median(np.abs(v).sum(axis=1))),
            }
        )
        print(json.dumps(report[-1]))
    path = osp.join(RESULTS_DIR, "declamp_store.json")
    with open(path, "w") as fh:
        json.dump({"flux_scale": FLUX_SCALE, "stores": report}, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
