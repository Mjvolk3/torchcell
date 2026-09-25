# experiments/031-betaxanthin-module/scripts/target_store_concat.py
# [[experiments.031-betaxanthin-module.scripts.target_store_concat]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_store_concat

r"""Append one affine store's columns to another's, gene by gene.

The affine reads one row per deleted gene, so two stores over different gene sets are
joined on the base store's genes: a gene the appended store does not cover carries that
store's wild-type row (the vector a deletion of no covered gene leaves), as the GO store
did for the flux block. The wild-type rows concatenate, the column names concatenate,
and the provenance records both sources with their sha256 so the joined store traces to
the two it was built from. Stores are never overwritten.

Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_store_concat.py \
        --base YPD_bx_gopfba --append YPD_bx_mdfcap100polymean --out YPD_bx_gopfbapoly \
        --study 031-betaxanthin-module
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
from ess_reference_flux_table import StoreProvenance  # noqa: E402


class ConcatStoreProvenance(StoreProvenance):
    """The base store's record plus the appended store's origin."""

    base_store: str
    base_sha256: str
    appended_store: str
    appended_sha256: str
    n_genes_with_appended_row: int
    column_layout: str


def load(stem: str) -> dict:
    path = osp.join(TABLE_DIR, f"ess_solve_encodings_{stem}.npz")
    z = np.load(path, allow_pickle=False)
    return {
        "path": path,
        "genes": [str(g) for g in z["genes"]],
        "v_deletion": z["v_deletion"].astype(np.float32),
        "v_wild_type": z["v_wild_type"].astype(np.float32),
        "rxn_ids": [str(r) for r in z["rxn_ids"]],
        "provenance": json.loads(str(z["provenance"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out-dir", default=TABLE_DIR)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    started = time.time()
    base, app = load(args.base), load(args.append)
    app_rows = {g: app["v_deletion"][i] for i, g in enumerate(app["genes"])}
    n_app = sum(1 for g in base["genes"] if g in app_rows)
    v_del = np.stack(
        [
            np.concatenate([base["v_deletion"][i], app_rows.get(g, app["v_wild_type"])])
            for i, g in enumerate(base["genes"])
        ]
    )
    v_wt = np.concatenate([base["v_wild_type"], app["v_wild_type"]])
    names = base["rxn_ids"] + [f"{args.append}:{r}" for r in app["rxn_ids"]]
    bp = base["provenance"]
    keep = {k: bp[k] for k in StoreProvenance.model_fields if k in bp}
    keep.update(
        method=(
            f"the {args.base} store with the {args.append} store's columns appended per gene "
            f"(the appended store's wild-type row for a gene it does not cover); written by "
            "target_store_concat.py"
        ),
        flux_sha256=hashlib.sha256(v_del.tobytes()).hexdigest(),
        n_sets=len(base["genes"]) + 1,
    )
    provenance = ConcatStoreProvenance(
        **keep,
        base_store=base["path"],
        base_sha256=hashlib.sha256(base["v_deletion"].tobytes()).hexdigest(),
        appended_store=app["path"],
        appended_sha256=hashlib.sha256(app["v_deletion"].tobytes()).hexdigest(),
        n_genes_with_appended_row=n_app,
        column_layout=f"{args.base} ({len(base['rxn_ids'])}) | {args.append} ({len(app['rxn_ids'])})",
    )
    path = osp.join(args.out_dir, f"ess_solve_encodings_{args.out}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    np.savez(
        path,
        genes=np.array(base["genes"]),
        v_deletion=v_del.astype(np.float32),
        v_wild_type=v_wt.astype(np.float32),
        rxn_ids=np.array(names),
        provenance=np.array(provenance.model_dump_json()),
    )
    summary = {
        "store": path,
        "base": base["path"],
        "append": app["path"],
        "n_genes": len(base["genes"]),
        "n_genes_with_appended_row": n_app,
        "n_columns": int(v_del.shape[1]),
        "column_layout": provenance.column_layout,
        "seconds": time.time() - started,
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_store_concat.py",
    }
    results_dir = osp.join(EXPERIMENT_ROOT, args.study, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_store_concat_{args.out}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"{path}: {v_del.shape}, {n_app} genes with an appended row")
    print(out)


if __name__ == "__main__":
    main()
