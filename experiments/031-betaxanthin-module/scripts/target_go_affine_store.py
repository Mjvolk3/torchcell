# experiments/031-betaxanthin-module/scripts/target_go_affine_store.py
# [[experiments.031-betaxanthin-module.scripts.target_go_affine_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_go_affine_store

r"""The GO multi-hot as affine columns on the targets, over every annotated gene.

``target_go_models.py`` read a perceptron over the propagated GO multi-hot at 0.200 on
betaxanthin, above every transformer arm, whose backbone reads the pathway multi-hot and
ProtT5 and never saw GO. The essentiality study's ``go_affine_store.py`` put GO in the
affine slot (round 50) over the 1,121 labeled genes; the target screens are genome-wide, so
these stores cover every gene of the ``go_term_hot`` store (the trainer reads 0 for a gene
a store does not cover, and a gene with no annotation reads 0 either way). Two stores per
target, in the solve store's shape and read through ``ESS_SOLVE_TABLE`` with
``--ess-solve-features pfba``, where the trainer's $\log(1+|v|)$ maps a 0/1 column to
0/0.693:

* ``ess_solve_encodings_<stem>_go.npz``: ``v_deletion`` is the GO multi-hot (2,362 terms),
  ``v_wild_type`` the zero row (no deletion, no annotation lit);
* ``ess_solve_encodings_<stem>_gopfba.npz``: the HiGHS parsimonious deletion vector of the
  target store with the GO multi-hot appended; a gene the flux store does not cover (outside
  the GEM) carries the wild-type flux row, the flux a deletion of no metabolic gene leaves.

Writes ``results/target_go_affine_store_<target>.json``. Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_go_affine_store.py --target betaxanthin
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
import torch
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
GO_DIR = osp.join(DATA_ROOT, "data/scerevisiae/go_term_hot/processed")
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
from ess_reference_flux_table import StoreProvenance  # noqa: E402

TARGETS = {
    "betaxanthin": {"flux_store": "YPD_bx_pfba", "out_stem": "YPD_bx", "study": "031-betaxanthin-module"},
    "amino_acids": {"flux_store": "SM_gem", "out_stem": "SM_gem", "study": "032-amino-acid-module"},
}


class GOStoreProvenance(StoreProvenance):
    """The store record plus the GO columns' origin."""

    target: str
    go_store: str
    n_go_terms: int
    n_genes_annotated: int
    flux_store: str | None
    n_genes_with_flux_row: int
    column_layout: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    spec = TARGETS[args.target]
    results_dir = args.results_dir or osp.join(EXPERIMENT_ROOT, spec["study"], "results")
    started = time.time()

    data, _ = torch.load(osp.join(GO_DIR, "go.pt"), weights_only=False)
    with open(osp.join(GO_DIR, "go.terms.json")) as fh:
        meta = json.load(fh)
    go_genes = [str(g) for g in data.id]
    hot = data.embeddings["go"].numpy().astype(np.float32)
    terms = [t["term_id"] for t in meta["terms"]]
    assert hot.shape == (len(go_genes), len(terms))
    annotated = hot.sum(axis=1) > 0

    flux_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{spec['flux_store']}.npz")
    flux = np.load(flux_path, allow_pickle=False)
    flux_provenance = json.loads(str(flux["provenance"]))
    flux_genes = [str(g) for g in flux["genes"]]
    flux_rows = {g: flux["v_deletion"][i].astype(np.float32) for i, g in enumerate(flux_genes)}
    flux_wild = flux["v_wild_type"].astype(np.float32)
    rxn_ids = [str(r) for r in flux["rxn_ids"]]
    n_flux_row = sum(1 for g in go_genes if g in flux_rows)

    common = dict(
        gem=flux_provenance["gem"],
        growth_fraction=flux_provenance.get("growth_fraction", 1.0),
        flux_scale=flux_provenance["flux_scale"],
        solver="none (annotation) / the flux store's",
        n_nonoptimal_zero_flux=0,
        medium=flux_provenance["medium"],
        media_report=flux_provenance["media_report"],
        split_file=flux_provenance["split_file"],
        wild_type_growth=flux_provenance["wild_type_growth"],
        wild_type_key=flux_provenance["wild_type_key"],
        n_sets=len(go_genes) + 1,
        n_lethal_zero_flux=0,
        n_genes_outside_the_gem=len(go_genes) - n_flux_row,
        background_genes=list(flux_provenance["background_genes"]),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        target=args.target,
        go_store=GO_DIR,
        n_go_terms=len(terms),
        n_genes_annotated=int(annotated.sum()),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    written = {}
    for suffix, with_flux in (("go", False), ("gopfba", True)):
        if with_flux:
            v_del = np.stack([np.concatenate([flux_rows.get(g, flux_wild), hot[i]]) for i, g in enumerate(go_genes)])
            v_wt = np.concatenate([flux_wild, np.zeros(len(terms), np.float32)])
            names = rxn_ids + terms
            layout = "HiGHS parsimonious deletion flux (wild-type row outside the GEM) | GO multi-hot"
        else:
            v_del = hot
            v_wt = np.zeros(len(terms), np.float32)
            names = terms
            layout = "GO multi-hot (propagated, 2,362 terms with ten or more genes)"
        provenance = GOStoreProvenance(
            **common,
            method=(
                f"the propagated GO multi-hot of the go_term_hot store as affine columns over every "
                f"gene it lists{' appended to the HiGHS parsimonious deletion vector of ' + spec['flux_store'] if with_flux else ''}; "
                "written by target_go_affine_store.py"
            ),
            flux_sha256=hashlib.sha256(v_del.astype(np.float32).tobytes()).hexdigest(),
            flux_store=flux_path if with_flux else None,
            n_genes_with_flux_row=n_flux_row if with_flux else 0,
            column_layout=layout,
        )
        path = osp.join(args.out_dir, f"ess_solve_encodings_{spec['out_stem']}_{suffix}.npz")
        assert not osp.exists(path), f"{path} exists; stores are never overwritten"
        np.savez(
            path,
            genes=np.array(go_genes),
            v_deletion=v_del.astype(np.float32),
            v_wild_type=v_wt.astype(np.float32),
            rxn_ids=np.array(names),
            provenance=np.array(provenance.model_dump_json()),
        )
        print(f"{path}: {v_del.shape}", flush=True)
        written[suffix] = path
    summary = {
        "stores": written,
        "target": args.target,
        "n_genes": len(go_genes),
        "n_genes_annotated": int(annotated.sum()),
        "n_go_terms": len(terms),
        "n_genes_with_flux_row": n_flux_row,
        "flux_store": flux_path,
        "seconds": time.time() - started,
    }
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_go_affine_store_{args.target}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "stores"}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
