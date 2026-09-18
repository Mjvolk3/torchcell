# experiments/028-gene-essentiality/scripts/go_affine_store.py
# [[experiments.028-gene-essentiality.scripts.go_affine_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/go_affine_store

"""The GO multi-hot as affine columns: two stores in the solve store's shape.

The transformer reads the GO multi-hot on the node at a cost of about 0.04 whatever
else is on the node (round 26 beside ProtT5, round 28 at width 24, round 47 beside the
pathway block and the layer-free solve), while the solve vector reads its full number
on the logit through the FBA affine and nothing extra on the node (rounds 33, 43, 44).
Whether GO is the same case, an input this backbone's pooled path cannot carry that the
logit can, is answerable with no new construction: the affine reads any store written
in the shape of ``ess_reference_flux_table.py`` (``genes``, ``v_deletion``,
``v_wild_type``, ``rxn_ids``, ``provenance``) through ``ESS_SOLVE_TABLE`` and encodes
it as $\\log(1 + |v|)$, which maps a 0/1 column to 0/0.693.

Two stores over the 1,121 labeled genes of the released split, the genes the pFBA store
covers (every training and held-out sample is one of them):

* ``ess_solve_encodings_YPD_go.npz``: ``v_deletion`` is the propagated GO multi-hot
  (the ``go`` registry embedding, 2,362 terms with ten or more genes), so the affine
  reads GO alone beside the four media ratios;
* ``ess_solve_encodings_YPD_gopfba.npz``: the pFBA deletion vector (raw flux, so the
  trainer's $\\log(1 + |v|)$ is the round-33 encoding) with the GO multi-hot appended,
  the one-model GO + layer-free form on the logit.

``v_wild_type`` is the pFBA wild-type vector padded with zeros over the GO columns (no
statistics block, so only the ``pfba`` feature kind reads the stores). Writes
``results/go_affine_store.json`` with the column counts and coverage.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SCRIPTS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from graph_baselines import embedding  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PFBA_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD.npz")


def main() -> None:
    pfba = np.load(PFBA_STORE, allow_pickle=False)
    genes = [str(g) for g in pfba["genes"]]
    v = pfba["v_deletion"].astype(np.float32)
    v_wt = pfba["v_wild_type"].astype(np.float32)
    rxn_ids = [str(r) for r in pfba["rxn_ids"]]
    go = embedding("go", genes).astype(np.float32)
    assert go.shape[0] == len(genes) and set(np.unique(go)) <= {0.0, 1.0}, go.shape
    n_terms = go.shape[1]
    with open(
        osp.join(DATA_ROOT, "data/scerevisiae/go_term_hot/processed/go.terms.json")
    ) as fh:
        terms = json.load(fh)
    assert terms["n_columns"] == n_terms, (terms["n_columns"], n_terms)
    go_ids = [t["term_id"] for t in terms["terms"]]
    covered = int((go.sum(axis=1) > 0).sum())
    pfba_provenance = json.loads(str(pfba["provenance"]))

    out: dict[str, dict[str, object]] = {}
    for suffix, block, names, wt in (
        ("go", go, go_ids, np.zeros(n_terms, dtype=np.float32)),
        (
            "gopfba",
            np.concatenate([v, go], axis=1),
            rxn_ids + go_ids,
            np.concatenate([v_wt, np.zeros(n_terms, dtype=np.float32)]),
        ),
    ):
        path = osp.join(TABLE_DIR, f"ess_solve_encodings_YPD_{suffix}.npz")
        provenance = {
            "method": (
                "the propagated GO multi-hot (registry embedding `go`, SGD annotations "
                "propagated over is_a and part_of, terms with >= 10 genes) as affine "
                "columns over the 1,121 labeled genes"
                + (
                    "; appended to the pFBA deletion vector of ess_solve_encodings_YPD.npz"
                    if suffix == "gopfba"
                    else ""
                )
                + "; written by go_affine_store.py"
            ),
            "source_store": PFBA_STORE,
            "source_store_provenance": pfba_provenance,
            "go_terms_json": terms["model_name"],
            "n_go_columns": n_terms,
            "n_genes_with_go_annotation": covered,
            "n_columns": int(block.shape[1]),
        }
        np.savez(
            path,
            genes=np.array(genes),
            v_deletion=block.astype(np.float32),
            v_wild_type=wt.astype(np.float32),
            rxn_ids=np.array(names),
            provenance=np.array(json.dumps(provenance)),
        )
        out[suffix] = {"path": path, **provenance}
        print(f"{path}: {block.shape}")
    result = osp.join(RESULTS_DIR, "go_affine_store.json")
    with open(result, "w") as fh:
        json.dump(out, fh, indent=2)
    print(result)


if __name__ == "__main__":
    main()
