# experiments/028-gene-essentiality/scripts/pdlp_fill_store.py
# [[experiments.028-gene-essentiality.scripts.pdlp_fill_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/pdlp_fill_store

"""The PDLP $10^{-8}$ store with its uncertified rows filled from the looser solves.

Round 55 trained the layer-free cell over the PDLP store at KKT $10^{-8}$, where 320 of
the 1,121 programs were uncertified within the 120 s cap and sit as zero rows, and read
0.02 under the $10^{-4}$ and $10^{-6}$ stores (12 and 70 zero rows). Whether that cost
is the missing rows, which a second solve pass at a looser tolerance would recover, or
the $10^{-8}$ point itself, is answerable with a store that keeps every row PDLP
certified at $10^{-8}$ and takes the rest from the $10^{-6}$ store where that certified
the program and from the $10^{-4}$ store otherwise, the fallback a batched solver on the
card would run. A row is uncertified when its deletion vector is all zero and the gene
is not lethal at the looser tolerance (a lethal program has zero growth and a zero
vector under every solver, so its row is zero in every store and stays so).

Writes ``ess_solve_encodings_YPD_pdlp1e-8fill.npz`` in the solve store's shape and
``results/pdlp_fill_store.json`` with the counts per source.
"""

from __future__ import annotations

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
SOURCES = ["pdlp1e-8", "pdlp1e-6", "pdlp"]


def main() -> None:
    stores = {
        s: np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_YPD_{s}.npz"), allow_pickle=False)
        for s in SOURCES
    }
    genes = [str(g) for g in stores["pdlp1e-8"]["genes"]]
    for s in SOURCES[1:]:
        assert [str(g) for g in stores[s]["genes"]] == genes, s
    v = {s: stores[s]["v_deletion"].astype(np.float32) for s in SOURCES}
    zero = {s: ~np.any(v[s] != 0, axis=1) for s in SOURCES}
    filled = v["pdlp1e-8"].copy()
    source = np.full(len(genes), "pdlp1e-8", dtype=object)
    for i in range(len(genes)):
        if not zero["pdlp1e-8"][i]:
            continue
        for s in SOURCES[1:]:
            if not zero[s][i]:
                filled[i] = v[s][i]
                source[i] = s
                break
        else:
            source[i] = "zero_in_every_store"
    counts = {k: int((source == k).sum()) for k in SOURCES + ["zero_in_every_store"]}
    provenance = {
        "method": (
            "the PDLP KKT 1e-8 store with every uncertified (all-zero, non-lethal) row "
            "taken from the PDLP 1e-6 store where that certified the program and from the "
            "PDLP 1e-4 store otherwise; written by pdlp_fill_store.py"
        ),
        "sources": {s: json.loads(str(stores[s]["provenance"])) for s in SOURCES},
        "rows_by_source": counts,
        "solver": "pdlp",
    }
    path = osp.join(TABLE_DIR, "ess_solve_encodings_YPD_pdlp1e-8fill.npz")
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=filled,
        v_wild_type=stores["pdlp1e-8"]["v_wild_type"].astype(np.float32),
        rxn_ids=stores["pdlp1e-8"]["rxn_ids"],
        provenance=np.array(json.dumps(provenance)),
    )
    out = {"path": path, "rows_by_source": counts, "n_genes": len(genes)}
    with open(osp.join(RESULTS_DIR, "pdlp_fill_store.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
