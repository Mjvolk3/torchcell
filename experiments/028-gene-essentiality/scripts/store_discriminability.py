# experiments/028-gene-essentiality/scripts/store_discriminability.py
# [[experiments.028-gene-essentiality.scripts.store_discriminability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/store_discriminability

r"""Why the dual-simplex vertex is the worst of the four stores.

Rounds 36 and 38 trained the same layer-free cell over four stores of the same growth
program's optimal face and read 0.912, 0.911 and 0.907 from three of them and 0.840 from
the growth program's own dual-simplex vertex. ``store_stats.py`` described the vectors;
this asks what a head can read off them.

Three quantities per store, over the 1,121 labeled genes and their encoding
:math:`\log(1 + |v|)`, which is what the affine sees:

1. **Motion that tracks the deletion.** The distance of each deletion's row from the
   wild-type row, against that deletion's drop in growth: Spearman over all genes, and
   over the genes the solve calls viable. A store whose rows move for reasons other than
   the deletion has motion a head cannot use.
2. **Where the variance sits.** The share of the total column variance carried by the ten
   widest columns, and how many of those columns touch the layer's clamp, where a loop can
   spin.
3. **What a single column can do.** The best single-column AUROC, as a floor on what a
   linear head has to work with.

Measured, not argued: the script prints and stores the numbers, and the reading goes in
the document beside them.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/store_discriminability.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT_DIR = osp.join(RESULTS_DIR, "splits")
#: The released split, whose ``labels`` map is where every probe in this experiment
#: takes its essentiality calls from.
SPLIT_FILE = "ess_fcl_val0.2_s0.json.gz"
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
BIOMASS = "r_2111"
FLUX_SCALE = 100.0
#: The trained read of each store, from rounds 36 and 38, carried here so the description
#: sits beside the number it is meant to explain. Six seeds each, pooled.
TRAINED_AUROC = {
    "pFBA (two-stage, scipy HiGHS)": 0.912,
    "dual simplex (vertex)": 0.840,
    "interior point, no crossover": 0.911,
    "PDLP, no crossover": 0.907,
}
STORES = {
    "pFBA (two-stage, scipy HiGHS)": "ess_solve_encodings_YPD.npz",
    "dual simplex (vertex)": "ess_solve_encodings_YPD_simplex.npz",
    "interior point, no crossover": "ess_solve_encodings_YPD_ipm.npz",
    "PDLP, no crossover": "ess_solve_encodings_YPD_pdlp.npz",
}
REFERENCE = "pFBA (two-stage, scipy HiGHS)"


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUROC with ties at their mid-rank."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = np.sort(score)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class StoreRead(BaseModel):
    """One store, read as a feature table."""

    store: str
    trained_auroc: float
    n_genes: int
    #: Motion against the deletion.
    spearman_distance_growth_drop: float
    spearman_distance_growth_drop_viable: float
    median_distance_viable: float
    median_distance_lethal: float
    #: A viable deletion whose row is the wild-type row carries nothing the wild type
    #: does not, so this is the share of the store that is dead weight on those genes.
    frac_viable_rows_identical_to_wild_type: float
    #: Where the variance sits.
    n_columns_varying: int
    variance_share_top10: float
    n_clamp_columns_in_top10: int
    #: What one column can do.
    best_single_column_auroc: float
    n_columns_auroc_above_0_7: int


def main() -> None:
    payloads = {
        name: np.load(osp.join(TABLE_DIR, f), allow_pickle=False)
        for name, f in STORES.items()
    }
    genes = [str(g) for g in payloads[REFERENCE]["genes"]]
    with gzip.open(osp.join(SPLIT_DIR, SPLIT_FILE), "rt") as fh:
        labels = {g: int(v) for g, v in json.load(fh)["labels"].items()}
    keep = np.array([g in labels for g in genes])
    label = np.array([labels[g] for g in genes if g in labels], dtype=int)
    rxn_ids = [str(r) for r in payloads[REFERENCE]["rxn_ids"]]
    biomass = rxn_ids.index(BIOMASS)
    rows: list[StoreRead] = []
    for name, p in payloads.items():
        v = p["v_deletion"].astype(np.float64)[keep]
        enc = np.log1p(np.abs(v))
        wt_v = p["v_wild_type"].astype(np.float64)
        wt = np.log1p(np.abs(wt_v))
        growth = v[:, biomass]
        wt_growth = float(wt_v[biomass])
        drop = (wt_growth - growth) / wt_growth
        dist = np.linalg.norm(enc - wt[None, :], axis=1)
        # "Viable" is the solve's own call, not the label: a deletion the program still
        # grows on. The question is whether the rows of those move for no reason.
        viable = drop < 0.01
        lethal = drop > 0.99
        var = enc.var(axis=0)
        top10 = np.argsort(var)[::-1][:10]
        clamp_col = (np.abs(v) >= FLUX_SCALE - 1e-6).any(axis=0)
        col_auroc = np.array(
            [auroc(enc[:, j], label) if var[j] > 1e-12 else 0.5 for j in range(enc.shape[1])]
        )
        # A column that ranks essential genes LOW is as useful as one that ranks them
        # high, so the floor is on the distance from chance.
        signed = np.abs(col_auroc - 0.5) + 0.5
        rows.append(
            StoreRead(
                store=name,
                trained_auroc=TRAINED_AUROC[name],
                n_genes=int(keep.sum()),
                spearman_distance_growth_drop=float(spearmanr(dist, drop).statistic),
                spearman_distance_growth_drop_viable=float(
                    spearmanr(dist[viable], drop[viable]).statistic
                ),
                median_distance_viable=float(np.median(dist[viable])),
                median_distance_lethal=float(np.median(dist[lethal])),
                frac_viable_rows_identical_to_wild_type=float(
                    (dist[viable] < 1e-9).mean()
                ),
                n_columns_varying=int((var > 1e-12).sum()),
                variance_share_top10=float(var[top10].sum() / max(var.sum(), 1e-12)),
                n_clamp_columns_in_top10=int(clamp_col[top10].sum()),
                best_single_column_auroc=float(signed.max()),
                n_columns_auroc_above_0_7=int((signed > 0.7).sum()),
            )
        )
    out = {
        "question": (
            "why the growth program's dual-simplex vertex is the worst of the four "
            "stores (0.840 against 0.907 to 0.912, rounds 36 and 38)"
        ),
        "encoding": "log1p(|v|), the affine's input",
        "n_genes": int(keep.sum()),
        "split_file": SPLIT_FILE,
        "n_essential": int(label.sum()),
        "viable_rule": "growth drop below 1 percent of the wild type's",
        "lethal_rule": "growth drop above 99 percent",
        "stores": [r.model_dump() for r in rows],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, "store_discriminability.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows:
        print(
            f"{r.store:34s} trained {r.trained_auroc:.3f} | rho(dist, drop) "
            f"{r.spearman_distance_growth_drop:+.3f} viable "
            f"{r.spearman_distance_growth_drop_viable:+.3f} | dist viable "
            f"{r.median_distance_viable:.3f} lethal {r.median_distance_lethal:.3f} "
            f"(wt-identical {r.frac_viable_rows_identical_to_wild_type:.2f}) | "
            f"cols {r.n_columns_varying} "
            f"top10 var {r.variance_share_top10:.2f} clamp {r.n_clamp_columns_in_top10} "
            f"| best col {r.best_single_column_auroc:.3f} "
            f"(>0.7: {r.n_columns_auroc_above_0_7})"
        )
    print(path)


if __name__ == "__main__":
    main()
