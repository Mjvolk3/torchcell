# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/aux_matrices.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.aux_matrices]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/aux_matrices
"""The other chemogenomic stores as gene-by-compound matrices, for pretraining.

A compound-inductive model of Vanacloig trains its compound encoder on 28 compounds per
fold. The other four sources in the 033 pooled store hold thousands more, measured on
largely the same deletion genes. Each becomes one matrix here, a SOURCE:

``hoepfner_hip``  Hoepfner 2014 heterozygous diploid arm (functional dose 0.5)
``hoepfner_hop``  Hoepfner 2014 homozygous diploid arm (functional dose 0)
``hillenmeyer_het``  Hillenmeyer 2008 heterozygous diploid
``wildenhain``    Wildenhain 2015 haploid deletions

A cell enters only when it has exactly one compound and that compound has an InChIKey.
Every Vanacloig compound is REMOVED from every source, so no Vanacloig test compound
reaches training through another dataset. The response is ORIENTED so a sick strain is
negative, as in Vanacloig (Hillenmeyer reports positive-is-sick; 033
``store_against_plan.SICK_SIGN``), and the (gene, compound) value is the mean over every
dose and measurement of that pair. Each source is then standardized by its own mean and
sd over cells.

Writes ``$DATA_ROOT/experiments/038-env-chemgen-vanacloig-cgt-corrected/aux/<source>.npz`` with
``genes``, ``inchikeys``, ``Y`` (genes x compounds, NaN where unmeasured) and
``aux_sources.csv`` beside the results.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
OUT = osp.join(
    DATA_ROOT, "experiments", "038-env-chemgen-vanacloig-cgt-corrected", "aux"
)
RESULTS = osp.join(
    EXPERIMENT_ROOT, "038-env-chemgen-vanacloig-cgt-corrected", "results"
)
VANACLOIG = "EnvChemgenVanacloig2022Dataset"
#: (source, dataset, functional dose or None, sign of a SICK strain as served)
SOURCES = (
    ("hoepfner_hip", "EnvChemgenHoepfner2014Dataset", 0.5, -1),
    ("hoepfner_hop", "EnvChemgenHoepfner2014Dataset", 0.0, -1),
    ("hillenmeyer_het", "HetHillenmeyer2008Dataset", None, +1),
    ("wildenhain", "EnvChemgenWildenhain2015Dataset", None, -1),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", required=True)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    columns = [
        "dataset",
        "query_gene",
        "functional_dose",
        "n_compounds",
        "n_with_inchikey",
        "inchikeys",
    ]
    vanacloig = pd.read_parquet(
        args.cell_table, columns=["inchikeys"], filters=[("dataset", "==", VANACLOIG)]
    )
    # ``inchikeys`` is one string, the compounds' keys joined by "|"
    excluded = set(vanacloig["inchikeys"])
    assert len(excluded) == 41

    rows = []
    for source, dataset, dose, sick_sign in SOURCES:
        t = pd.read_parquet(
            args.cell_table,
            columns=columns + ["responses"],
            filters=[("dataset", "==", dataset)],
        )
        if dose is not None:
            t = t[t["functional_dose"] == dose]
        n_cells = len(t)
        t = t[(t["n_compounds"] == 1) & (t["n_with_inchikey"] == 1)]
        # oriented so a sick strain is NEGATIVE, the Vanacloig convention
        t = t.assign(
            inchikey=t["inchikeys"], y=-sick_sign * t["responses"].map(np.mean)
        )
        n_vanacloig = int(t["inchikey"].isin(excluded).sum())
        t = t[~t["inchikey"].isin(excluded)]
        pair = t.groupby(["query_gene", "inchikey"], as_index=False)["y"].mean()
        mu, sd = pair["y"].mean(), pair["y"].std(ddof=1)
        pair["y"] = (pair["y"] - mu) / sd
        genes = sorted(pair["query_gene"].unique())
        keys = sorted(pair["inchikey"].unique())
        gi = {g: i for i, g in enumerate(genes)}
        ki = {k: i for i, k in enumerate(keys)}
        y = np.full((len(genes), len(keys)), np.nan, dtype=np.float32)
        y[pair["query_gene"].map(gi), pair["inchikey"].map(ki)] = pair["y"]
        np.savez_compressed(
            osp.join(OUT, f"{source}.npz"),
            genes=np.array(genes),
            inchikeys=np.array(keys),
            Y=y,
        )
        rows.append(
            {
                "source": source,
                "dataset": dataset,
                "functional_dose": dose,
                "cells_served": n_cells,
                "cells_dropped_vanacloig_compound": n_vanacloig,
                "pairs": len(pair),
                "genes": len(genes),
                "compounds": len(keys),
                "fill": float(np.isfinite(y).mean()),
                "oriented_mean_before_standardizing": mu,
                "oriented_sd_before_standardizing": sd,
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(osp.join(RESULTS, "aux_sources.csv"), index=False)


if __name__ == "__main__":
    main()
