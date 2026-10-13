# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/baselines_same_folds.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.baselines_same_folds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/baselines_same_folds
"""The references a trained model has to beat, on the folds the model is scored on.

Experiment 031 scored a ridge map from the compound fingerprint at Spearman 0.311 on the
centered target, leaving one compound out in turn, 41 folds. The models here train on
grouped folds with fewer training compounds, and a score under one scheme is not a
reference for a score under the other. So the two baselines are recomputed on exactly
the folds of ``vanacloig_data.make_folds``, with the same training compounds a model
sees (the validation compounds are left out of the fit, as they are for the model).

``gene_mean``  each gene's mean over the training compounds. It uses no compound
               feature, so it is the floor: zero by construction on the centered target.
``ridge``      multi-output ridge from the standardized compound fingerprint to the gene
               profile, penalty by leave-one-out on the training compounds, as in 031
               ``baseline_ceilings.ridge_predict``.

Writes ``results/baselines_same_folds.csv``, one row per compound, target and model.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import RidgeCV

sys.path.insert(0, osp.dirname(__file__))
from vanacloig_data import load_cells, make_folds, score_compounds  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "038-env-chemgen-vanacloig-cgt-corrected", "results"
)
ALPHAS = np.logspace(-1, 6, 22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cells = load_cells(args.cell_table, args.embedding)
    measured = cells.matrix(cells.response)
    print(
        f"{len(cells.response):,} cells, {measured.shape[0]:,} genes, "
        f"{measured.shape[1]} compounds, features {cells.compound_features.shape}"
    )

    frames = []
    for fold in make_folds(len(cells.compounds), args.n_folds, args.n_val, args.seed):
        held_out = fold.val + fold.test
        gene_mean = np.nanmean(measured[:, fold.train], axis=1)

        # a gene measured under no training compound has no training mean; it is left
        # out of the fit and gets no prediction, so it is not scored
        fit = np.isfinite(gene_mean)
        x_train = cells.compound_features[fold.train]
        mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        # a gene unmeasured under SOME training compound takes its own training mean
        # there, so the fit sees a complete matrix and that cell carries no signal
        block = measured[np.ix_(fit, fold.train)]
        y_train = np.where(np.isnan(block), gene_mean[fit, None], block).T
        ridge = RidgeCV(alphas=ALPHAS).fit((x_train - mu) / sd, y_train)

        predictions = {
            "gene_mean": np.repeat(gene_mean[:, None], measured.shape[1], axis=1),
            "ridge": np.full(measured.shape, np.nan),
        }
        # the training columns are predicted too: the centered score subtracts the
        # model's own mean over them
        columns = fold.train + held_out
        predictions["ridge"][np.ix_(fit, columns)] = ridge.predict(
            (cells.compound_features[columns] - mu) / sd
        ).T
        for model, prediction in predictions.items():
            for split, columns in (("val", fold.val), ("test", fold.test)):
                scores = score_compounds(cells, prediction, fold.train, columns)
                frames.append(
                    scores.assign(
                        model=model,
                        split=split,
                        fold=fold.fold,
                        n_train_compounds=len(fold.train),
                        ridge_alpha=float(ridge.alpha_),
                    )
                )

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(osp.join(RESULTS_DIR, "baselines_same_folds.csv"), index=False)
    summary = (
        out[out["split"] == "test"]
        .groupby(["model", "target"])
        .agg(
            compounds=("spearman", "size"),
            scored=("spearman", "count"),
            spearman_median=("spearman", "median"),
            pearson_median=("pearson", "median"),
            ceiling_median=("ceiling", "median"),
        )
        .reset_index()
    )
    summary.to_csv(
        osp.join(RESULTS_DIR, "baselines_same_folds_summary.csv"), index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
