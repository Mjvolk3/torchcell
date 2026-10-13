# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/learning_curve.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.learning_curve]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/learning_curve
"""The held-out score as a function of the number of fitted compounds.

Every model of this experiment ties nested ridge when about 33 compounds are fitted.
Whether more compounds would help is read off a learning curve: each fold's non-test
pool is cut to nested subsets of 8, 16 and 24 compounds
(``vanacloig_data.subsample_pool``), a model is fitted on each subset, and the fold's
held-out compounds are scored exactly as everywhere else, each side centered by its own
mean over the FITTED compounds.

``--ridge`` fits the reference on every subset and on the whole pool: kernel ridge over
the linear kernel of standardized FCFP4 counts, its penalty, rank and clip chosen by
leave-one-compound-out inside the fitted subset (the ``krr|linear:fcfp4_count`` row of
``baseline_ladder.py``). The whole-pool prediction must reproduce the saved ladder
prediction, which is asserted. Writes ``results/learning_curve/ridge_scores.csv``.

Without the flag the script only summarizes: ridge from that file, the environment
encoder from round 17 (``results/factorized/r17_curve``, one seed) and, for the whole
pool, the seed-0 member of round 10. Writes ``results/learning_curve/summary.csv``, one
row per (target, number of fitted compounds): each model's median and mean Spearman over
the compound-evaluations both scored, and the paired mean difference encoder minus ridge
with a 95% bootstrap interval resampling compounds.
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from baseline_ladder import (  # noqa: E402
    EMBEDDING_DIR,
    PREDICTIONS,
    fold_kernel,
    grid,
    inner_scores,
    load_features,
    predict_candidates,
)
from compare_models import bootstrap_mean  # noqa: E402
from train_factorized import CELL_TABLE  # noqa: E402
from vanacloig_data import (  # noqa: E402
    load_cells,
    make_folds,
    score_compounds,
    subsample_pool,
)

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "038-env-chemgen-vanacloig-cgt-corrected", "results"
)
OUT = osp.join(RESULTS, "learning_curve")
KERNEL = "linear:fcfp4_count"
SIZES: tuple[int | None, ...] = (8, 16, 24, None)
FOLD_SEEDS = (0, 1, 2)


def ridge_scores() -> pd.DataFrame:
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    x = load_features(cells, "fcfp4_count")
    assert x is not None
    y = cells.matrix(cells.response)
    candidates = [c for c in grid(KERNEL) if c.model == "krr"]
    frames = []
    for fold_seed in FOLD_SEEDS:
        for fold in make_folds(len(cells.compounds), 5, 4, fold_seed):
            for n in SIZES:
                fit = subsample_pool(fold, fold_seed, n)
                kernel = fold_kernel("linear", x, np.array(fit))
                inner = inner_scores(candidates, kernel, y, fit)
                chosen = candidates[int(np.nanargmax(inner))]
                columns = fit + fold.test
                prediction = np.full(y.shape, np.nan)
                prediction[:, columns] = predict_candidates(
                    [chosen], kernel, y, fit, columns
                )[0]
                if n is None:
                    saved = np.load(
                        osp.join(
                            PREDICTIONS, f"ridge_fold{fold.fold}_seed{fold_seed}.npy"
                        )
                    )
                    assert np.allclose(prediction, saved, atol=1e-4, equal_nan=True), (
                        "the whole-pool fit does not reproduce the ladder's ridge"
                    )
                frames.append(
                    score_compounds(cells, prediction, fit, fold.test).assign(
                        fold_seed=fold_seed,
                        fold=fold.fold,
                        n_fit=len(fit),
                        size="pool" if n is None else str(n),
                        options=chosen.options(),
                    )
                )
                print(
                    f"fold seed {fold_seed} fold {fold.fold} n {len(fit)}: "
                    f"{chosen.options()}, inner {np.nanmax(inner):.3f}",
                    flush=True,
                )
    return pd.concat(frames, ignore_index=True)


def encoder_scores() -> pd.DataFrame:
    """Seed-0 scores of the environment encoder at every size."""
    frames = []
    for sweep, size_of in (
        ("r17_curve", lambda name: name.split("_n")[1].split("_")[0]),
        ("r10_envenc", lambda name: "pool"),
    ):
        for path in sorted(
            glob.glob(osp.join(RESULTS, "factorized", sweep, "*_scores.csv"))
        ):
            d = pd.read_csv(path)
            d = d[d["member"] == "seed0"]
            frames.append(d.assign(size=d["name"].map(size_of)))
    return pd.concat(frames, ignore_index=True)


def summarize(ridge: pd.DataFrame, encoder: pd.DataFrame) -> pd.DataFrame:
    key = ["fold_seed", "fold", "compound", "target", "size"]
    paired = encoder[key + ["spearman"]].merge(
        ridge[key + ["spearman", "n_fit"]], on=key, suffixes=("_encoder", "_ridge")
    )
    paired = paired.dropna(subset=["spearman_encoder", "spearman_ridge"])
    rng = np.random.default_rng(0)
    rows = []
    for (target, size), g in paired.groupby(["target", "size"]):
        per_compound = (
            (g["spearman_encoder"] - g["spearman_ridge"]).groupby(g["compound"]).mean()
        )
        low, high = bootstrap_mean(per_compound.to_numpy(), rng)
        rows.append(
            {
                "target": target,
                "size": size,
                "n_fit_compounds_min": int(g["n_fit"].min()),
                "n_fit_compounds_max": int(g["n_fit"].max()),
                "compound_evaluations": len(g),
                "compounds": len(per_compound),
                "ridge_spearman_median": g["spearman_ridge"].median(),
                "ridge_spearman_mean": g["spearman_ridge"].mean(),
                "encoder_spearman_median": g["spearman_encoder"].median(),
                "encoder_spearman_mean": g["spearman_encoder"].mean(),
                "encoder_minus_ridge_mean_diff": (
                    g["spearman_encoder"] - g["spearman_ridge"]
                ).mean(),
                "encoder_minus_ridge_cluster_ci_low": low,
                "encoder_minus_ridge_cluster_ci_high": high,
                "compounds_improved": int((per_compound > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "n_fit_compounds_min"])


def ridge_curve(ridge: pd.DataFrame) -> pd.DataFrame:
    """Ridge alone at every size, on every compound-evaluation it scored."""
    return (
        ridge.groupby(["target", "size"])
        .agg(
            n_fit_compounds_min=("n_fit", "min"),
            n_fit_compounds_max=("n_fit", "max"),
            compound_evaluations=("spearman", "count"),
            ridge_spearman_median=("spearman", "median"),
            ridge_spearman_mean=("spearman", "mean"),
        )
        .reset_index()
        .sort_values(["target", "n_fit_compounds_min"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge", action="store_true", help="refit ridge first")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    if args.ridge:
        ridge_scores().to_csv(osp.join(OUT, "ridge_scores.csv"), index=False)
    ridge = pd.read_csv(osp.join(OUT, "ridge_scores.csv"), dtype={"size": str})
    curve = ridge_curve(ridge)
    curve.to_csv(osp.join(OUT, "ridge_curve.csv"), index=False)
    print(curve.round(3).to_string(index=False))
    summary = summarize(ridge, encoder_scores())
    summary.to_csv(osp.join(OUT, "summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
