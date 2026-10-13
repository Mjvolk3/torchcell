# experiments/035-env-chemgen-vanacloig-cgt/scripts/similarity_vs_score.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.similarity_vs_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/similarity_vs_score
"""Is a held-out compound predicted well only when a training compound looks like it?

For every (fold seed, fold, test compound) of the ladder, the nested ridge's centered
Spearman is set beside the compound's nearest training neighbor: the largest min/max
Tanimoto similarity on FCFP4 counts to any of the fold's non-test compounds, and the same
for MACCS keys. The compound's ceiling and whether the paper reports it ride along. If
the score tracks the nearest-neighbor similarity, the gap to the ceiling is a coverage
limit of 41 compounds, not a modeling limit, and denser chemical space is the remedy.

Writes ``results/similarity_vs_score.csv`` and a figure to ``ASSET_IMAGES_DIR``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, osp.dirname(__file__))
from baseline_ladder import EMBEDDING_DIR, load_features, tanimoto  # noqa: E402
from train_factorized import CELL_TABLE  # noqa: E402
from vanacloig_data import UNREPORTED_COMPOUNDS, load_cells, make_folds  # noqa: E402

from torchcell.utils import PLOT_PALETTE, mm_to_in  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-tag", default="ladder_r2")
    args = parser.parse_args()
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    similarity = {
        name: tanimoto(load_features(cells, name)) for name in ("fcfp4_count", "maccs")
    }
    np.fill_diagonal(similarity["fcfp4_count"], np.nan)
    np.fill_diagonal(similarity["maccs"], np.nan)

    ladder = pd.read_csv(osp.join(RESULTS, "ladder", f"{args.ladder_tag}_scores.csv"))
    ridge = ladder[
        (ladder["model"] == "krr")
        & (ladder["kernel"] == "linear:fcfp4_count")
        & (ladder["target"] == "centered")
    ]
    index = {c: i for i, c in enumerate(cells.compounds)}
    rows = []
    for (fold_seed, fold_index), g in ridge.groupby(["fold_seed", "fold"]):
        fold = make_folds(len(cells.compounds), 5, 4, int(fold_seed))[int(fold_index)]
        pool = sorted(fold.train + fold.val)
        for _, r in g.iterrows():
            j = index[r["compound"]]
            rows.append(
                {
                    "fold_seed": fold_seed,
                    "fold": fold_index,
                    "compound": r["compound"],
                    "published": r["compound"] not in UNREPORTED_COMPOUNDS,
                    "ridge_centered_spearman": r["spearman"],
                    "ceiling": r["ceiling"],
                    "nearest_fcfp4": float(
                        np.nanmax(similarity["fcfp4_count"][j, pool])
                    ),
                    "nearest_maccs": float(np.nanmax(similarity["maccs"][j, pool])),
                    "mean_fcfp4": float(np.nanmean(similarity["fcfp4_count"][j, pool])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(osp.join(RESULTS, "similarity_vs_score.csv"), index=False)

    ok = out.dropna(subset=["ridge_centered_spearman"])
    for col in ("nearest_fcfp4", "nearest_maccs", "mean_fcfp4", "ceiling"):
        rho, p = spearmanr(ok[col], ok["ridge_centered_spearman"])
        print(f"score vs {col}: Spearman {rho:+.3f} (p {p:.2g}, n {len(ok)})")
    pub = ok[ok["published"]]
    rho, p = spearmanr(pub["nearest_fcfp4"], pub["ridge_centered_spearman"])
    print(
        f"published only, score vs nearest_fcfp4: {rho:+.3f} (p {p:.2g}, n {len(pub)})"
    )

    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(118.9), mm_to_in(50)), constrained_layout=True
    )
    for ax, col, label in (
        (axes[0], "nearest_fcfp4", "nearest training compound, FCFP4 Tanimoto"),
        (axes[1], "ceiling", "compound ceiling (sqrt reliability)"),
    ):
        for published, color, marker in (
            (True, PLOT_PALETTE[0], "o"),
            (False, PLOT_PALETTE[5], "x"),
        ):
            sub = ok[ok["published"] == published]
            ax.scatter(
                sub[col],
                sub["ridge_centered_spearman"],
                s=8,
                c=color,
                marker=marker,
                label="reported by the paper" if published else "unreported",
                linewidths=0.6,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("ridge centered Spearman, held out")
        ax.axhline(0, color="black", lw=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[0].legend(frameon=False)
    for ext in ("svg", "png"):
        fig.savefig(
            osp.join(ASSET_IMAGES_DIR, f"035-similarity_vs_score.{ext}"), dpi=300
        )
    print(osp.join(ASSET_IMAGES_DIR, "035-similarity_vs_score.svg"))


if __name__ == "__main__":
    main()
