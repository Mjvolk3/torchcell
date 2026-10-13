# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/summarize_cgt_rounds.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.summarize_cgt_rounds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/summarize_cgt_rounds
"""The derived tables of rounds 8 to 13 that ``compare_models.py`` does not produce.

``compare_models.py`` scores the selected prediction of every arm against nested ridge.
Three further tables are read off the per-epoch histories and the saved predictions:

``results/cgt_rounds_curves.csv``
    per arm, the last-epoch mean over folds of the held-out centered Spearman (mean over a
    fold's held-out compounds of the centered Spearman across strains, then over the run's
    seeds), the best epoch of that fold-mean curve, the train loss of the last batch, the
    wall time per seed and the peak GPU memory. The last-epoch number does not depend on
    the validation selector, which is what separates the arms of round 8.

``results/cgt_rounds_loss_curves.csv``
    for every arm whose history has them (round 11 on), the mean over folds, per epoch, of
    the standardized MSE over every strain on the fitted, validation and held-out
    compounds, beside the validation and held-out centered Spearman.

``results/cgt_rounds_loss_baselines.csv``
    per fold of fold seed 0, the held-out standardized MSE of two constant predictors
    (the global mean, each gene's mean over the non-test compounds), nested ridge, and
    the saved prediction of the named arms; and the response variance each compound
    carries, which is why one compound sets the level of the loss.
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from train_factorized import CELL_TABLE, EMBEDDING_DIR, PREDICTIONS  # noqa: E402
from vanacloig_data import load_cells, make_folds  # noqa: E402

load_dotenv()
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "038-env-chemgen-vanacloig-cgt-corrected", "results"
)
SWEEPS = "r8_*", "r9_*", "r10_*", "r11_*", "r12_*", "r13_*", "r14_*", "r15_*"
#: arm -> (sweep of fold k, config name of fold k, compounds the arm was fitted on)
LOSS_ARMS = {
    "L1_lam0 (val-selected, 1 seed)": (
        lambda k: "r8_small_a",
        lambda k: f"L1_lam0_f{k}",
        "train",
    ),
    "L8_lam1 (val-selected, 1 seed)": (
        lambda k: {0: "r8_deep_a", 1: "r8_deep_c", 2: "r8_deep_c"}.get(k, "r8_deep_d"),
        lambda k: f"L8_lam1_f{k}",
        "train",
    ),
    "bil_L1_lam0 (pool, 3 seeds)": (
        lambda k: "r9_control",
        lambda k: f"bil_L1_lam0_fs0_f{k}",
        "pool",
    ),
    "enc_L1_lam0 (pool, 3 seeds)": (
        lambda k: "r10_envenc",
        lambda k: f"enc_L1_lam0_fs0_f{k}",
        "pool",
    ),
}


def arm_of(name: str) -> str:
    name = re.sub(r"_f\d$", "", name)
    name = re.sub(r"_fs\d$", "", name)
    return re.sub(r"_s\d{3}$", "", name)


def histories() -> pd.DataFrame:
    frames = []
    for pattern in SWEEPS:
        for path in glob.glob(
            osp.join(RESULTS, "factorized", pattern, "*_history.csv")
        ):
            name = osp.basename(path).removesuffix("_history.csv")
            h = pd.read_csv(path)
            frames.append(
                h.assign(
                    arm=arm_of(name),
                    config=name,
                    sweep=osp.basename(osp.dirname(path)),
                    epoch=h["epoch"].round().astype(int),
                )
            )
    return pd.concat(frames, ignore_index=True)


def curves(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in h.groupby("arm"):
        last = g[g["epoch"] == g["epoch"].max()]
        by_epoch = g.groupby("epoch")["test_centered_mean"].mean()
        rows.append(
            {
                "arm": arm,
                "configs": g["config"].nunique(),
                "seeds_per_config": g.groupby("config")["seed"].nunique().max(),
                "epochs": int(g["epoch"].max()),
                "held_out_centered_spearman_last_epoch": last[
                    "test_centered_mean"
                ].mean(),
                "held_out_centered_spearman_best_epoch": by_epoch.max(),
                "best_epoch": int(by_epoch.idxmax()),
                "train_loss_last_batch_last_epoch": last["train_loss"].mean(),
                "minutes_per_seed": last["seconds"].mean() / 60,
                "gpu_peak_gb": g["gpu_peak_gb"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("arm")


def loss_curves(h: pd.DataFrame) -> pd.DataFrame:
    with_loss = h[h["val_loss"].notna()] if "val_loss" in h else h.iloc[:0]
    columns = {
        "train_loss_full": "train_loss_standardized_mse",
        "val_loss": "validation_loss_standardized_mse",
        "test_loss": "held_out_loss_standardized_mse",
        "val_centered_mean": "validation_centered_spearman",
        "test_centered_mean": "held_out_centered_spearman",
    }
    out = (
        with_loss.groupby(["arm", "epoch"])[list(columns)]
        .mean()
        .rename(columns=columns)
        .reset_index()
    )
    out["configs"] = out["arm"].map(with_loss.groupby("arm")["config"].nunique())
    return out


def loss_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    y = cells.matrix(cells.response)
    rows = []
    for k in range(5):
        fold = make_folds(len(cells.compounds), 5, 4, 0)[k]
        pool = sorted(fold.train + fold.val)
        sd = {"train": np.nanstd(y[:, fold.train]), "pool": np.nanstd(y[:, pool])}

        def mse(prediction: np.ndarray, scale: float, fold=fold) -> float:
            return float(
                np.nanmean(((prediction[:, fold.test] - y[:, fold.test]) / scale) ** 2)
            )

        gene_mean = np.repeat(np.nanmean(y[:, pool], axis=1)[:, None], y.shape[1], 1)
        row = {
            "fold": k,
            "global mean": mse(np.full_like(y, np.nanmean(y[:, pool])), sd["pool"]),
            "gene mean": mse(gene_mean, sd["pool"]),
            "nested ridge": mse(
                np.load(osp.join(PREDICTIONS, f"ridge_fold{k}_seed0.npy")), sd["pool"]
            ),
        }
        for arm, (sweep, name, fit_on) in LOSS_ARMS.items():
            path = osp.join(PREDICTIONS, sweep(k), f"{name(k)}_fold{k}_seed0.npy")
            row[arm] = mse(np.load(path).astype(np.float64), sd[fit_on])
        rows.append(row)
    baselines = pd.DataFrame(rows)
    mean = baselines.drop(columns="fold").mean().to_dict() | {"fold": "mean"}
    baselines = pd.concat([baselines, pd.DataFrame([mean])], ignore_index=True)
    variance = np.nanvar(y, axis=0)
    share = pd.DataFrame(
        {
            "compound": cells.compounds,
            "response_variance_across_strains": variance,
            "share_of_total_variance": variance / variance.sum(),
            "ratio_to_median_compound": variance / np.median(variance),
        }
    ).sort_values("response_variance_across_strains", ascending=False)
    return baselines, share


def main() -> None:
    h = histories()
    table = curves(h)
    table.to_csv(osp.join(RESULTS, "cgt_rounds_curves.csv"), index=False)
    print(table.round(3).to_string(index=False))
    losses = loss_curves(h)
    losses.to_csv(osp.join(RESULTS, "cgt_rounds_loss_curves.csv"), index=False)
    shown = losses[losses["epoch"].isin([1, 5, 10, 20, 40, 60, 100])]
    print(shown.round(3).to_string(index=False))
    baselines, share = loss_baselines()
    baselines.to_csv(osp.join(RESULTS, "cgt_rounds_loss_baselines.csv"), index=False)
    share.to_csv(osp.join(RESULTS, "compound_variance_share.csv"), index=False)
    print(baselines.round(3).to_string(index=False))
    print(share.head(4).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
