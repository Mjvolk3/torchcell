# experiments/040-inhibitor-synergy-wetlab/scripts/r11_readout.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r11_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r11_readout
"""Round 11 read arm by arm, every number beside the round's own control.

Reads every ``results/mixture-delta/r11_*/mix_*_scores.csv`` (the Delta tables, pulled
to GilaHyper) and keeps the fold ENSEMBLE rows (the mean prediction of a fold's seeds),
then averages them over the arm's folds. The control is ``van_e150_cell_bliss_attend``,
round 10's Bliss arm re-run at the same folds and seeds on the same inputs, so every
difference below is within the round. The noise floor printed with it is the spread of
the control's single seeds around their fold ensemble.

Columns (served growth call where a call applies):

``van``     Vanacloig compound-cold centered Spearman, median over the fold's test
            compounds (nested ridge on FCFP4 counts: 0.359 on 32 compounds); on the
            gene split (``split`` gene), the gene-centered Spearman over held-out genes
``tit``     ex21 titration Spearman over all 180 wells (a train fit when ex21 is fitted)
``dose``    how many of the six inhibitors' own titrations the model orders at all
            (a per-inhibitor Spearman exists only when the prediction moves with dose)
            and their mean. A prediction that falls strictly with dose scores 0.767
            here, the data's own ordering by dose, whatever the curve; a curve that
            saturates ties some doses and moves it (the Hill arms read 0.767 to
            0.796). It separates a model that reads dose from one that does not, not
            one dose-response curve from another; ``tit`` does that
``auroc``   ex23 growth AUROC over the 63 combinations (Loewe 0.958, Bliss 0.872)
``worst``   ex23 Spearman over all 63 with no growth at zero (Loewe 0.772)
``grown``   ex23 fitness Spearman over the combinations that grew (Bliss 0.573)
``iso``     mean over the three isobole grids of |observed minus predicted| on the 81
            interior cells (Bliss rule 0.18 = mean of 0.08, 0.45, 0.02)
``hop``, ``het``, ``wil``  the auxiliary screens' held-out centered Spearman, over
            held-out compounds or, where ``held`` says dose, a held-out dose of a seen
            compound

    python experiments/040-inhibitor-synergy-wetlab/scripts/r11_readout.py

Writes ``results/r11_readout.csv`` (one row per arm) and prints one table per grid.
"""

from __future__ import annotations

import glob
import os.path as osp
import re

import numpy as np
import pandas as pd

EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
CONTROL = "van_e150_cell_bliss_attend"
INHIBITORS = 6


def grid_of(arm: str) -> str:
    """The round-11 grid an arm belongs to, by the names ``round11_arms`` gives."""
    if arm.endswith("_gene"):
        return "8 gene-cold"
    if re.search(r"_fs[12]$", arm):
        return "9 partitions"
    if "heldout" in arm or arm.endswith("_hill_ft"):
        return "4 dose"
    if "anchors_bliss" in arm and arm.endswith("_s345"):
        return "7 zero-shot seeds"
    if "_dev" in arm:
        return "3 deviation"
    if "hill_loewe" in arm and not arm.startswith("van_"):
        return "6 source ladder"
    if "hill" in arm or "hsa" in arm:
        return "2 functional form"
    if any(k in arm for k in ("displace", "_co", "attend")):
        return "1 operator"
    return "5 public bar"


def read_scores() -> pd.DataFrame:
    frames = []
    for path in sorted(
        glob.glob(osp.join(RESULTS, "mixture-delta", "r11_*", "*_scores.csv"))
    ):
        if "/r11_smoke/" in path:
            continue
        frames.append(pd.read_csv(path))
    scores = pd.concat(frames, ignore_index=True)
    scores["arm"] = (
        scores["name"]
        .str.replace(r"^mix_", "", regex=True)
        .str.replace(r"_f\d+$", "", regex=True)
    )
    return scores


def value(
    frame: pd.DataFrame, task: str, metric: str, call: str, subset: str
) -> pd.Series:
    rows = frame[
        (frame["task"] == task)
        & (frame["metric"] == metric)
        & (frame["call"] == call)
        & (frame["subset"] == subset)
    ]
    return rows.set_index("fold")["value"]


def arm_row(arm: str, frame: pd.DataFrame) -> dict[str, object]:
    ens = frame[frame["member"] == "ensemble"]
    tit_metric = (
        "fitness_spearman_train_fit"
        if (ens["metric"] == "fitness_spearman_train_fit").any()
        else "fitness_spearman"
    )
    per_inhibitor = ens[
        (ens["task"] == "ex21")
        & (ens["subset"] != "all_wells")
        & (ens["metric"] == tit_metric)
    ]
    finite = per_inhibitor["value"].dropna()
    n_folds = ens["fold"].nunique()
    iso = [
        value(
            ens,
            f"isobole_{run}",
            "mean_observed_minus_predicted",
            "served",
            "interior_cells",
        )
        .abs()
        .mean()
        for run in ("ex26", "ex27", "ex28")
    ]
    row: dict[str, object] = {
        "grid": grid_of(arm),
        "arm": arm,
        "folds": n_folds,
        "split": "gene" if (ens["subset"] == "test_genes").any() else "compound",
        "van": (
            value(
                ens, "vanacloig", "gene_centered_spearman_median", "both", "test_genes"
            )
            if (ens["subset"] == "test_genes").any()
            else value(
                ens, "vanacloig", "centered_spearman_median", "both", "test_compounds"
            )
        ).mean(),
        "tit": value(ens, "ex21", tit_metric, "both", "all_wells").mean(),
        "tit_fit": tit_metric.endswith("train_fit"),
        "dose_n": round(len(finite) / max(n_folds, 1), 1),
        "dose": finite.mean() if len(finite) else np.nan,
        "auroc": value(ens, "ex23", "growth_auroc", "served", "combinations").mean(),
        "worst": value(
            ens, "ex23", "worst_scenario_spearman", "served", "combinations"
        ).mean(),
        "grown": value(
            ens, "ex23", "fitness_spearman", "served", "combinations_that_grew"
        ).mean(),
        "iso": float(np.mean(iso)),
    }
    held = ens[ens["subset"].str.startswith("heldout_")]["subset"].unique()
    row["held"] = held[0].removeprefix("heldout_") if len(held) else ""
    for task, short in (
        ("hoepfner_hop", "hop"),
        ("hillenmeyer_het", "het"),
        ("wildenhain", "wil"),
    ):
        series = (
            value(
                ens, task, "centered_spearman_median", "both", f"heldout_{row['held']}"
            )
            if row["held"]
            else pd.Series(dtype=float)
        )
        row[short] = series.mean() if len(series) else np.nan
    return row


def seed_floor(frame: pd.DataFrame) -> dict[str, float]:
    """Spread of single seeds around their fold ensemble, for the control arm."""
    out = {}
    for key, task, metric, call, subset in (
        ("van", "vanacloig", "centered_spearman_median", "both", "test_compounds"),
        ("auroc", "ex23", "growth_auroc", "served", "combinations"),
    ):
        rows = frame[
            (frame["task"] == task)
            & (frame["metric"] == metric)
            & (frame["call"] == call)
            & (frame["subset"] == subset)
            & (frame["member"] != "ensemble")
        ]
        out[key] = float(rows.groupby("fold")["value"].std().mean())
    return out


def main() -> None:
    scores = read_scores()
    table = pd.DataFrame([arm_row(arm, frame) for arm, frame in scores.groupby("arm")])
    table = table.sort_values(["grid", "arm"], ignore_index=True)
    table.to_csv(osp.join(RESULTS, "r11_readout.csv"), index=False)
    control = table[table["arm"] == CONTROL].iloc[0]
    floor = seed_floor(scores[scores["arm"] == CONTROL])
    print(
        f"control {CONTROL}: van {control['van']:.3f}  auroc {control['auroc']:.3f}  "
        f"seed sd within a fold: van {floor['van']:.3f}, auroc {floor['auroc']:.3f}"
    )
    show = [
        "arm",
        "folds",
        "split",
        "van",
        "tit",
        "dose_n",
        "dose",
        "auroc",
        "worst",
        "grown",
        "iso",
        "held",
        "hop",
        "het",
        "wil",
    ]
    pd.set_option("display.width", 250)
    for grid, block in table.groupby("grid"):
        block = pd.concat([table[table["arm"] == CONTROL], block]).drop_duplicates(
            "arm"
        )
        print(f"\n== {grid}")
        print(
            block[show].to_string(
                index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"
            )
        )
    print(f"\n{len(table)} arms, {scores['name'].nunique()} configs read")


if __name__ == "__main__":
    main()
