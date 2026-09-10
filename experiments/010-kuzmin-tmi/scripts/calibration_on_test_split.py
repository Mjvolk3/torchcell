#!/usr/bin/env python
# experiments/010-kuzmin-tmi/scripts/calibration_on_test_split.py
# [[experiments.010-kuzmin-tmi.scripts.calibration_on_test_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/calibration_on_test_split
"""Calibration of the three checkpoints on the whole 010 test split.

WHY THIS EXISTS. The positive-panel report carried a calibration slope of 0.877 on the
held-out test split, and the inference_4 measured-triple diagnostic carried 0.24 on its
930 held-out triples. Those two numbers say opposite things about whether a predicted
tau can be read as an estimate of tau, and the 0.877 had no generating script anywhere
in the repo, so it could not be checked. This computes the test-split number from stored
artifacts so both sides of the comparison have provenance.

WHAT IT USES, AND WHY NO GPU IS NEEDED. The per-record test predictions for all three
checkpoints were written once and kept:

  results/cgt_predictions_M0{1,2,3}_<run>_test.npy   37,673 predictions each
  results/cgt_record_rows_test.npy                   LMDB row for each position

So this is an LMDB read of the test rows plus a regression. It never touches a GPU.

WHAT IT REPORTS. The slope regresses MEASURED on PREDICTED, so 1 is calibrated and below
1 is inflation. Three populations, because the disagreement is about which one the panel
operates in:

  whole test split   every one of the 37,673 records, dominated by the bulk near zero
  called subset      records whose measured tau clears the Kuzmin 2020 positive call
  prediction tail    records the model ranks highest, which is where a panel is drawn

Run from repo root:
  python experiments/010-kuzmin-tmi/scripts/calibration_on_test_split.py

Outputs, under results/:
  test_split_calibration.csv    one row per population
  test_split_calibration.json   the numbers quoted in prose
"""

import json
import os
import os.path as osp

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats as sps
from tqdm import tqdm

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
BUILD_010 = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)

CHECKPOINTS = {"M01": "lzs9pcj3", "M02": "yv4r30bi", "M03": "c7671wgj"}
POSITIVE_CALL, P_CUT = 0.08, 0.05
# The rank windows the report quotes for inference_4, so the two sides are comparable.
TAIL_WINDOWS = (100, 500, 1_000, 5_000)


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Ensemble mean prediction per test record, and the LMDB row it belongs to."""
    rows = np.load(osp.join(RESULTS, "cgt_record_rows_test.npy"))
    preds = []
    for tag, run in CHECKPOINTS.items():
        p = np.load(osp.join(RESULTS, f"cgt_predictions_{tag}_{run}_test.npy"))
        if p.shape != rows.shape:
            raise SystemExit(f"{tag}: {p.shape} predictions for {rows.shape} rows")
        preds.append(p)
    return np.mean(preds, axis=0), rows


def load_labels(rows: np.ndarray) -> pd.DataFrame:
    """Measured tau and its p-value for the given LMDB rows."""
    with open(osp.join(BUILD_010, "data_module_cache/index_seed_42.json")) as f:
        index = json.load(f)
    test = set(index["test"])
    missing = [int(r) for r in rows if int(r) not in test]
    if missing:
        raise SystemExit(f"{len(missing)} stored rows are not in the test split")

    env = lmdb.open(
        osp.join(BUILD_010, "processed/lmdb"), readonly=True, lock=False, readahead=False
    )
    tau, pval = [], []
    with env.begin() as txn:
        for r in tqdm(rows, desc="reading test labels"):
            rec = json.loads(txn.get(str(int(r)).encode()))
            phen = rec[0]["experiment"]["phenotype"]
            tau.append(phen.get("gene_interaction"))
            pval.append(phen.get("gene_interaction_p_value"))
    env.close()
    return pd.DataFrame({"row": rows, "measured_tau": tau, "measured_p": pval})


def stats(sub: pd.DataFrame, label: str) -> dict:
    x = sub["measured_tau"].to_numpy(dtype=float)
    y = sub["pred_mean"].to_numpy(dtype=float)
    row = {"population": label, "n": int(len(sub))}
    if len(sub) < 3:
        return row
    row["pearson_r"] = float(sps.pearsonr(x, y)[0])
    row["spearman_rho"] = float(sps.spearmanr(x, y)[0])
    row["mean_measured"] = float(x.mean())
    row["mean_predicted"] = float(y.mean())
    row["mean_signed_error"] = float((y - x).mean())
    # Regress MEASURED on PREDICTED. Noise in the measurement inflates the standard
    # error of this slope but does not bias it, which is why the slope is the statistic
    # to quote when the labels are themselves noisy.
    row["calibration_slope"] = float(np.polyfit(y, x, 1)[0])
    row["sd_measured"] = float(x.std())
    row["sd_predicted"] = float(y.std())
    # Both halves of the call are reported separately and deliberately. Quoting the
    # magnitude-only share as "a real call" overstates it by roughly sixfold here,
    # because most measured tau in this build are not significant at p < 0.05.
    magnitude = x > POSITIVE_CALL
    called = magnitude & (sub["measured_p"].to_numpy(dtype=float) < P_CUT)
    row["n_magnitude_positive"] = int(magnitude.sum())
    row["share_magnitude_positive"] = float(magnitude.mean())
    row["n_called_positive"] = int(called.sum())
    row["share_called_positive"] = float(called.mean())
    return row


def main():
    pred, rows = load_predictions()
    df = load_labels(rows)
    df["pred_mean"] = pred
    df = df.dropna(subset=["measured_tau", "measured_p"]).reset_index(drop=True)
    print(f"test split: {len(df):,} records with a measured tau and p-value")

    df = df.sort_values("pred_mean", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    out = [stats(df, "whole test split")]

    called = df[(df.measured_tau > POSITIVE_CALL) & (df.measured_p < P_CUT)]
    out.append(stats(called, f"measured positive calls, tau > {POSITIVE_CALL}"))

    strong = df[df.measured_tau > 0.20]
    out.append(stats(strong, "measured tau > +0.20, magnitude only"))

    for k in TAIL_WINDOWS:
        out.append(stats(df.head(k), f"top {k:,} by prediction"))

    # The report's own claim, checked on its own terms: records the model PREDICTS above
    # +0.20. This is the bin the abstract's "mean actual tau of +0.312" came from.
    for cut in (0.08, 0.20, 0.30):
        sub = df[df.pred_mean > cut]
        out.append(stats(sub, f"predicted > +{cut:.2f}"))

    table = pd.DataFrame(out)
    table.to_csv(osp.join(RESULTS, "test_split_calibration.csv"), index=False)
    print("\n" + table.to_string(index=False))

    whole = out[0]
    summary = {
        "n_test_records": int(len(df)),
        "checkpoints": CHECKPOINTS,
        "positive_call": POSITIVE_CALL,
        "p_cut": P_CUT,
        "whole_test_split": whole,
        "populations": out,
        "note": (
            "Slope regresses measured on predicted. The whole-split slope is the "
            "honest summary; a slope computed only inside a predicted-positive bin is "
            "conditioned on the predictor and is not a calibration statistic."
        ),
    }
    with open(osp.join(RESULTS, "test_split_calibration.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {RESULTS}/test_split_calibration.{{csv,json}}")


if __name__ == "__main__":
    main()
