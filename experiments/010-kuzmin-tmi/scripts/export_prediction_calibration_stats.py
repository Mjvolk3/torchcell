# experiments/010-kuzmin-tmi/scripts/export_prediction_calibration_stats.py
# [[experiments.010-kuzmin-tmi.scripts.export_prediction_calibration_stats]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/export_prediction_calibration_stats
"""Export every quantity the W019 calibration argument rests on, as one tidy CSV.

The calibration in [[experiments.W019-echo-crispr-array.next-strains-to-construct]] needs
four things: the prediction distribution the model actually emitted, the label population
it was validated against, the val metrics of the checkpoint that emitted those predictions,
and the val metrics logged at the end of each training run. Three of those four lived only
as numbers typed into a note, and one lived only in W&B. This script produces all of them
from the real artifacts so the note's numbers can be checked rather than trusted.

What it reads

  1. The full inference parquet for the checkpoint that produced the panel-12 predictions,
     `c7671wgj-best-pearson-epoch=24-val/gene_interaction/Pearson=0.4619`. 465,735,532 rows,
     ~4 GB. Streamed row group by row group, accumulating count, sum and sum of squares.
     Never materialized. Its sha256 goes in the CSV so a regenerated or swapped inference
     run is detected rather than silently absorbed.
  2. The 010 training build's `label_df.parquet` and the seed-42 split index, giving the
     train / val / test label populations.
  3. The three eval slurm logs that re-evaluated the best-pearson checkpoints on the val
     split. These carry the val metrics of the checkpoints that were actually used, which
     are NOT the last-epoch metrics in the W&B summaries.
  4. The W&B summaries for the three training runs and for the failed run `vjfp4d83`, so
     the last-epoch metrics and the failure survive without network access afterwards.

SD convention: **population** standard deviation, ddof = 0, for the predictions and for
every label split. The prediction SD is computed as sqrt(sum_sq / n - mean^2), which is the
population form; the label SDs use `Series.std(ddof=0)` to match. Every SD reported by this
script and every SD consumed from its CSV is population SD.

There is no fallback anywhere. A missing parquet, a missing slurm log or an unreachable
W&B all raise. A silently cached number is the exact failure this file exists to prevent.

Output
  results/prediction_calibration_stats.csv    one row per quantity: quantity, value, source

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/010-kuzmin-tmi/scripts/export_prediction_calibration_stats.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import os.path as osp

import pandas as pd
import pyarrow.parquet as pq
import wandb
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXP_DIR, "results")
SLURM_OUT = osp.join(EXP_DIR, "slurm/output")

CKPT_STEM = (
    "models-checkpoints-compute-3-3-2036902_"
    "bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5f60f100d6dc0e0288cd97e366fc15e-"
    "c7671wgj-best-pearson-epoch=24-val-gene_interaction-Pearson=0.4619.parquet"
)
INFERENCE_PARQUET = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_3/inferred", CKPT_STEM
)
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
LABEL_DF = osp.join(BUILD_DIR, "processed/label_df.parquet")
SPLIT_INDEX = osp.join(BUILD_DIR, "data_module_cache/index_seed_42.json")

WANDB_PROJECT = "zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"
# (checkpoint label, W&B run id, eval slurm log that re-evaluated its best-pearson ckpt)
CHECKPOINTS = (
    ("M01", "lzs9pcj3", "010-eval-m00_759.out"),
    ("M02", "yv4r30bi", "010-eval-m01_769.out"),
    ("M03", "c7671wgj", "010-eval-m02_761.out"),
)
# the run the note names as a failed run, kept so the claim survives without W&B
FAILED_RUN = "vjfp4d83"
# the checkpoint whose predictions the W019 panel-12 selection consumed
INFERENCE_CKPT = "c7671wgj"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def stream_prediction_moments(path: str) -> dict[str, float]:
    """Count, mean, population SD, min and max of `prediction`, one row group at a time."""
    pf = pq.ParquetFile(path)
    n, s, s2 = 0, 0.0, 0.0
    lo, hi = math.inf, -math.inf
    for i in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(i, columns=["prediction"]).column("prediction")
        a = col.to_numpy().astype("float64")
        n += a.size
        s += float(a.sum())
        s2 += float((a * a).sum())
        lo = min(lo, float(a.min()))
        hi = max(hi, float(a.max()))
    mean = s / n
    var = s2 / n - mean * mean  # population variance, ddof = 0
    return {"n": n, "mean": mean, "sd": math.sqrt(var), "min": lo, "max": hi,
            "row_groups": pf.metadata.num_row_groups}


def label_population() -> dict[str, dict[str, float]]:
    lab = pd.read_parquet(LABEL_DF).set_index("index")["gene_interaction"]
    with open(SPLIT_INDEX) as fh:
        split = json.load(fh)
    out = {"all": {"n": int(len(lab)), "mean": float(lab.mean()),
                   "sd": float(lab.std(ddof=0))}}
    for name in ("train", "val", "test"):
        s = lab.loc[split[name]]
        out[name] = {"n": int(len(s)), "mean": float(s.mean()),
                     "sd": float(s.std(ddof=0))}
    return out


def eval_val_metrics(log_name: str) -> dict[str, float]:
    """The `Validation results: [{...}]` dict a re-evaluation run printed."""
    path = osp.join(SLURM_OUT, log_name)
    text = open(path).read()
    marker = "Validation results: "
    start = text.index(marker) + len(marker)
    end = text.index("\n", start)
    payload = ast.literal_eval(text[start:end])[0]
    return {
        "pearson": float(payload["val/gene_interaction/Pearson"]),
        "mse": float(payload["val/gene_interaction/MSE"]),
        "rmse": float(payload["val/gene_interaction/RMSE"]),
        "path": path,
        "sha256": sha256_of(path),
    }


def main() -> None:
    rows: list[dict[str, object]] = []

    def add(quantity: str, value: object, source: str) -> None:
        rows.append({"quantity": quantity, "value": value, "source": source})

    # ---- 1. the prediction distribution ----------------------------------------------
    print(f"streaming {INFERENCE_PARQUET}")
    moments = stream_prediction_moments(INFERENCE_PARQUET)
    print(f"  n={moments['n']} mean={moments['mean']:.9f} sd={moments['sd']:.9f}")
    digest = sha256_of(INFERENCE_PARQUET)
    src = "inference parquet, streamed by row group"
    add("inference_parquet_path", INFERENCE_PARQUET, "pinned input")
    add("inference_parquet_sha256", digest, "sha256 of the file above")
    add("inference_parquet_bytes", osp.getsize(INFERENCE_PARQUET), "os.path.getsize")
    add("inference_parquet_row_groups", moments["row_groups"], "parquet metadata")
    add("inference_checkpoint", INFERENCE_CKPT,
        "checkpoint whose predictions the panel-12 selection consumed")
    add("pred_n", moments["n"], src)
    add("pred_mean", f"{moments['mean']:.9f}", src)
    add("pred_sd_pop", f"{moments['sd']:.9f}", src + ", population SD, ddof=0")
    add("pred_min", f"{moments['min']:.9f}", src)
    add("pred_max", f"{moments['max']:.9f}", src)

    # ---- 2. the label population -----------------------------------------------------
    labels = label_population()
    add("label_build_path", BUILD_DIR, "010 training build")
    for name, st in labels.items():
        lsrc = f"label_df.parquet + index_seed_42.json, split={name}"
        add(f"label_{name}_n", st["n"], lsrc)
        add(f"label_{name}_mean", f"{st['mean']:.9f}", lsrc)
        add(f"label_{name}_sd_pop", f"{st['sd']:.9f}", lsrc + ", population SD, ddof=0")

    # ---- 3. val metrics of the checkpoints actually used ------------------------------
    for label, run_id, log_name in CHECKPOINTS:
        m = eval_val_metrics(log_name)
        esrc = f"{log_name}, re-evaluation of the best-pearson checkpoint"
        add(f"ckpt_{label}_run_id", run_id, esrc)
        add(f"ckpt_{label}_val_pearson", f"{m['pearson']:.9f}", esrc)
        add(f"ckpt_{label}_val_mse", f"{m['mse']:.9f}", esrc)
        add(f"ckpt_{label}_val_rmse", f"{m['rmse']:.9f}", esrc)
        add(f"ckpt_{label}_eval_log_sha256", m["sha256"], f"sha256 of {log_name}")

    # ---- 4. W&B summaries, pulled live or the script fails ----------------------------
    api = wandb.Api(timeout=60)
    for label, run_id, _ in CHECKPOINTS:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        s = run.summary
        wsrc = f"W&B {WANDB_PROJECT}/{run_id} summary, last logged epoch"
        add(f"last_{label}_val_pearson", f"{float(s['val/gene_interaction/Pearson']):.9f}",
            wsrc)
        add(f"last_{label}_val_rmse", f"{float(s['val/gene_interaction/RMSE']):.9f}", wsrc)
        # `summary["epoch"]` is the end-of-run counter, one past the last epoch that
        # logged a val metric. Record the epoch the summary metrics above belong to.
        hist = run.history(keys=["epoch", "val/gene_interaction/Pearson"], pandas=True)
        hist = hist.dropna(subset=["val/gene_interaction/Pearson"])
        add(f"last_{label}_val_epoch", int(hist["epoch"].iloc[-1]),
            wsrc + ", epoch of the last logged val metric")
        add(f"last_{label}_n_val_epochs", int(len(hist)), wsrc + ", val epochs logged")
        add(f"last_{label}_group", run.group, wsrc)
        add(f"last_{label}_norm_std",
            f"{float(s['normalization/gene_interaction/std']):.9f}",
            wsrc + ", normalization/gene_interaction/std")
        add(f"last_{label}_norm_mean",
            f"{float(s['normalization/gene_interaction/mean']):.9f}",
            wsrc + ", normalization/gene_interaction/mean")

    failed = api.run(f"{WANDB_PROJECT}/{FAILED_RUN}")
    fs = failed.summary
    fsrc = f"W&B {WANDB_PROJECT}/{FAILED_RUN} summary"
    add("failed_run_id", FAILED_RUN, fsrc)
    add("failed_run_group", failed.group, fsrc)
    add("failed_run_val_pearson", f"{float(fs['val/gene_interaction/Pearson']):.9f}", fsrc)
    add("failed_run_val_rmse", f"{float(fs['val/gene_interaction/RMSE']):.9f}", fsrc)
    add("failed_run_train_pearson",
        f"{float(fs['train/gene_interaction/Pearson']):.9f}", fsrc)

    out = osp.join(RESULTS, "prediction_calibration_stats.csv")
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
