# experiments/025-solid-growth/scripts/disjoint_embedding_readout.py
# [[experiments.025-solid-growth.scripts.disjoint_embedding_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/disjoint_embedding_readout
"""Read out the query-pair-disjoint sequence-embedding arms from W&B.

One row per config tag: the rank-0 run (the one carrying `val/gene_interaction/Pearson`),
epochs logged, max validation Pearson and its epoch (an upward-biased order statistic over
the epochs run), the value at epoch 29 (the protocol's fixed reading), the mean over epochs
10 to 29 (a window average that is not a max), the mean over epochs 60 to 99 for the
100-epoch runs, validation point loss (z-scored MSE) at epoch 29, train Pearson and the
graph penalty at the last epoch. Writes results/disjoint_embedding_readout.csv and prints a
markdown table. Runs are selected by the config-name tag the run script attaches
(`cgt_s0_q_kl_*`), so the table follows the configs, not hand-typed run ids.
"""

import os
import os.path as osp
import statistics

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
PROJECT = "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"

ARMS = [
    ("cgt_s0_q_kl_ctrl_016", "learnable table (control)", 30),
    ("cgt_s0_q_kl_emb_017", "composite: promoter + CaLM + ProtT5 + terminator", 30),
    ("cgt_s0_q_kl_calm_020", "CaLM alone", 30),
    ("cgt_s0_q_kl_prot_021", "ProtT5 alone", 30),
    ("cgt_s0_q_kl_emb_022", "composite", 100),
    ("cgt_s0_q_kl_calm_023", "CaLM alone", 100),
    ("cgt_s0_q_kl_prot_024", "ProtT5 alone", 100),
    ("cgt_s0_q_kl_fudt_026", "promoter + terminator alone", 100),
]
KEYS = (
    "val/gene_interaction/Pearson",
    "val/point_loss",
    "train/gene_interaction/Pearson",
    "train/graph_reg_loss",
)


def per_epoch(run) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {k: {} for k in KEYS}
    for row in run.scan_history():
        e = row.get("epoch")
        if e is None:
            continue
        for k in KEYS:
            if row.get(k) is not None:
                out[k][int(e)] = float(row[k])
    return out


def window_mean(d: dict[int, float], lo: int, hi: int) -> float | None:
    vals = [v for e, v in d.items() if lo <= e <= hi]
    return statistics.fmean(vals) if len(vals) == hi - lo + 1 else None


def main() -> None:
    api = wandb.Api()
    rows = []
    for cfg, label, budget in ARMS:
        runs = list(api.runs(PROJECT, filters={"tags": {"$in": [cfg]}}))
        picked = None
        for r in runs:
            h = per_epoch(r)
            if h["val/gene_interaction/Pearson"]:
                picked = (r, h)
                break
        if picked is None:
            print(f"{cfg}: no rank-0 run with validation history yet")
            continue
        r, h = picked
        val = h["val/gene_interaction/Pearson"]
        best_epoch = max(val, key=val.get)
        last = max(val)
        rows.append(
            {
                "config": cfg,
                "arm": label,
                "budget_epochs": budget,
                "run_id": r.id,
                "run_url": r.url,
                "epochs_logged": len(val),
                "complete": len(val) >= budget,
                "val_pearson_max": round(val[best_epoch], 4),
                "val_pearson_max_epoch": best_epoch,
                "val_pearson_epoch29": round(val[29], 4) if 29 in val else None,
                "val_pearson_mean_ep10_29": (
                    round(m, 4) if (m := window_mean(val, 10, 29)) is not None else None
                ),
                "val_pearson_mean_ep60_99": (
                    round(m, 4) if (m := window_mean(val, 60, 99)) is not None else None
                ),
                "val_point_loss_epoch29": (
                    round(h["val/point_loss"][29], 4) if 29 in h["val/point_loss"] else None
                ),
                "val_point_loss_last": round(h["val/point_loss"][max(h["val/point_loss"])], 4),
                "train_pearson_last": round(
                    h["train/gene_interaction/Pearson"][
                        max(h["train/gene_interaction/Pearson"])
                    ],
                    4,
                ),
                "graph_reg_loss_last": round(
                    h["train/graph_reg_loss"][max(h["train/graph_reg_loss"])], 3
                ),
                "last_epoch": last,
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, "disjoint_embedding_readout.csv")
    df.to_csv(path, index=False)
    cols = [
        "config", "budget_epochs", "epochs_logged", "val_pearson_max",
        "val_pearson_max_epoch", "val_pearson_epoch29", "val_pearson_mean_ep10_29",
        "val_pearson_mean_ep60_99", "val_point_loss_epoch29", "val_point_loss_last",
        "train_pearson_last", "graph_reg_loss_last",
    ]
    sub = df[cols]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in sub.iterrows():
        print("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    print()
    for _, r in df.iterrows():
        print(r["config"], r["run_url"])
    print("wrote", path)


if __name__ == "__main__":
    main()
