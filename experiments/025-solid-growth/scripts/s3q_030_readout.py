# experiments/025-solid-growth/scripts/s3q_030_readout.py
# [[experiments.025-solid-growth.scripts.s3q_030_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3q_030_readout
"""Read out the 030 closure arm on both splits from W&B, for Figure 2 d.

Two arms of the 030 per-entry build, each the closure pool with the composite embedding,
fitness co-training and the dataset token, seed 0 on IGB mmli: cgt_030_s3_r_tok_embfit_003
on the random split (a chain of three resumed runs, finished at epoch 183) and
cgt_030_s3_q_tok_embfit_004 on the query-pair-disjoint split (resumed once, in flight, synced
from its offline run directory). The Delta runs under the Q tag are another study's seeds
1 and 2, several restarts each, and are left out. One row per logged epoch (a resumed
segment overrides the epochs it re-logs) with validation Pearson on the split's held-out
triples and validation fitness Pearson. Writes results/s3q_030_readout.csv; the sheet draws
the mean of the last ten logged epochs of each arm and names the epoch reached.
"""
import os
import os.path as osp

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "025-solid-growth", "results")
PROJECT = "zhao-group/torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer"
ARMS = [
    ("random", "cgt_030_s3_r_tok_embfit_003", 0, ["xy3xnpau", "xy3xnpau-cont2421912", "u28pgzde"]),
    ("disjoint", "cgt_030_s3_q_tok_embfit_004", 0, ["rkwyav5i", "53qqijnu"]),
]


def main() -> None:
    api = wandb.Api(timeout=120)
    rows = []
    for split, config, seed, chain in ARMS:
        per_epoch: dict[int, dict] = {}
        for run_id in chain:
            run = api.run(f"{PROJECT}/{run_id}")
            h = run.history(keys=["epoch", "val/gene_interaction/Pearson", "val/fitness/Pearson"], pandas=True)
            h = h.dropna(subset=["val/gene_interaction/Pearson"]).groupby("epoch", as_index=False).last()
            for _, r in h.iterrows():
                per_epoch[int(r.epoch)] = {"split": split, "config": config, "seed": seed, "run_id": run_id, "run_url": run.url, "state": run.state, "epoch": int(r.epoch), "val_pearson": float(r["val/gene_interaction/Pearson"]), "val_fitness_pearson": float(r["val/fitness/Pearson"])}
        rows.extend(per_epoch[e] for e in sorted(per_epoch))
        tail = pd.DataFrame(rows[-10:])
        print(f"{split:8s} {config}: epochs 0 to {max(per_epoch)}, last run {chain[-1]} {tail.state.iloc[-1]}, mean of last 10 epochs {tail.val_pearson.mean():.4f}, max {max(v['val_pearson'] for v in per_epoch.values()):.4f}")
    df = pd.DataFrame(rows)
    path = osp.join(RESULTS_DIR, "s3q_030_readout.csv")
    df.to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
