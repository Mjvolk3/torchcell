# experiments/025-solid-growth/scripts/s3q_030_readout.py
# [[experiments.025-solid-growth.scripts.s3q_030_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3q_030_readout
"""Read out the 030 closure arm on the query-pair-disjoint split from W&B, for Figure 2 d.

The arm is cgt_030_s3_q_tok_embfit_004 (S3Q pool of the 030 per-entry build, composite
embedding, fitness co-training, dataset token), one seed, launched on IGB mmli and synced
from its offline run directory while it runs, so the table is a partial reading: every epoch
logged so far with validation Pearson on the 030 Q split's held-out triples and validation
fitness Pearson. Writes results/s3q_030_readout.csv; the sheet draws the latest epoch and
names it as in flight.
"""
import os
import os.path as osp

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
RESULTS_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "025-solid-growth", "results")
PROJECT = "zhao-group/torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer"
RUNS = [("cgt_030_s3_q_tok_embfit_004", 0, "53qqijnu")]


def main() -> None:
    api = wandb.Api(timeout=120)
    rows = []
    for config, seed, run_id in RUNS:
        run = api.run(f"{PROJECT}/{run_id}")
        h = run.history(keys=["epoch", "val/gene_interaction/Pearson", "val/fitness/Pearson"], pandas=True)
        h = h.dropna(subset=["val/gene_interaction/Pearson"]).groupby("epoch", as_index=False).last()
        for _, r in h.iterrows():
            rows.append({"config": config, "seed": seed, "run_id": run_id, "run_url": run.url, "state": run.state, "epoch": int(r.epoch), "val_pearson": float(r["val/gene_interaction/Pearson"]), "val_fitness_pearson": float(r["val/fitness/Pearson"])})
    df = pd.DataFrame(rows)
    path = osp.join(RESULTS_DIR, "s3q_030_readout.csv")
    df.to_csv(path, index=False)
    last = df.sort_values("epoch").iloc[-1]
    print(f"{len(df)} epochs; latest epoch {last.epoch} val Pearson {last.val_pearson:.4f}, max {df.val_pearson.max():.4f} at epoch {int(df.loc[df.val_pearson.idxmax(), 'epoch'])}; wrote {path}")


if __name__ == "__main__":
    main()
