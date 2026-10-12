# experiments/030-solid-growth-multi/scripts/wandb_seed_histories.py
# [[experiments.030-solid-growth-multi.scripts.wandb_seed_histories]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/wandb_seed_histories
"""Per-seed W&B histories of the 030 transformer arms, shared by the Figure 2 gate scripts.

A 030 arm is a W&B tag; its rank-0 runs are the ones whose summary carries the metric;
a run's seed is its config's ``seed``. The runs of one seed are stitched by epoch (a run
and its continuations), the last logged value of an epoch winning, and seeds are never
mixed: a second seed launched later is its own history, not a continuation of the first.

This module imports only pandas and wandb on purpose. ``lower_order_baselines_030.py``
pulls in the training-side imports, and importing it resets matplotlib's rcParams to a
16 pt style, which the gate's sheet script must not inherit.
"""

from __future__ import annotations

import pandas as pd

WANDB_ENTITY = "zhao-group"
WANDB_PROJECT = "torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer"
WANDB_ARM_TAG = {"R": "cgt_030_s3_r_tok_embfit_003", "Q": "cgt_030_s3_q_tok_embfit_004"}
WANDB_WINDOW = 10
#: The seed an arm's headline number reports. Every seed that carries the tag is recorded
#: and, once it has logged at least WANDB_WINDOW epochs, drawn as its own bar.
WANDB_HEADLINE_SEED = 0


def wandb_seed_histories(tag: str, key: str) -> dict[int, pd.Series]:
    """Per seed, the epoch-indexed history of ``key`` over the rank-0 runs tagged ``tag``.

    Each series carries its run ids in ``attrs["run_ids"]``."""
    import wandb

    api = wandb.Api(timeout=120)
    runs = [
        r
        for r in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": tag})
        if key in r.summary
    ]
    assert runs, f"no rank-0 run tagged {tag} carries {key}"
    out: dict[int, pd.Series] = {}
    for seed in sorted({int(r.config["seed"]) for r in runs}):
        mine = [r for r in runs if int(r.config["seed"]) == seed]
        frames = [pd.DataFrame(r.scan_history(keys=["epoch", key])).dropna() for r in mine]
        out[seed] = pd.concat(frames).groupby("epoch")[key].last().sort_index()
        out[seed].attrs["run_ids"] = sorted(r.id for r in mine)
    return out
