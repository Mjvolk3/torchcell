# experiments/025-solid-growth/scripts/store_affine_readout.py
# [[experiments.025-solid-growth.scripts.store_affine_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/store_affine_readout

"""The store arm against the control, paired by seed, per epoch.

Reads the W&B histories of the constant-rate control (``cgt_s0_r_kl_ctrl_013``) and the
store arm (``cgt_s0_r_kl_store_033``) on the 010 replication subset, two seeds each, and
writes the validation Pearson on gene interaction per epoch, the paired difference
(store minus control) per epoch, the fixed-epoch read at the last epoch every run has
reached, and each run's best validation epoch (an upward-biased maximum, reported as
such). Writes ``results/store_affine_readout.json`` and a two-panel figure.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
PROJECT = "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"
METRIC = "val/gene_interaction/Pearson"
#: (arm, seed) -> run id; the four one-card runs of 2026-09-18 (jobs 2476 to 2479).
RUNS: dict[tuple[str, int], str] = {
    ("control", 42): "98yvinas",
    ("control", 43): "d0dubbdn",
    ("store", 42): "2yg4p0sf",
    ("store", 43): "bjvynfrk",
}


def history(run_id: str) -> dict[int, float]:
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    out: dict[int, float] = {}
    for row in run.scan_history(keys=["epoch", METRIC]):
        if row.get(METRIC) is not None:
            out[int(row["epoch"])] = float(row[METRIC])
    return out


def main() -> None:
    curves = {key: history(rid) for key, rid in RUNS.items()}
    seeds = sorted({s for _, s in RUNS})
    common = None
    for c in curves.values():
        common = set(c) if common is None else common & set(c)
    assert common, "no epoch reached by every run"
    fixed = max(common)
    paired = {
        s: {e: curves[("store", s)][e] - curves[("control", s)][e] for e in sorted(common)}
        for s in seeds
    }
    out: dict[str, Any] = {
        "runs": {f"{a}_s{s}": rid for (a, s), rid in RUNS.items()},
        "metric": METRIC,
        "fixed_epoch": fixed,
        "fixed_epoch_read": {
            f"{a}_s{s}": curves[(a, s)][fixed] for (a, s) in RUNS
        },
        "fixed_epoch_paired_difference_store_minus_control": {
            f"s{s}": paired[s][fixed] for s in seeds
        },
        "best_epoch_read_upward_biased_max": {
            f"{a}_s{s}": {"epoch": max(c, key=c.get), "value": max(c.values())}
            for (a, s), c in curves.items()
        },
        "paired_difference_by_epoch": {f"s{s}": paired[s] for s in seeds},
        "curves": {f"{a}_s{s}": c for (a, s), c in curves.items()},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "store_affine_readout.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none"})
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["wide"]), mm_to_in(50))
    )
    colors = {"control": PLOT_PALETTE[0], "store": PLOT_PALETTE[1]}
    styles = {42: "-", 43: "--"}
    for (a, s), c in curves.items():
        e = sorted(c)
        axes[0].plot(e, [c[k] for k in e], styles[s], color=colors[a], lw=0.8, label=f"{a}, seed {s}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation Pearson, gene interaction")
    for s in seeds:
        e = sorted(paired[s])
        axes[1].plot(e, [paired[s][k] for k in e], styles[s], color=PLOT_PALETTE[2], lw=0.8, label=f"store minus control, seed {s}")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("paired difference")
    for ax in axes:
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    stem = osp.join(ASSET_IMAGES_DIR, "025-solid-growth", f"store_affine_readout_{timestamp()}")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print(json.dumps({k: v for k, v in out.items() if k not in ("curves", "paired_difference_by_epoch")}, indent=1))
    print(stem + ".svg")


if __name__ == "__main__":
    main()
