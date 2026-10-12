# experiments/028-gene-essentiality/scripts/lp_probe_readout.py
# [[experiments.028-gene-essentiality.scripts.lp_probe_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/lp_probe_readout

r"""The layer that solves, probed per epoch: feasibility, slack, binding bounds, gradient.

The round figures show the score; this one shows what the program did on the way, on
the cells whose flux is a linear program (rounds 15c, 15d and 18), seed mean per arm:

- **a**, validation AUROC per epoch.
- **b**, the fraction of genotypes whose program was infeasible (the hard program's
  absorbing state) as a solid line, and, on the elastic program, the fraction whose
  maintenance flux was left unmet (``feas_lp_slack_row_frac``) as a dotted line.
- **c**, the fraction of gene-owned reactions at their enzyme-capacity ceiling, which
  is where a gene-owned bound binds and a dual exists.
- **d**, the fraction of genotypes on which any gene-owned bound carries a dual
  (``feas_lp_gene_dual_row_frac``, logged from round 18), which is the fraction on
  which the backbone receives any gradient through the solve.
- **e**, the flux layer's share of the summed per-module gradient norm, and on the
  round-18 cells the abundance head's share as a dotted line.
- **f**, the preprocessor's gradient norm.

    python experiments/028-gene-essentiality/scripts/lp_probe_readout.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from grad_flow_readout import MODULES, seed_mean, series  # noqa: E402
from wave12_readout import ARM_LABELS, arm_of  # noqa: E402

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")

#: (glob, arm) in figure order: the hard program's forced cells, then the elastic ones,
#: then the elastic gated cell. Six arms, six hues; the gated arm is dashed.
ARMS: list[tuple[str, str]] = [
    ("w30_lpcapbio100_h12l1p1_e200_s*.json", "lpcapbio100"),
    ("w31_lpcapbio1000_h12l1p1_e200_s*.json", "lpcapbio1000"),
    ("w32_lpelbio100_h12l1p1_e200_s*.json", "lpelbio100"),
    ("w32_lpelbio1000_h12l1p1_e200_s*.json", "lpelbio1000"),
    ("w32_lpelpoolbio100_h12l1p1_e200_s*.json", "lpelpoolbio100"),
    ("w32_lpelgate100_h12l1p1_e200_s*.json", "lpelgate100"),
]
HUES = [PLOT_PALETTE[i] for i in (0, 1, 2, 3, 4, 5)]


def infeasible(history: list[dict]) -> np.ndarray:
    """The infeasible fraction: the round-18 key where logged, else the zero-flux one."""
    key = (
        "feas_lp_infeasible_frac"
        if "feas_lp_infeasible_frac" in history[-1]
        else "feas_lp_zero_flux_frac"
    )
    return series(history, key)


def main() -> None:
    apply_paper_style()
    arms: list[dict] = []
    for pattern, arm in ARMS:
        files = sorted(glob.glob(osp.join(RESULTS, pattern)))
        files = [f for f in files if arm_of(osp.basename(f)[:-5]) == arm]
        if not files:
            continue
        cols: dict[str, list[np.ndarray]] = {
            k: []
            for k in (
                "val",
                "infeasible",
                "slack",
                "capacity",
                "dual",
                "share",
                "abundance",
                "preprocessor",
            )
        }
        for f in files:
            with open(f) as fh:
                h = json.load(fh)["history"]
            parts = np.stack([series(h, k) for k in MODULES if k in h[-1]])
            total = np.nansum(parts, axis=0)
            cols["val"].append(series(h, "val_pearson"))
            cols["infeasible"].append(infeasible(h))
            cols["slack"].append(series(h, "feas_lp_slack_row_frac"))
            cols["capacity"].append(series(h, "feas_capacity_active_frac"))
            cols["dual"].append(series(h, "feas_lp_gene_dual_row_frac"))
            cols["share"].append(series(h, "grad_norm_flux_layer_mean") / total)
            cols["abundance"].append(
                series(h, "grad_norm_flux_layer.abundance_mean") / total
            )
            cols["preprocessor"].append(
                series(h, "grad_norm_embedding_preprocessor_mean")
            )
        arms.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "n": len(files),
                **{k: seed_mean(v) for k, v in cols.items()},
            }
        )

    width = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(2, 3, figsize=(width, mm_to_in(105)))
    ax = axes.ravel()
    for i, a in enumerate(arms):
        color = HUES[i % len(HUES)]
        style = "--" if "gate" in a["arm"] else "-"
        x = np.arange(len(a["val"]))
        ax[0].plot(x, a["val"], style, color=color, lw=0.9, label=a["label"])
        ax[1].plot(x, a["infeasible"], style, color=color, lw=0.9)
        if np.isfinite(a["slack"]).any():
            ax[1].plot(x, a["slack"], ":", color=color, lw=0.9)
        ax[2].plot(x, a["capacity"], style, color=color, lw=0.9)
        if np.isfinite(a["dual"]).any():
            ax[3].plot(x, a["dual"], style, color=color, lw=0.9)
        ax[4].plot(x, a["share"], style, color=color, lw=0.9)
        if np.isfinite(a["abundance"]).any():
            ax[4].plot(x, a["abundance"], ":", color=color, lw=0.9)
        ax[5].plot(x, a["preprocessor"], style, color=color, lw=0.9)
    ax[0].set_ylabel("validation AUROC")
    ax[0].set_ylim(0.4, 1.0)
    ax[1].set_ylabel("infeasible (solid), unmet maintenance (dotted)")
    ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_ylabel("reactions at their capacity ceiling")
    ax[3].set_ylabel("genotypes with a gene-owned dual")
    ax[3].set_ylim(-0.02, 1.02)
    ax[4].set_yscale("log")
    ax[4].set_ylabel("flux layer share (solid), abundance head (dotted)")
    ax[5].set_yscale("log")
    ax[5].set_ylabel("preprocessor gradient norm")
    for axis, letter in zip(ax, "abcdef", strict=True):
        axis.set_xlabel("epoch")
        axis.grid(axis="y", which="major", linewidth=0.3, color="0.85")
        axis.text(
            -0.3, 1.06, letter, transform=axis.transAxes, fontsize=8, fontweight="bold"
        )
    handles, labels = ax[0].get_legend_handles_labels()
    ncol = 2
    rows = int(np.ceil(len(labels) / ncol))
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.09,
        top=1.0 - (rows * 3.2 + 5.0) / 105,
        wspace=0.5,
        hspace=0.45,
    )
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 1.0),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"lp_probe_{timestamp()}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    out = osp.join(RESULTS, "lp_probe_readout.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "figure": f"{base}.svg",
                "arms": [
                    {
                        "arm": a["arm"],
                        "label": a["label"],
                        "n_seeds": a["n"],
                        "infeasible_first_last": [
                            float(np.nan_to_num(a["infeasible"][0])),
                            float(np.nan_to_num(a["infeasible"][-1])),
                        ],
                        "slack_row_frac_last": float(np.nan_to_num(a["slack"][-1])),
                        "capacity_active_last": float(
                            np.nan_to_num(a["capacity"][-1])
                        ),
                        "gene_dual_row_frac_last": float(np.nan_to_num(a["dual"][-1])),
                        "flux_share_last": float(np.nan_to_num(a["share"][-1])),
                    }
                    for a in arms
                ],
            },
            fh,
            indent=2,
        )
    print(f"{base}.svg\n{out}")


if __name__ == "__main__":
    main()
