# experiments/019-echo-crispr-array/scripts/spread_comparison.py
# [[experiments.019-echo-crispr-array.scripts.spread_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/spread_comparison
"""Compare the SPREAD of our fitness measurements against the published Costanzo
SMF: (left) the distribution of fitness values (dynamic range overlap) and
(right) the per-strain SD (measurement precision). Caveat printed on the figure:
the two SDs are different replicate structures — ours is colony-level (~11
colonies), Costanzo's is screen-level (17/350 control screens) — so this is a
rough precision comparison, not apples-to-apples.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.utils import PLOT_PALETTE

load_dotenv()
IMG = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-echo-crispr-array")
EXP = osp.dirname(osp.dirname(osp.abspath(__file__)))
RES = osp.join(EXP, "results")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)


def main():
    fit = pd.read_csv(f"{RES}/plate5_fitness_by_volume.csv")
    ref = pd.read_csv(f"{RES}/reference_smf_12panel.csv").set_index("strain")
    ours = fit[fit.volume_nl == 5.0].set_index("strain")
    m = ours.join(ref).dropna(subset=["costanzo_smf"])

    fig, (axv, axs) = plt.subplots(1, 2, figsize=(5.6, 2.8))

    # --- left: distribution of fitness values (dynamic range / spread) ---
    ours_v = ours["fitness"].dropna().to_numpy()
    ref_v = ref["costanzo_smf"].dropna().to_numpy()
    bp = axv.boxplot(
        [ours_v, ref_v],
        labels=["ours\n(5 nL)", "Costanzo\nSMF"],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=PLOT_PALETTE[i], edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=0.8)
    rng = np.random.default_rng(19)
    for i, v in enumerate([ours_v, ref_v]):
        axv.scatter(
            np.full(len(v), i + 1) + rng.uniform(-0.12, 0.12, len(v)),
            v,
            s=10,
            color="black",
            alpha=0.6,
            zorder=3,
        )
    axv.set_ylabel("fitness (WT = 1)")
    axv.set_title("value spread / dynamic range")

    # --- right: DISTRIBUTION of SD values, ours split BY VOLUME (2.5 vs 5 nL)
    # vs Costanzo. Per-strain SD is already the bar-chart error bars, so this
    # compares the overall SD spread, per volume. ---
    vols = sorted(fit["volume_nl"].dropna().unique())
    sd_25 = fit[fit.volume_nl == vols[0]]["fitness_sd"].dropna().to_numpy()
    sd_5 = fit[fit.volume_nl == vols[1]]["fitness_sd"].dropna().to_numpy()
    ref_sd = ref["costanzo_sd"].dropna().to_numpy()
    groups = [sd_25, sd_5, ref_sd]
    labels = [
        f"ours {vols[0]} nL\n(n={len(sd_25)})",
        f"ours {vols[1]} nL\n(n={len(sd_5)})",
        f"Costanzo\n(n={len(ref_sd)})",
    ]
    colors = [PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2]]  # match bar chart
    bp2 = axs.boxplot(
        groups, labels=labels, patch_artist=True, widths=0.6, showfliers=False
    )
    for box, col in zip(bp2["boxes"], colors):
        box.set(facecolor=col, edgecolor="black", linewidth=0.5)
    for med in bp2["medians"]:
        med.set(color="black", linewidth=0.8)
    for i, v in enumerate(groups):
        axs.scatter(
            np.full(len(v), i + 1) + rng.uniform(-0.12, 0.12, len(v)),
            v,
            s=10,
            color="black",
            alpha=0.6,
            zorder=3,
        )
    axs.set_ylabel("SD of fitness")
    axs.set_title("SD spread by volume vs reference")
    axs.set_ylim(0, max(g.max() for g in groups) * 1.30)
    axs.text(
        0.02,
        0.97,
        f"median SD: {vols[0]} nL {np.median(sd_25):.3f}, "
        f"{vols[1]} nL {np.median(sd_5):.3f}, Costanzo {np.median(ref_sd):.3f}",
        transform=axs.transAxes,
        fontsize=5,
        va="top",
    )

    for ax in (axv, axs):
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.suptitle("Spread of our fitness vs published SMF", fontsize=7)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(
            osp.join(IMG, f"plate5_spread_vs_reference.{ext}"),
            dpi=200,
            bbox_inches="tight" if ext == "png" else None,
        )
    print(
        "our fitness range:",
        round(ours_v.min(), 2),
        "-",
        round(ours_v.max(), 2),
        "| Costanzo range:",
        round(ref_v.min(), 2),
        "-",
        round(ref_v.max(), 2),
    )
    print(
        "median SD  ours:",
        round(float(m["fitness_sd"].median()), 3),
        "Costanzo:",
        round(float(m["costanzo_sd"].median()), 3),
    )
    print(f"wrote {IMG}/plate5_spread_vs_reference.svg/.png")


if __name__ == "__main__":
    main()
