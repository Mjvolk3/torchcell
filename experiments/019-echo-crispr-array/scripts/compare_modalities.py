# experiments/019-echo-crispr-array/scripts/compare_modalities.py
# [[experiments.019-echo-crispr-array.scripts.compare_modalities]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/compare_modalities
"""Does the imaging MODALITY change the fitness scores? Same Plate 5, two
captures: dark-field (pixl, `Original.png`, bright colonies on dark agar) vs
transillumination (iPhone on a light box, `IMG_4572_cropped.jpg`, dark colonies
on bright agar; imaged through the base so the layout is mirrored). Both are
quantified (auto-polarity), registered to the same picklist via the blank
pattern (which also resolves the mirror), normalized, and scored WT-anchored.
We then compare per-strain fitness between the two.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.sga import (
    NormalizationConfig,
    normalize_plate,
    quantify_plate_image,
    read_echo_picklist,
    resolve_orientation,
    score_plate,
    score_table,
)
from torchcell.utils import PLOT_PALETTE

load_dotenv()
ASSET = os.environ["ASSET_IMAGES_DIR"]
EXP = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA, RES = osp.join(EXP, "data"), osp.join(EXP, "results")
IMG = osp.join(ASSET, "019-echo-crispr-array")
PICK = osp.join(DATA, "ECHO_picklist_Plate5_384_OD1_2p5-5nL.csv")


def fitness_of(image, polarity, cfg, layout, label):
    grid = quantify_plate_image(image, polarity=polarity)
    merged, op, agree = resolve_orientation(grid, layout)
    print(
        f"  {label}: orientation={op} ({agree:.0%} blank agreement), "
        f"{(grid['size'] > 0).sum()} colonies"
    )
    df = normalize_plate(merged, cfg)
    rep = score_plate(df, cfg, plate_id=label)
    t = score_table(rep).set_index("strain")["relative_fitness"]
    return t, op, agree


def main():
    cfg = NormalizationConfig()
    layout = read_echo_picklist(PICK)
    dark, op_d, ag_d = fitness_of(
        osp.join(DATA, "Original.png"), "auto", cfg, layout, "dark-field"
    )
    trans, op_t, ag_t = fitness_of(
        osp.join(DATA, "IMG_4572_cropped.jpg"), "auto", cfg, layout, "transillumination"
    )

    cmp = pd.DataFrame({"dark_field": dark, "transillumination": trans}).dropna()
    cmp = cmp.drop(index=[cfg.blank_name], errors="ignore")
    cmp.to_csv(osp.join(RES, "plate5_modality_comparison.csv"))
    r = float(np.corrcoef(cmp["dark_field"], cmp["transillumination"])[0, 1])
    r2 = r**2
    print(
        f"\nper-strain fitness correlation dark-field vs transillum: r = {r:.2f} "
        f"(r^2 = {r2:.2f}, n={len(cmp)})"
    )
    print(cmp.round(3).sort_values("dark_field").to_string())

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(
        cmp["dark_field"],
        cmp["transillumination"],
        s=22,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    lims = [min(0.3, cmp.min().min() * 0.95), max(1.15, cmp.max().max() * 1.05)]
    ax.plot(lims, lims, ls="--", color=PLOT_PALETTE[5], lw=0.6)
    for s in cmp.index:
        ax.annotate(
            s,
            (cmp.loc[s, "dark_field"], cmp.loc[s, "transillumination"]),
            fontsize=4,
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax.text(
        0.05,
        0.92,
        f"r = {r:.2f},  $r^2$ = {r2:.2f}  (n={len(cmp)})",
        transform=ax.transAxes,
        fontsize=6,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("fitness — dark-field (pixl)", fontsize=6)
    ax.set_ylabel("fitness — transillumination (iPhone)", fontsize=6)
    ax.set_title("Imaging modality barely changes scores", fontsize=6.5, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(True)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(
            osp.join(IMG, f"plate5_modality_comparison.{ext}"),
            dpi=200,
            bbox_inches="tight" if ext == "png" else None,
        )
    print(f"wrote {IMG}/plate5_modality_comparison.svg/.png")


if __name__ == "__main__":
    main()
