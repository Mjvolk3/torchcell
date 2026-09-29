# experiments/033-env-chemgen-pooled/scripts/pooled_response_profile.py
# [[experiments.033-env-chemgen-pooled.scripts.pooled_response_profile]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/033-env-chemgen-pooled/scripts/pooled_response_profile
"""What the pooled chemogenomic store holds, per source, from the built label table.

Build 001 (slurm 2929) writes one scalar per cell under the single phenotype label
``environment_response``, so the store's whole label surface is 6,042,771 floats drawn
from four screens that do not share a response scale. This reads that table and the
processed store's own dataset index, and reports per source: the cell count, the
distribution of the response, and the sign convention.

The sign matters more than the scale. Hillenmeyer reports a sensitivity score, where a
larger value means the strain is sicker; the other three report a ratio or a z score,
where a smaller value means the strain is sicker. Pooling without orienting them trains
the decoder against two opposed definitions of the same word.

Reads only ``processed/label_df.parquet`` and ``processed/dataset_name_index.json``, so
it does not open the 106 GB LMDB. Writes ``results/pooled_response_profile.csv`` and a
figure to ``ASSET_IMAGES_DIR/033-env-chemgen-pooled/``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from torchcell.timestamp import timestamp
from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_ROOT = "/db/experiments/033-env-chemgen-pooled-001-pooled-build"
PROCESSED = osp.join(BUILD_ROOT, "processed")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "033-env-chemgen-pooled", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "033-env-chemgen-pooled")

#: Source order is the 031 planning order: application target first, then by size.
DISPLAY: dict[str, str] = {
    "EnvChemgenVanacloig2022Dataset": "Vanacloig 2022",
    "HetHillenmeyer2008Dataset": "Hillenmeyer HET",
    "EnvChemgenHoepfner2014Dataset": "Hoepfner 2014",
    "EnvChemgenWildenhain2015Dataset": "Wildenhain 2015",
}
#: Which direction of the reported statistic means a sick strain, from the 031 axis work.
POLARITY: dict[str, str] = {
    "EnvChemgenVanacloig2022Dataset": "smaller is sicker",
    "HetHillenmeyer2008Dataset": "larger is sicker",
    "EnvChemgenHoepfner2014Dataset": "smaller is sicker",
    "EnvChemgenWildenhain2015Dataset": "smaller is sicker",
}


def load_labels() -> pd.DataFrame:
    """The label table with the source of each cell attached."""
    labels = pd.read_parquet(osp.join(PROCESSED, "label_df.parquet"))
    with open(osp.join(PROCESSED, "dataset_name_index.json")) as f:
        index = json.load(f)
    source = pd.concat(
        [pd.DataFrame({"index": idx, "dataset": name}) for name, idx in index.items()],
        ignore_index=True,
    )
    merged = labels.merge(source, on="index", how="left", validate="one_to_one")
    assert merged["dataset"].notna().all(), "a cell carries no dataset in the index"
    return merged


def profile(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, display in DISPLAY.items():
        r = merged.loc[merged["dataset"] == name, "environment_response"]
        rows.append(
            {
                "dataset": display,
                "cells": len(r),
                "polarity": POLARITY[name],
                "mean": round(float(r.mean()), 4),
                "sd": round(float(r.std()), 4),
                "min": round(float(r.min()), 4),
                "q01": round(float(r.quantile(0.01)), 4),
                "median": round(float(r.median()), 4),
                "q99": round(float(r.quantile(0.99)), 4),
                "max": round(float(r.max()), 4),
                "frac_negative": round(float((r < 0).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def plot(merged: pd.DataFrame, path_stem: str) -> None:
    """One panel per source, each on its OWN x-range.

    A shared axis is the wrong picture here. Wildenhain's 1st percentile is -44 while
    Hillenmeyer's whole range is about 7 wide, so one axis collapses three of the four
    into a spike at zero. Separate ranges are the honest display, and the differing tick
    values are themselves the finding.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(
        len(DISPLAY), 1, figsize=(mm_to_in(88), mm_to_in(88)), sharey=False
    )
    for ax, (i, (name, display)) in zip(axes, enumerate(DISPLAY.items()), strict=True):
        r = merged.loc[merged["dataset"] == name, "environment_response"]
        lo, hi = float(r.quantile(0.01)), float(r.quantile(0.99))
        ax.hist(
            r[(r >= lo) & (r <= hi)],
            bins=140,
            histtype="stepfilled",
            density=True,
            color=PLOT_PALETTE[i],
            edgecolor=PLOT_PALETTE[i],
            linewidth=0.5,
        )
        ax.axvline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.set_title(
            f"{display}, n={len(r):,}, {POLARITY[name]}", fontsize=6, loc="left", pad=2
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[-1].set_xlabel(
        "environment_response, 1st to 99th percentile of its own source"
    )
    fig.tight_layout(h_pad=0.8)
    fig.savefig(f"{path_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{path_stem}.svg")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    with open(osp.join(PROCESSED, "gene_set.json")) as f:
        gene_set = json.load(f)
    print(f"gene set: {len(gene_set)} genes perturbed across the four sources")

    merged = load_labels()
    print(f"label table: {len(merged):,} cells")

    summary = profile(merged)
    summary.to_csv(osp.join(RESULTS_DIR, "pooled_response_profile.csv"), index=False)
    print(summary.to_string(index=False))

    plot(merged, osp.join(IMAGE_DIR, f"pooled_response_distributions_{timestamp()}"))


if __name__ == "__main__":
    main()
