# experiments/039-inhibitor-combinations-wetlab/scripts/plot_inhibitor_runs.py
# [[experiments.039-inhibitor-combinations-wetlab.scripts.plot_inhibitor_runs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/039-inhibitor-combinations-wetlab/scripts/plot_inhibitor_runs
"""The wet-lab inhibitor runs of 2021 as they stand in the thesis archive.

Three Bioscreen C experiments on the MAGIC strain BY4742-iAID6 in YPD, read from the
archive on ``/bulk`` (``thesis_archive/README.md``, every file sha256-pinned in its
``MANIFEST.tsv``):

ex21
    single-inhibitor titrations: six sorghum-hydrolysate inhibitors (furfural, acetic
    acid, 5-HMF, formic acid, levulinic acid, lactic acid), nine concentrations each,
    three biological replicates. Wells run in blocks of ten per replicate: the block's
    first well is the uninhibited control, the next nine the titration from the highest
    concentration down (``MV_ex21_data_processing_v2.ipynb``). The concentration labels
    are the notebook's, verbatim, in g/L.
ex23
    all 63 non-empty combinations of the six inhibitors, one concentration each (the
    ``C2`` column of ``inhibitors.xlsx``: FF 6, AA 4, HMF 2.522, FA 1, LVA 2, LA 40 g/L),
    three biological replicates, 200 wells.
ex26
    the furfural x acetic acid isobole: a 10 x 10 grid of concentrations, two plates
    (``n2--plotting_isoboles.ipynb``: FF 0 to 3 g/L in rows, AA 0 to 3.6 g/L in columns).

FITNESS, as ``process_bsc.py`` of the analysis repo defined it: the mean wild-type
generation time divided by the well's generation time, so 1 is wild-type growth and
smaller is slower. A well whose curve never rose (the trait file records a lag of 48 h
and no generation time) has no fitness; it is NOT missing, it is a culture that did not
grow within the run, and every plot shows those wells as their own category.

Writes, under ``results/``: ``ex23_conditions.csv`` (one row per combination: the order,
the fitness mean and sd over replicates, whether it grew), ``ex21_titration.csv`` (one
row per inhibitor, replicate and step) and ``ex26_isobole.csv`` (one row per grid cell);
and the figures to ``$ASSET_IMAGES_DIR/039-inhibitor-combinations-wetlab/``.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ARCHIVE = "/bulk/thesis/thesis_archive/data"
ANALYSIS = osp.join(ARCHIVE, "03_analysis_code", "multi_knockout")
EXPERIMENT = osp.join(
    os.environ["EXPERIMENT_ROOT"], "039-inhibitor-combinations-wetlab"
)
RESULTS = osp.join(EXPERIMENT, "results")
IMAGES = osp.join(os.environ["ASSET_IMAGES_DIR"], "039-inhibitor-combinations-wetlab")

INHIBITORS = ["FF", "AA", "HMF", "FA", "LVA", "LA"]
NAMES = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-HMF",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
#: ex23: one concentration per inhibitor, g/L (inhibitors.xlsx, column C2)
EX23_G_PER_L = {"FF": 6.0, "AA": 4.0, "HMF": 2.522, "FA": 1.0, "LVA": 2.0, "LA": 40.0}
#: ex21: the first well of each inhibitor's first replicate block, and the notebook's
#: concentration labels (g/L, highest first), verbatim; two labels are out of order in
#: the notebook (FF 28 between 24 and 12, FA 1.25 between 0.25 and 0.063) and are kept
#: as recorded, so the x axis is the titration step, not the concentration.
EX21_BLOCK_START = {"FF": 1, "AA": 31, "HMF": 61, "FA": 101, "LVA": 131, "LA": 161}
EX21_LABELS = {
    "FF": ["30", "24", "28", "12", "6", "3", "1.5", "0.75", "0.375"],
    "AA": ["20", "16", "12", "8", "4", "2", "1", "0.5", "0.25"],
    "HMF": ["12.5", "10", "7.5", "5", "2.52", "1.26", "0.63", "0.32", "0.16"],
    "FA": ["5", "4", "3", "2", "1", "0.5", "0.25", "1.25", "0.063"],
    "LVA": ["10", "8", "6", "4", "2", "1", "0.5", "0.25", "0.125"],
    "LA": ["200", "160", "120", "80", "40", "20", "10", "5", "2.5"],
}
#: ex26: grid concentrations, g/L, from n2--plotting_isoboles.ipynb
EX26_FF = [0, 0.33, 0.67, 1, 1.33, 1.67, 2, 2.33, 2.67, 3]
EX26_AA = [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
NO_GROWTH = "#E6E6E6"

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
    }
)


class Traits(BaseModel):
    """One Bioscreen C trait file: generation time per well, NaN where no growth."""

    run: str
    gt: dict[int, float]  # well -> generation time, NaN where the curve never rose

    @classmethod
    def read(cls, run: str, path: str) -> Traits:
        """Parse a ``*_Traits.txt`` / ``*.tsv`` file (``Container Name``, ``GT``)."""
        d = pd.read_csv(path, sep="\t")
        well = d["Container Name"].str.split(" ", expand=True)[1].astype(int)
        return cls(run=run, gt=dict(zip(well, d["GT"].astype(float), strict=True)))


def box(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)


def save(fig: plt.Figure, name: str) -> list[str]:
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"{name}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    return [stem + ".svg", stem + ".png"]


# ---- ex23: the 63 combinations ------------------------------------------- #
def ex23_conditions() -> pd.DataFrame:
    """One row per combination with its fitness over replicates and whether it grew."""
    pre = pd.read_csv(
        osp.join(ANALYSIS, "experiments", "bsc", "ex23", "MV_ex23_preprocessed.csv")
    )
    wt_gt = pre.loc[pre["name"] == "WT", "gt"].mean()
    pre["fitness"] = wt_gt / pre["gt"]
    rows = []
    # "WT" is the uninhibited control (8 wells), "blank" uninoculated YPD (3 wells)
    for name, g in pre[~pre["name"].isin(["WT", "blank"])].groupby("name"):
        members = name.split("_")
        rows.append(
            {
                "name": name,
                "order": len(members),
                "members": ";".join(members),
                "n_wells": len(g),
                "n_grew": int(g["gt"].notna().sum()),
                "fitness_mean": g["fitness"].mean(),
                "fitness_sd": g["fitness"].std(),
            }
        )
    out = pd.DataFrame(rows)
    out["grew"] = out["n_grew"] > 0
    assert len(out) == 63, len(out)
    return out.sort_values(
        ["order", "fitness_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def plot_ex23_conditions(cond: pd.DataFrame) -> list[str]:
    """Every combination on one axis, ordered by the number of inhibitors."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(65)))
    x = np.arange(len(cond))
    handles = []
    for k, color in zip(range(1, 7), PLOT_PALETTE, strict=False):
        sel = cond["order"] == k
        grew = sel & cond["grew"]
        ax.bar(
            x[grew],
            cond.loc[grew, "fitness_mean"],
            yerr=cond.loc[grew, "fitness_sd"],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            width=0.8,
            error_kw={"elinewidth": 0.5, "capsize": 1, "capthick": 0.5},
        )
        none = sel & ~cond["grew"]
        ax.bar(
            x[none], 0.04, color=NO_GROWTH, edgecolor="black", linewidth=0.4, width=0.8
        )
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.4,
                label=f"{k} inhibitor{'s' if k > 1 else ''}",
            )
        )
    handles.append(
        Patch(
            facecolor=NO_GROWTH,
            edgecolor="black",
            linewidth=0.4,
            label="no growth in 48 h",
        )
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cond["name"].str.replace("_", "+"), rotation=90, fontsize=5)
    ax.set_xlim(-0.8, len(cond) - 0.2)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.set_ylabel("fitness (wild-type GT / GT)")
    ax.set_xlabel("inhibitor combination, ex23 (one concentration each)")
    ax.legend(handles=handles, frameon=False, ncol=7, loc="upper right")
    box(ax)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.97, bottom=0.36)
    return save(fig, "ex23_combinations_fitness")


def plot_ex23_pairs(cond: pd.DataFrame) -> list[str]:
    """The 6 x 6 matrix of single (diagonal) and pair fitness."""
    by_name = cond.set_index("name")
    m = np.full((6, 6), np.nan)
    grew = np.ones((6, 6), dtype=bool)
    for i, a in enumerate(INHIBITORS):
        for j, b in enumerate(INHIBITORS):
            if i == j:
                row = by_name.loc[a]
            else:
                key = f"{a}_{b}" if f"{a}_{b}" in by_name.index else f"{b}_{a}"
                row = by_name.loc[key]
            m[i, j] = row["fitness_mean"]
            grew[i, j] = bool(row["grew"])
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(70)))
    cmap = matplotlib.colormaps["Oranges_r"].copy()
    cmap.set_bad(NO_GROWTH)
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            text = f"{m[i, j]:.2f}" if grew[i, j] else "none"
            ax.text(j, i, text, ha="center", va="center", fontsize=5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([f"{k}\n{EX23_G_PER_L[k]:g} g/L" for k in INHIBITORS])
    ax.set_yticklabels([f"{k} {EX23_G_PER_L[k]:g} g/L" for k in INHIBITORS])
    ax.set_title("ex23 singles (diagonal) and pairs; gray = no growth in 48 h")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("fitness (wild-type GT / GT)")
    cbar.outline.set_linewidth(0.5)
    box(ax)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.92, bottom=0.14)
    return save(fig, "ex23_pair_matrix")


def plot_ex23_curves() -> list[str]:
    """Blanked OD curves of the control and the six single inhibitors, mean over wells."""
    ex23 = osp.join(ANALYSIS, "experiments", "bsc", "ex23")
    curves = pd.read_csv(
        osp.join(ex23, "MV_ex23_Magic_inhibitor_combinations_curves_Processed.tsv"),
        sep="\t",
    )
    pre = pd.read_csv(osp.join(ex23, "MV_ex23_preprocessed.csv"))
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(55)))
    series = [("WT", "black")] + list(zip(INHIBITORS, PLOT_PALETTE, strict=False))
    for name, color in series:
        wells = pre.loc[pre["name"] == name, "well"].astype(int)
        y = curves[[f"Well {w}" for w in wells]].mean(axis=1)
        label = "no inhibitor" if name == "WT" else f"{name} {EX23_G_PER_L[name]:g} g/L"
        ax.plot(
            curves["Time"],
            y,
            color=color,
            linewidth=0.8,
            linestyle="--" if name == "WT" else "-",
            label=label,
        )
    ax.set_xlim(0, 48)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("OD600, blanked (mean over wells)")
    ax.set_title("ex23: the six single inhibitors at the combination concentration")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    box(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.16)
    return save(fig, "ex23_single_inhibitor_curves")


# ---- ex21: the titrations ------------------------------------------------- #
def ex21_titration() -> pd.DataFrame:
    traits = Traits.read(
        "ex21",
        osp.join(ANALYSIS, "inhibitor_tolerance", "MV_ex21_inhibitor_titration.tsv"),
    )
    controls = [
        traits.gt[s + 10 * r] for s in EX21_BLOCK_START.values() for r in range(3)
    ]
    wt_gt = float(np.nanmean(controls))
    rows = []
    for inhibitor, start in EX21_BLOCK_START.items():
        for replicate in range(3):
            block = start + 10 * replicate
            for step, label in enumerate(EX21_LABELS[inhibitor], start=1):
                gt = traits.gt[block + step]
                rows.append(
                    {
                        "inhibitor": inhibitor,
                        "replicate": replicate + 1,
                        "step": step,
                        "label_g_per_l": label,
                        "well": block + step,
                        "generation_time_h": gt,
                        "fitness": wt_gt / gt,
                        "grew": bool(np.isfinite(gt)),
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["wild_type_generation_time_h"] = wt_gt
    return out


def plot_ex21(titration: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(75)), sharey=True
    )
    for ax, (inhibitor, color) in zip(
        axes.ravel(), zip(INHIBITORS, PLOT_PALETTE, strict=False), strict=True
    ):
        g = titration[titration["inhibitor"] == inhibitor]
        steps = sorted(g["step"].unique())
        for replicate, marker in zip((1, 2, 3), ("o", "s", "^"), strict=True):
            r = g[g["replicate"] == replicate].set_index("step").reindex(steps)
            ax.plot(
                steps,
                r["fitness"],
                marker=marker,
                markersize=2.5,
                linewidth=0.5,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=f"replicate {replicate}",
            )
            none = r[~r["grew"].fillna(False).astype(bool)]
            ax.scatter(
                none.index,
                np.zeros(len(none)),
                marker="x",
                s=10,
                color="black",
                linewidths=0.6,
            )
        ax.set_xticks(steps)
        ax.set_xticklabels(EX21_LABELS[inhibitor], rotation=60, fontsize=5)
        ax.set_title(f"{NAMES[inhibitor]} ({inhibitor}), label g/L", fontsize=6)
        ax.set_ylim(-0.05, 1.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="minor", length=0)
        ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
        box(ax)
    axes[0, 0].scatter(
        [], [], marker="x", s=10, color="black", label="no growth in 48 h"
    )
    axes[0, 0].legend(frameon=False, loc="lower left")
    for ax in axes[:, 0]:
        ax.set_ylabel("fitness (wild-type GT / GT)")
    fig.suptitle(
        "ex21 single-inhibitor titrations, highest concentration left; x = no growth in 48 h",
        fontsize=6,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.9, bottom=0.17, hspace=0.6, wspace=0.08
    )
    return save(fig, "ex21_titrations")


# ---- ex26: the furfural x acetic acid isobole ---------------------------- #
def ex26_isobole() -> pd.DataFrame:
    traits = Traits.read(
        "ex26",
        osp.join(
            ARCHIVE,
            "03_analysis_code",
            "inhibitor_tolerance",
            "MV_ex26_inhibitor_isobole_FF_AA_Traits.txt",
        ),
    )
    rows = []
    for plate, offset in ((1, 0), (2, 100)):
        wt_gt = traits.gt[offset + 1]
        for well in range(1, 101):
            # the notebook reshapes wells 1..100 as a 10 x 10 grid, transposed: the
            # column index runs with the slow well digit, the row with the fast one
            col, row = divmod(well - 1, 10)
            gt = traits.gt[offset + well]
            rows.append(
                {
                    "plate": plate,
                    "well": offset + well,
                    "furfural_g_per_l": EX26_FF[row],
                    "acetic_acid_g_per_l": EX26_AA[col],
                    "generation_time_h": gt,
                    "fitness": wt_gt / gt,
                    "grew": bool(np.isfinite(gt)),
                }
            )
    return pd.DataFrame(rows)


def plot_ex26(iso: pd.DataFrame) -> list[str]:
    mean = (
        iso.groupby(["furfural_g_per_l", "acetic_acid_g_per_l"])["fitness"]
        .mean()
        .unstack("acetic_acid_g_per_l")
        .reindex(index=EX26_FF, columns=EX26_AA)
    )
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(70)))
    cmap = matplotlib.colormaps["Oranges_r"].copy()
    cmap.set_bad(NO_GROWTH)
    im = ax.imshow(
        np.ma.masked_invalid(mean.to_numpy()), cmap=cmap, vmin=0, vmax=1, origin="lower"
    )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([f"{v:g}" for v in EX26_AA])
    ax.set_yticklabels([f"{v:g}" for v in EX26_FF])
    ax.set_xlabel("acetic acid (g/L)")
    ax.set_ylabel("furfural (g/L)")
    ax.set_title("ex26 isobole, mean of two plates; gray = no growth in 48 h")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("fitness (wild-type GT / GT)")
    cbar.outline.set_linewidth(0.5)
    box(ax)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.92, bottom=0.13)
    return save(fig, "ex26_isobole_furfural_acetic_acid")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    cond = ex23_conditions()
    cond.to_csv(osp.join(RESULTS, "ex23_conditions.csv"), index=False)
    summary = cond.groupby("order").agg(
        combinations=("name", "size"),
        grew=("grew", "sum"),
        fitness_mean=("fitness_mean", "mean"),
    )
    print("ex23 by number of inhibitors:\n" + summary.round(3).to_string())
    titration = ex21_titration()
    titration.to_csv(osp.join(RESULTS, "ex21_titration.csv"), index=False)
    print(
        f"ex21 wild-type generation time {titration.attrs['wild_type_generation_time_h']:.2f} h; "
        f"wells grown {int(titration['grew'].sum())} of {len(titration)}"
    )
    iso = ex26_isobole()
    iso.to_csv(osp.join(RESULTS, "ex26_isobole.csv"), index=False)
    print(f"ex26 wells grown {int(iso['grew'].sum())} of {len(iso)}")
    written = (
        plot_ex23_conditions(cond)
        + plot_ex23_pairs(cond)
        + plot_ex23_curves()
        + plot_ex21(titration)
        + plot_ex26(iso)
    )
    print("\n".join(written))


if __name__ == "__main__":
    main()
