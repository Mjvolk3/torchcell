# experiments/010-kuzmin-tmi/scripts/constructed_10_smf_figures.py
# [[experiments.010-kuzmin-tmi.scripts.constructed_10_smf_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/constructed_10_smf_figures
"""SMF figure set for the 10 properly-constructed genes.

Two figures characterizing single-mutant fitness of the 10 genes the wet-lab plate
actually built (inference_3 panel-12 minus YIL174W and LCL2/YLR104W):
  1. forest plot — between-source fitness +/- SD (Costanzo2016 / Kuzmin2018 / Kuzmin2020)
  2. Gaussian ridgeline — Costanzo Gaussians on a shared axis, ordered by mean, colored by sigma

Source: results/inference_3/singles_table_panel12_k200_queried.csv (filtered to the 10).
Output: notes/assets/images/010-kuzmin-tmi/constructed_10_smf_{forest,ridgeline}.{png,svg}

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/constructed_10_smf_figures.py
"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import PLOT_PALETTE, PANEL_WIDTHS_MM, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
SRC = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_3",
               "singles_table_panel12_k200_queried.csv")

TEN = ["YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
       "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W"]
COMMON = {"YBR203W": "COS111", "YDR057W": "YOS9", "YER079W": "", "YGL087C": "MMS2",
          "YJR060W": "CBF1", "YKL033W-A": "", "YLL012W": "YEH1", "YLR312C-B": "",
          "YPL046C": "ELC1", "YPL081W": "RPS9A"}

# source -> palette color (Costanzo = gray primary, Kuzmin2018 = orange, Kuzmin2020 = blue)
SOURCES = [("SmfCostanzo2016", PLOT_PALETTE[5], "Costanzo2016"),
           ("SmfKuzmin2018", PLOT_PALETTE[0], "Kuzmin2018"),
           ("SmfKuzmin2020", PLOT_PALETTE[4], "Kuzmin2020")]

plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                     "svg.fonttype": "none", "axes.linewidth": 0.5})


def label(orf: str) -> str:
    return f"{orf} / {COMMON[orf]}" if COMMON[orf] else orf


def box(ax) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.5)


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df = df[df["gene"].isin(TEN)].copy()
    return df.sort_values("SmfCostanzo2016_fitness").reset_index(drop=True)


def forest(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(75)))
    ys = np.arange(len(df))
    for k, (src, color, name) in enumerate(SOURCES):
        off = (k - 1) * 0.22
        f = df[f"{src}_fitness"].to_numpy(dtype=float)
        s = df[f"{src}_std"].to_numpy(dtype=float)
        ax.errorbar(f, ys + off, xerr=np.nan_to_num(s), fmt="o", ms=3, lw=0,
                    elinewidth=0.6, capsize=1.5, color=color, label=name, zorder=3)
    ax.axvline(1.0, color="0.4", ls=":", lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label(g) for g in df["gene"]])
    ax.set_xlabel("Single-mutant fitness")
    ax.set_title("SMF by source — 10 properly-constructed genes")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis="x", lw=0.3, color="0.88", zorder=0)
    ax.legend(frameon=True, fontsize=6, loc="lower right")
    box(ax)
    fig.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "constructed_10_smf_forest.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "constructed_10_smf_forest.svg"))
    plt.close(fig)


def ridgeline(df: pd.DataFrame) -> None:
    d = df.rename(columns={"SmfCostanzo2016_fitness": "mu", "SmfCostanzo2016_std": "sd"})
    x = np.linspace(0.2, 1.4, 800)
    overlap, cmap = 1.8, plt.cm.viridis
    smin, smax = d["sd"].min(), d["sd"].max()
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(75)))
    for i, r in d.iterrows():
        y = np.exp(-0.5 * ((x - r.mu) / r.sd) ** 2) / (r.sd * np.sqrt(2 * np.pi))
        y = y / y.max() * overlap
        base = i
        color = cmap(0.15 + 0.7 * (r.sd - smin) / (smax - smin))
        ax.fill_between(x, base, base + y, color=color, alpha=0.85, zorder=len(d) - i)
        ax.plot(x, base + y, color="black", lw=0.6, zorder=len(d) - i)
        ax.text(0.21, base + 0.05, label(r.gene), va="bottom", ha="left", fontsize=6, zorder=100)
        ax.text(1.39, base + 0.05, rf"${r.mu:.3f}\pm{r.sd:.3f}$", va="bottom", ha="right",
                fontsize=5.5, color="0.25", zorder=100)
    ax.axvline(1.0, color="0.4", ls=":", lw=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_xlabel("Single-mutant fitness (Costanzo2016)")
    ax.set_xlim(0.2, 1.4)
    ax.set_title(rf"SMF Gaussians — 10 genes (mean $\sigma$={d['sd'].mean():.3f})")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "constructed_10_smf_ridgeline.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "constructed_10_smf_ridgeline.svg"))
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    forest(df)
    ridgeline(df)
    print(f"10 genes; Costanzo fitness range "
          f"{df['SmfCostanzo2016_fitness'].min():.3f}-{df['SmfCostanzo2016_fitness'].max():.3f}, "
          f"mean SD {df['SmfCostanzo2016_std'].mean():.3f}")
    print(f"saved forest + ridgeline to {OUT_DIR}")


if __name__ == "__main__":
    main()
