# experiments/010-kuzmin-tmi/scripts/constructed_10_dmf_reference.py
# [[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/constructed_10_dmf_reference
"""Double-mutant fitness (DMF) +/- SD reference for the doubles among the 10 genes.

Pulls the published DMF +/- SD (and digenic interaction epsilon + p-value) for every
pair among the 10 properly-constructed genes, so that when the wet-lab constructs
double mutants their measured fitness can be compared to Costanzo2016 / Kuzmin.
Also flags the 8 set-cover doubles from
optimized_doubles_setcover_constructed_10.py.

Source: results/inference_3/doubles_table_panel12_k200_queried.csv (filtered to the 10).
Output: results/constructed_10_dmf_costanzo_kuzmin.csv
        notes/assets/images/010-kuzmin-tmi/constructed_10_dmf_forest.{png,svg}

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/constructed_10_dmf_reference.py
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
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
SRC = osp.join(RESULTS_DIR, "inference_3", "doubles_table_panel12_k200_queried.csv")

TEN = {"YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
       "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W"}

# the 8 set-cover doubles (gene1<gene2) to construct
OPTIMIZED = {
    ("YBR203W", "YPL046C"), ("YDR057W", "YER079W"), ("YDR057W", "YLL012W"),
    ("YER079W", "YLR312C-B"), ("YLL012W", "YPL046C"), ("YJR060W", "YPL046C"),
    ("YDR057W", "YPL081W"), ("YLR312C-B", "YPL081W"),
}
COLOR_OPT = PLOT_PALETTE[1]   # red — the doubles to construct
COLOR_OTHER = PLOT_PALETTE[5]  # gray — other measured doubles

plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                     "svg.fonttype": "none", "axes.linewidth": 0.5})


def build_reference() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df = df[df.apply(lambda r: {r.gene1, r.gene2}.issubset(TEN), axis=1)].copy()
    # order genes within each pair
    g = df.apply(lambda r: tuple(sorted((r.gene1, r.gene2))), axis=1)
    df["gene1"], df["gene2"] = [x[0] for x in g], [x[1] for x in g]
    df["is_optimized_double"] = df.apply(
        lambda r: (r.gene1, r.gene2) in OPTIMIZED, axis=1
    )
    keep = ["gene1", "gene2", "is_optimized_double",
            "DmfCostanzo2016_fitness", "DmfCostanzo2016_std",
            "DmiCostanzo2016_gene_interaction", "DmiCostanzo2016_gene_interaction_p_value",
            "DmfKuzmin2018_fitness", "DmfKuzmin2018_std",
            "DmfKuzmin2020_fitness", "DmfKuzmin2020_std",
            "DmfCostanzo2016_strain_id"]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.sort_values("DmfCostanzo2016_fitness").reset_index(drop=True)


def forest(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["DmfCostanzo2016_fitness"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(120)))
    ys = np.arange(len(d))
    for i, r in d.iterrows():
        opt = r["is_optimized_double"]
        c = COLOR_OPT if opt else COLOR_OTHER
        ax.errorbar(r["DmfCostanzo2016_fitness"], i,
                    xerr=0.0 if pd.isna(r["DmfCostanzo2016_std"]) else r["DmfCostanzo2016_std"],
                    fmt="o", ms=3.2 if opt else 2.6, lw=0, elinewidth=0.6, capsize=1.4,
                    color=c, zorder=3)
    ax.axvline(1.0, color="0.4", ls=":", lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.gene1}+{r.gene2}" + (" *" if r.is_optimized_double else "")
                        for _, r in d.iterrows()], fontsize=5)
    ax.set_xlabel("Double-mutant fitness (Costanzo2016)")
    ax.set_title("DMF ± SD, doubles of the 10 genes\n(red * = set-cover double to construct)")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis="x", lw=0.3, color="0.88", zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "constructed_10_dmf_forest.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "constructed_10_dmf_forest.svg"))
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_reference()
    out = osp.join(RESULTS_DIR, "constructed_10_dmf_costanzo_kuzmin.csv")
    df.to_csv(out, index=False)
    forest(df)
    n_meas = df["DmfCostanzo2016_fitness"].notna().sum()
    n_opt = int(df["is_optimized_double"].sum())
    print(f"within-10 doubles: {len(df)} (C(10,2)=45); Costanzo DMF measured {n_meas}/{len(df)}")
    print(f"  optimized set-cover doubles present: {n_opt}/8")
    print(f"Saved: {out}")
    print(f"       {osp.join(OUT_DIR, 'constructed_10_dmf_forest.svg')}")


if __name__ == "__main__":
    main()
