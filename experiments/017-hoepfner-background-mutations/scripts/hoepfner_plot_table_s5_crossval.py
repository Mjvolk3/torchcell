# experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_table_s5_crossval.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_plot_table_s5_crossval]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_table_s5_crossval
"""Capstone figure: authoritative Table_S5 clusters on the HIP promiscuity genome map.

Overlays the exact Table_S5 cluster assignments (coloured by cluster/mutation) on the
per-strain HIP enrichment, marking which strains the empirical detector recovered. Shows the
four batch clusters at their genomic positions and that the detector's misses are still
elevated (it is conservative, not wrong).
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

import torchcell
from torchcell.timestamp import timestamp

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "017-hoepfner-background-mutations", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "017-hoepfner-background-mutations")
BLACK, GRAY = "#000000", "#6D666F"
CLUSTER_COLOR = {"CL1": "#34699D", "CL2": "#52B2A8", "CL3": "#D86E2F", "CL4": "#775A9F"}
CLUSTER_LABEL = {
    "CL1": "Cluster 1 · chr XI aneuploidy", "CL2": "Cluster 2 · chr XI aneuploidy",
    "CL3": "Cluster 3 · WHI2 nonsense", "CL4": "Cluster 4 · chr V amplification",
}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI"]


def main() -> None:
    summary = json.load(open(osp.join(RESULTS_DIR, "summary_stats.json")))
    offsets = {int(k): v for k, v in summary["chrom_offsets"].items()}
    extent = {int(k): v for k, v in summary["chrom_extent"].items()}
    base_hip = summary["HIP"]["baseline_neg_rate"]["neg_rate_4"]

    hip = pd.read_csv(osp.join(RESULTS_DIR, "hip_strain_metrics.csv"))
    hip["orf"] = hip["orf"].str.upper()
    hip["enr"] = hip["frac_hyper_4"] / base_hip
    aff = pd.read_csv(osp.join(RESULTS_DIR, "table_s5_affected_strains.csv"))
    aff["orf"] = aff["orf"].str.upper()
    aff = aff.merge(hip[["orf", "cum_pos", "enr"]], on="orf", how="left",
                    suffixes=("", "_h"))

    bounds = [offsets[i] for i in range(1, 17)] + [offsets[16] + extent[16]]
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(16)]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    hg = hip.dropna(subset=["cum_pos"])
    ax.scatter(hg["cum_pos"], hg["enr"], s=6, color=GRAY, alpha=0.18,
               label="HIP strain (all)", rasterized=True)
    for cl in ("CL1", "CL2", "CL3", "CL4"):
        sub = aff[aff["base_cluster"] == cl].dropna(subset=["cum_pos"])
        ax.scatter(sub["cum_pos"], sub["enr"], s=34, color=CLUSTER_COLOR[cl],
                   edgecolor=BLACK, linewidth=0.3, label=f"{CLUSTER_LABEL[cl]} (n={len(sub)})")
    caught = aff[aff["empirically_flagged"]].dropna(subset=["cum_pos"])
    ax.scatter(caught["cum_pos"], caught["enr"], s=95, facecolors="none",
               edgecolor=BLACK, linewidth=1.1,
               label=f"recovered by empirical detector (n={len(caught)})")
    for b in bounds:
        ax.axvline(b, color=BLACK, lw=0.3, alpha=0.25)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(centers)
    ax.set_xticklabels(ROMAN, fontsize=10)
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_xlabel("chromosomal position of the deleted ORF")
    ax.set_ylabel("HIP promiscuity enrichment (x baseline)")
    ax.set_title("Authoritative Table_S5 background-mutation clusters recovered on the "
                 "HIP genome map (188 strains, 4 batches)")
    ax.legend(loc="upper center", ncol=2, framealpha=0.9, fontsize=9)
    path = osp.join(IMG_DIR, f"08_table_s5_cluster_recovery_{timestamp()}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
