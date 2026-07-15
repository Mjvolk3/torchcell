# experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_risk_palette.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_plot_risk_palette]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_risk_palette
"""Palette/SVG remake of the Hoepfner background-mutation figures (paper-ready).

Same content as `hoepfner_plot_risk.py` + `hoepfner_plot_table_s5_crossval.py`, re-rendered
against the repo's ordered `torchcell.utils.PLOT_PALETTE` (green-free), at Nature panel widths,
exported as TRUE-physical-size SVG via `savefig_true_size_svg` (imports into draw.io at exact
mm). Reads only the committed 017 result files. Output: `NN_*_palette.svg` in
`ASSET_IMAGES_DIR/017-hoepfner-background-mutations/`.

Palette semantics (documented; green-free, primaries-first): HIP = amber (focus collection);
HOP = steel blue (comparison/control); flagged/at-risk = brick (the red alert); whi2 = lilac;
clusters CL1-4 = amber/brick/lilac/wheat; neutral context = gray. Lightness is NOT used to
separate categories (reserved for validation-vs-test per repo standard).
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import torchcell
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
    "legend.title_fontsize": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "lines.linewidth": 0.8, "patch.linewidth": 0.4,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.01,
})

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "017-hoepfner-background-mutations", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "017-hoepfner-background-mutations")
os.makedirs(IMG_DIR, exist_ok=True)

AMBER, BRICK, LILAC, WHEAT, STEEL, GRAY = PLOT_PALETTE[:6]
DENIM = PLOT_PALETTE[10]
BLACK = "#000000"
HIP_C, HOP_C, FLAG_C, WHI2_C, NEUTRAL = AMBER, STEEL, BRICK, LILAC, GRAY
# Clusters need 4 clearly-distinct HUES on one overlaid panel, so CL4 uses the blue slot
# rather than wheat (amber's near-twin) — a documented distinguishability deviation from
# strict "first-N" ordering; still green-free and hue- (not lightness-) separated.
CLUSTER_C = {"CL1": AMBER, "CL2": BRICK, "CL3": LILAC, "CL4": STEEL}
CLUSTER_LABEL = {
    "CL1": "Cluster 1 · chr XI aneuploidy", "CL2": "Cluster 2 · chr XI aneuploidy",
    "CL3": "Cluster 3 · WHI2 nonsense", "CL4": "Cluster 4 · chr V amplification",
}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI"]
MS = 2.0  # base marker size (pt^2 handled per-call)


def _save(fig, name: str) -> None:
    fig.tight_layout(pad=0.4)
    path = osp.join(IMG_DIR, f"{name}_palette.svg")
    savefig_true_size_svg(fig, path)
    plt.close(fig)
    print(f"  saved {path}")


def load_all():
    summary = json.load(open(osp.join(RESULTS_DIR, "summary_stats.json")))
    hip = pd.read_csv(osp.join(RESULTS_DIR, "hip_strain_metrics.csv"))
    hop = pd.read_csv(osp.join(RESULTS_DIR, "hop_strain_metrics.csv"))
    hip["orf"] = hip["orf"].str.upper()
    hop["orf"] = hop["orf"].str.upper()
    base_hip = summary["HIP"]["baseline_neg_rate"]["neg_rate_4"]
    base_hop = summary["HOP"]["baseline_neg_rate"]["neg_rate_4"]
    hip["enr"] = hip["frac_hyper_4"] / base_hip
    hop["enr"] = hop["frac_hyper_4"] / base_hop
    flagged = set(pd.read_csv(
        osp.join(RESULTS_DIR, "flagged_hip_strains.csv"))["orf"].str.upper())
    hip["flagged"] = hip["orf"].isin(flagged)
    s5 = pd.read_csv(osp.join(RESULTS_DIR, "table_s5_affected_strains.csv"))
    s5["orf"] = s5["orf"].str.upper()
    samples = None
    npz = osp.join(RESULTS_DIR, "score_samples.npz")
    if osp.exists(npz):
        samples = np.load(npz)
    return summary, hip, hop, flagged, s5, samples, base_hip, base_hop


def genome_axes(summary, ax):
    offsets = {int(k): v for k, v in summary["chrom_offsets"].items()}
    extent = {int(k): v for k, v in summary["chrom_extent"].items()}
    bounds = [offsets[i] for i in range(1, 17)] + [offsets[16] + extent[16]]
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(16)]
    for b in bounds:
        ax.axvline(b, color=BLACK, lw=0.2, alpha=0.22)
    ax.set_xticks(centers)
    ax.set_xticklabels(ROMAN)
    ax.set_xlim(bounds[0], bounds[-1])
    return offsets


# ------------------------------------------------------------------- figures
def fig01_scores(samples):
    if samples is None:
        print("  [skip 01] score_samples.npz missing")
        return
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(56)))
    bins = np.linspace(-12, 8, 150)
    ax.hist(samples["HIP"], bins=bins, color=HIP_C, alpha=0.7, log=True,
            label=f"HIP heterozygous ({samples['HIP'].size/1e6:.1f}M cells)")
    ax.hist(samples["HOP"], bins=bins, color=HOP_C, alpha=0.6, log=True,
            label=f"HOP homozygous ({samples['HOP'].size/1e6:.1f}M cells)")
    ax.axvline(-4.0, color=FLAG_C, ls="--", lw=1.0, label="hypersensitive cut (−4)")
    ax.set_xlabel("adjusted MADL sensitivity score  (negative = hypersensitive)")
    ax.set_ylabel("cell count (log)")
    ax.legend(loc="upper left", framealpha=0.9, handlelength=1.3)
    _save(fig, "01_score_distribution_hip_vs_hop")


def fig02_ecdf(hip, hop):
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(56)))
    for df, c, lab in ((hop, HOP_C, "HOP"), (hip, HIP_C, "HIP")):
        x = np.sort(df["enr"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=c, lw=1.1, label=f"{lab} ({len(x)} strains)")
    ax.axvline(4.0, color=FLAG_C, ls="--", lw=1.0, label="flag (4×)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("per-strain promiscuity enrichment (× baseline)")
    ax.set_ylabel("cumulative fraction of strains")
    ax.legend(loc="lower right", framealpha=0.9, handlelength=1.3)
    _save(fig, "02_promiscuity_ecdf")


def fig03_genome(summary, hip, hop):
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(64)))
    genome_axes(summary, ax)
    hg = hop.dropna(subset=["cum_pos"])
    ax.scatter(hg["cum_pos"], hg["enr"], s=1.2, color=NEUTRAL, alpha=0.16,
               label="HOP (control)", rasterized=True, linewidths=0)
    un = hip[~hip["flagged"]].dropna(subset=["cum_pos"])
    for ci, sub in un.groupby("chrom_idx"):
        ax.scatter(sub["cum_pos"], sub["enr"], s=2.2,
                   color=STEEL if int(ci) % 2 else DENIM, alpha=0.6,
                   rasterized=True, linewidths=0)
    fl = hip[hip["flagged"]].dropna(subset=["cum_pos"])
    ax.scatter(fl["cum_pos"], fl["enr"], s=5, color=FLAG_C, alpha=0.95,
               edgecolor=BLACK, linewidth=0.15,
               label=f"flagged batch artifact (n={len(fl)})", rasterized=True)
    ax.axhline(4.0, color=FLAG_C, ls="--", lw=0.7, alpha=0.7)
    for orf, lab in (("YLR293C", "GSP1"), ("YIL046W", "MET30")):
        r = hip[hip["orf"] == orf]
        if len(r):
            ax.annotate(lab, (r["cum_pos"].iloc[0], r["enr"].iloc[0]), fontsize=5,
                        ha="center", va="bottom")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("chromosomal position of deleted ORF (chromosome = construction-lab region)")
    ax.set_ylabel("HIP promiscuity enrichment (× baseline)")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=2, handlelength=1.2)
    _save(fig, "03_genome_promiscuity_map_hip")


def fig04_whi2(hip):
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(56)))
    v = hip["whi2_corr"].dropna()
    ax.hist(v, bins=55, color=WHI2_C, alpha=0.85)
    n_c3 = int((hip["whi2_corr"] >= 0.5).sum())
    ax.axvline(0.5, color=FLAG_C, ls="--", lw=1.0, label=f"cluster-3 cut (≥0.5, n={n_c3})")
    ax.set_xlabel("Pearson r of strain profile vs whi2/YOR043w (HIP)")
    ax.set_ylabel("HIP strain count")
    ax.legend(loc="upper right", framealpha=0.9, handlelength=1.3)
    _save(fig, "04_whi2_cluster3_signature")


def fig05_specificity(hip, hop):
    m = hip.merge(hop[["orf", "enr"]], on="orf", suffixes=("_hip", "_hop"))
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(82)))
    nf = m[~m["flagged"]]
    fl = m[m["flagged"]]
    ax.scatter(nf["enr_hop"], nf["enr_hip"], s=1.6, color=NEUTRAL, alpha=0.35,
               rasterized=True, linewidths=0, label="HIP strain (in both)")
    ax.scatter(fl["enr_hop"], fl["enr_hip"], s=6, color=FLAG_C, alpha=0.95,
               edgecolor=BLACK, linewidth=0.15, label=f"flagged (n={len(fl)})")
    ax.plot([0, 60], [0, 60], color=BLACK, ls=":", lw=0.7, label="HIP = HOP")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("HOP enrichment (× baseline)")
    ax.set_ylabel("HIP enrichment (× baseline)")
    ax.set_title("flagged = HIP-specific (construction artifact)")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=2, handlelength=1.2)
    _save(fig, "05_hip_vs_hop_specificity")


def fig06_tiers(hip, flagged, s5):
    total = 29_996_238
    whi2 = hip[hip["whi2_corr"] >= 0.5]
    fl = hip[hip["flagged"]]
    pos = s5[s5["is_positional"]]
    tiers = [
        ("whi2 cluster", len(whi2), int(whi2["n_exp"].sum()), AMBER),
        ("empirical\nruns (≥4×)", len(fl), int(fl["n_exp"].sum()), BRICK),
        ("Table S5\npositional", len(pos), int(pos["n_exp"].sum()), LILAC),
        ("Table S5\nall flagged", len(s5), int(s5["n_exp"].sum()), WHEAT),
    ]
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(62)))
    xs = range(len(tiers))
    pct = [t[2] / total * 100 for t in tiers]
    ax.bar(xs, pct, color=[t[3] for t in tiers], edgecolor=BLACK, linewidth=0.4, width=0.72)
    for i, t in enumerate(tiers):
        ax.text(i, pct[i], f"{pct[i]:.2f}%\n{t[1]} str", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([t[0] for t in tiers])
    ax.set_ylabel("% of all 29,996,238 records")
    ax.set_ylim(0, max(pct) * 1.32)
    ax.set_title("Records at risk (HOP not implicated)")
    _save(fig, "06_risk_tiers_pct_records")


def fig07_composition(summary, s5):
    hip_r = summary["HIP"]["n_records"]
    hop_r = summary["HOP"]["n_records"]
    flag_r = int(s5["n_exp"].sum())
    hip_clean = hip_r - flag_r
    tot = hip_r + hop_r
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(66)))
    ax.pie([flag_r, hip_clean, hop_r], colors=[FLAG_C, HIP_C, HOP_C],
           labels=[f"HIP flagged\n{flag_r/tot*100:.1f}%",
                   f"HIP other\n{hip_clean/tot*100:.1f}%",
                   f"HOP (n.i.)\n{hop_r/tot*100:.1f}%"],
           wedgeprops=dict(edgecolor="white", linewidth=0.8), startangle=90,
           textprops=dict(fontsize=5.5))
    ax.set_title("Record composition (30.0M)")
    _save(fig, "07_record_composition")


def fig08_recovery(summary, hip, s5):
    aff = s5.merge(hip[["orf", "cum_pos", "enr"]], on="orf", how="left", suffixes=("", "_h"))
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(66)))
    genome_axes(summary, ax)
    hg = hip.dropna(subset=["cum_pos"])
    ax.scatter(hg["cum_pos"], hg["enr"], s=1.2, color=NEUTRAL, alpha=0.16,
               label="HIP strain (all)", rasterized=True, linewidths=0)
    for cl in ("CL1", "CL2", "CL3", "CL4"):
        sub = aff[aff["base_cluster"] == cl].dropna(subset=["cum_pos"])
        ax.scatter(sub["cum_pos"], sub["enr"], s=7, color=CLUSTER_C[cl],
                   edgecolor=BLACK, linewidth=0.2, label=f"{CLUSTER_LABEL[cl]} (n={len(sub)})")
    caught = aff[aff["empirically_flagged"]].dropna(subset=["cum_pos"])
    ax.scatter(caught["cum_pos"], caught["enr"], s=26, facecolors="none",
               edgecolor=BLACK, linewidth=0.6, label=f"recovered (n={len(caught)})")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("chromosomal position of the deleted ORF")
    ax.set_ylabel("HIP promiscuity enrichment (× baseline)")
    ax.legend(loc="upper center", ncol=3, framealpha=0.9, markerscale=1.4,
              handlelength=1.1, columnspacing=1.0)
    _save(fig, "08_table_s5_cluster_recovery")


def main():
    summary, hip, hop, flagged, s5, samples, _, _ = load_all()
    print("rendering palette SVG figures...")
    fig01_scores(samples)
    fig02_ecdf(hip, hop)
    fig03_genome(summary, hip, hop)
    fig04_whi2(hip)
    fig05_specificity(hip, hop)
    fig06_tiers(hip, flagged, s5)
    fig07_composition(summary, s5)
    fig08_recovery(summary, hip, s5)
    print("done.")


if __name__ == "__main__":
    main()
