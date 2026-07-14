# experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_risk.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_plot_risk]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_plot_risk
"""Phase 2 (detector + tiering + plots) for the Hoepfner background-mutation risk audit.

Reads the Phase-1 caches (per-strain metrics + score sample + summary) and:
  1. builds a per-assay-normalised promiscuity ENRICHMENT (frac hypersensitive / baseline),
  2. runs the paper's discriminator empirically -- flags RUNS of chromosomally ADJACENT HIP
     strains with broad hypersensitivity (batch artifact), separating them from isolated
     genuinely-pleiotropic strains (e.g. essential-gene dosage sensitivity),
  3. corroborates with the whi2/YOR043w correlation (cluster 3) and the HOP control
     (the finding is heterozygous-collection specific),
  4. tiers "% of the built dataset at risk" from the narrowest (characterised) to the
     widest (conservative HIP envelope) definition,
  5. writes flagged_hip_strains.csv + risk_tiers.json and renders all figures in the
     torchcell palette.

Every number traces to our own sha256-pinned raw matrices (Phase 1 reconciled to the exact
29,996,238-record LMDB), so the risk estimate maps 1:1 onto the built dataset.
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
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
os.makedirs(IMG_DIR, exist_ok=True)

# torchcell palette (torchcell.mplstyle)
BLACK, ORANGE, BLUEGRAY, GREEN = "#000000", "#D86E2F", "#7191A9", "#6B8D3A"
RED, BLUE, PURPLE, TEAL = "#B73C39", "#34699D", "#775A9F", "#52B2A8"
GRAY = "#6D666F"

# Detector parameters (documented, tunable; results reported across the enrichment sweep)
PRIMARY_THRESH = -4.0       # "strong hypersensitive" cell (frac_hyper_4)
ROLL_WINDOW = 7             # adjacent-strain window for the local-run detector
ENR_FLAG = 4.0             # primary local-enrichment flag (x over per-assay baseline)
ENR_SWEEP = (3.0, 4.0, 5.0)
WHI2_CORR_MIN = 0.5
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI"]
TS = timestamp()


def _save(fig, title: str) -> None:
    path = osp.join(IMG_DIR, f"{title}_{TS}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def load_all():
    summary = json.load(open(osp.join(RESULTS_DIR, "summary_stats.json")))
    hip = pd.read_csv(osp.join(RESULTS_DIR, "hip_strain_metrics.csv"))
    hop = pd.read_csv(osp.join(RESULTS_DIR, "hop_strain_metrics.csv"))
    samples = np.load(osp.join(RESULTS_DIR, "score_samples.npz"))
    base_hip = summary["HIP"]["baseline_neg_rate"]["neg_rate_4"]
    base_hop = summary["HOP"]["baseline_neg_rate"]["neg_rate_4"]
    hip["enr"] = hip["frac_hyper_4"] / base_hip
    hop["enr"] = hop["frac_hyper_4"] / base_hop
    return summary, hip, hop, samples, base_hip, base_hop


def flag_runs(hip: pd.DataFrame, enr_flag: float) -> pd.DataFrame:
    """Flag HIP strains inside a high-enrichment neighbourhood, rolling WITHIN a chromosome.

    Rolling median over `ROLL_WINDOW` adjacent (genome-ordered) strains -- a batch artifact
    is a contiguous stretch of neighbouring deletions all broadly hypersensitive, so the
    local median (not a single strain) is what must be elevated.
    """
    h = hip.dropna(subset=["cum_pos"]).sort_values("cum_pos").copy()
    h["roll_enr"] = (
        h.groupby("chrom_idx")["enr"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, center=True, min_periods=4).median())
    )
    h["flagged"] = h["roll_enr"] >= enr_flag
    return h


def suspect_runs(h: pd.DataFrame) -> list[dict]:
    """Summarise contiguous flagged stretches (>=3 adjacent flagged strains) per chromosome."""
    runs = []
    for ci, sub in h.sort_values("cum_pos").groupby("chrom_idx"):
        sub = sub.reset_index(drop=True)
        i = 0
        flags = sub["flagged"].to_numpy()
        while i < len(sub):
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(sub) and flags[j + 1]:
                j += 1
            if j - i + 1 >= 3:
                blk = sub.iloc[i : j + 1]
                runs.append(dict(
                    chrom=ROMAN[int(ci) - 1],
                    pos_start=int(blk["pos"].min()),
                    pos_end=int(blk["pos"].max()),
                    n_strains=int(len(blk)),
                    whi2_frac=float((blk["whi2_corr"] >= WHI2_CORR_MIN).mean()),
                    genes=list(blk["gene"]),
                    orfs=list(blk["orf"]),
                ))
            i = j + 1
    return sorted(runs, key=lambda r: -r["n_strains"])


# ----------------------------------------------------------------------------- plots
def plot_score_distributions(samples, base_hip, base_hop):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-12, 8, 160)
    ax.hist(samples["HIP"], bins=bins, color=BLUEGRAY, alpha=0.65,
            label=f"HIP (heterozygous)  n={samples['HIP'].size:,} cells", log=True)
    ax.hist(samples["HOP"], bins=bins, color=GREEN, alpha=0.55,
            label=f"HOP (homozygous)  n={samples['HOP'].size:,} cells", log=True)
    ax.axvline(PRIMARY_THRESH, color=RED, ls="--", lw=1.5,
               label=f"hypersensitive cut ({PRIMARY_THRESH:g})")
    ax.set_xlabel("adjusted MADL sensitivity score  (negative = hypersensitive)")
    ax.set_ylabel("cell count (log)")
    ax.set_title("Hoepfner 2014 sensitivity-score distribution: HIP vs HOP")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "01_score_distribution_hip_vs_hop")


def plot_promiscuity_ecdf(hip, hop):
    fig, ax = plt.subplots(figsize=(10, 6))
    for df, c, lab in ((hop, GREEN, "HOP"), (hip, BLUEGRAY, "HIP")):
        x = np.sort(df["enr"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=c, lw=2, label=f"{lab}  ({len(x)} strains)")
    ax.axvline(ENR_FLAG, color=RED, ls="--", lw=1.5, label=f"flag threshold ({ENR_FLAG:g}x)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("per-strain promiscuity enrichment  (frac hypersensitive / assay baseline)")
    ax.set_ylabel("cumulative fraction of strains")
    ax.set_title("Per-strain promiscuous hypersensitivity (baseline-normalised)")
    ax.legend(loc="lower right", framealpha=0.9)
    _save(fig, "02_promiscuity_ecdf")


def plot_genome_map(summary, h, hop):
    """Manhattan-style genome map of HIP promiscuity; flagged runs vs isolated pleiotropy."""
    offsets = {int(k): v for k, v in summary["chrom_offsets"].items()}
    extent = {int(k): v for k, v in summary["chrom_extent"].items()}
    bounds = [offsets[i] for i in range(1, 17)] + [offsets[16] + extent[16]]
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(16)]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    hop_g = hop.dropna(subset=["cum_pos"])
    ax.scatter(hop_g["cum_pos"], hop_g["enr"], s=5, color=GRAY, alpha=0.20,
               label="HOP strain (control)", rasterized=True)
    unflagged = h[~h["flagged"]]
    flagged = h[h["flagged"]]
    # alternate two blues by chromosome for the unflagged HIP background
    for ci, sub in unflagged.groupby("chrom_idx"):
        ax.scatter(sub["cum_pos"], sub["enr"], s=9,
                   color=BLUE if int(ci) % 2 else BLUEGRAY, alpha=0.55, rasterized=True)
    ax.scatter(flagged["cum_pos"], flagged["enr"], s=22, color=RED, alpha=0.9,
               edgecolor=BLACK, linewidth=0.2,
               label=f"flagged batch-artifact strain (n={len(flagged)})")
    ax.axhline(ENR_FLAG, color=RED, ls="--", lw=1.0, alpha=0.7)
    for b in bounds:
        ax.axvline(b, color=BLACK, lw=0.3, alpha=0.25)
    # annotate a couple of isolated genuine-pleiotropy essentials for contrast
    for orf, lab in (("YLR293C", "GSP1"), ("YIL046W", "MET30")):
        r = h[h["orf"] == orf]
        if len(r):
            ax.annotate(lab, (r["cum_pos"].iloc[0], r["enr"].iloc[0]),
                        fontsize=9, color=BLACK, ha="center", va="bottom")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(centers)
    ax.set_xticklabels(ROMAN, fontsize=10)
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_xlabel("chromosomal position of the deleted ORF (chromosome = construction-lab region)")
    ax.set_ylabel("HIP promiscuity enrichment (x baseline)")
    ax.set_title("HIP background-mutation map: RUNS of adjacent hypersensitive strains = "
                 "batch artifacts; isolated peaks = genuine dosage pleiotropy")
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, "03_genome_promiscuity_map_hip")


def plot_whi2(hip, hop):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    v = hip["whi2_corr"].dropna()
    ax.hist(v, bins=60, color=BLUEGRAY, alpha=0.8)
    ax.axvline(WHI2_CORR_MIN, color=RED, ls="--", lw=1.5,
               label=f"cluster-3 cut ({WHI2_CORR_MIN:g})")
    n_c3 = int((hip["whi2_corr"] >= WHI2_CORR_MIN).sum())
    ax.set_xlabel("Pearson r of strain profile vs whi2/YOR043w (HIP)")
    ax.set_ylabel("HIP strain count")
    ax.set_title(f"WHI2 co-behaviour: {n_c3} strains share the whi2 background signature")
    ax.legend(framealpha=0.9)

    ax = axes[1]
    c3 = hip[hip["whi2_corr"] >= WHI2_CORR_MIN].sort_values("cum_pos")
    order = c3["chrom"].value_counts()
    ax.bar([str(k) for k in order.index], order.values, color=ORANGE, alpha=0.85)
    ax.set_xlabel("chromosome of deleted ORF")
    ax.set_ylabel("# whi2-linked (cluster-3) strains")
    ax.set_title("whi2-linked strains cluster on adjacent chromosomal batches (XII, IX)")
    _save(fig, "04_whi2_cluster3_signature")


def plot_hip_vs_hop_specificity(hip, hop, h):
    """The control: for genes in BOTH collections, HIP vs HOP enrichment; artifacts are
    HIP-specific (off-diagonal, high HIP / low HOP)."""
    merged = hip.merge(hop[["orf", "enr"]], on="orf", suffixes=("_hip", "_hop"))
    flagged_orfs = set(h[h["flagged"]]["orf"])
    merged["flagged"] = merged["orf"].isin(flagged_orfs)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    nf = merged[~merged["flagged"]]
    fl = merged[merged["flagged"]]
    ax.scatter(nf["enr_hop"], nf["enr_hip"], s=8, color=BLUEGRAY, alpha=0.4,
               label="HIP strain (in both collections)", rasterized=True)
    ax.scatter(fl["enr_hop"], fl["enr_hip"], s=28, color=RED, alpha=0.9,
               edgecolor=BLACK, linewidth=0.2, label=f"flagged (n={len(fl)})")
    lim = 60
    ax.plot([0, lim], [0, lim], color=BLACK, ls=":", lw=1, label="HIP = HOP (gene biology)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("HOP enrichment (x baseline)")
    ax.set_ylabel("HIP enrichment (x baseline)")
    ax.set_title("Flagged strains are HETEROZYGOUS-collection specific\n"
                 "(high HIP / low HOP) -- construction artifact, not gene biology")
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, "05_hip_vs_hop_specificity")


def plot_risk_tiers(tiers, summary):
    labels = [t["short"] for t in tiers]
    pct = [t["pct_all_records"] for t in tiers]
    colors = [TEAL, GREEN, ORANGE, PURPLE, RED]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, pct, color=colors[: len(labels)], alpha=0.9, edgecolor=BLACK,
                  linewidth=0.4)
    for b, t in zip(bars, tiers):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{t['pct_all_records']:.2f}%\n{t['n_records']:,} rec\n{t['n_strains']} strains",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of all 29,996,238 records")
    ax.set_ylim(0, max(pct) * 1.25)
    ax.set_title("Records at risk of an unrepresented HIP background mutation, by definition")
    ax.tick_params(axis="x", labelsize=10)
    _save(fig, "06_risk_tiers_pct_records")


def plot_composition(summary, flagged_records):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    hip_r = summary["HIP"]["n_records"]
    hop_r = summary["HOP"]["n_records"]
    hip_clean = hip_r - flagged_records
    sizes = [flagged_records, hip_clean, hop_r]
    labels = [
        f"HIP flagged artifact\n{flagged_records:,} ({flagged_records/(hip_r+hop_r)*100:.1f}%)",
        f"HIP other\n{hip_clean:,} ({hip_clean/(hip_r+hop_r)*100:.1f}%)",
        f"HOP (not implicated)\n{hop_r:,} ({hop_r/(hip_r+hop_r)*100:.1f}%)",
    ]
    ax.pie(sizes, labels=labels, colors=[RED, BLUEGRAY, GREEN],
           wedgeprops=dict(edgecolor="white", linewidth=1.5), startangle=90,
           textprops=dict(fontsize=10))
    ax.set_title("Dataset record composition (30.0M records)")
    _save(fig, "07_record_composition")


def main():
    summary, hip, hop, samples, base_hip, base_hop = load_all()
    total = summary["total_records"]
    hip_r, hop_r = summary["HIP"]["n_records"], summary["HOP"]["n_records"]
    hip_s = summary["HIP"]["n_strains"]

    h = flag_runs(hip, ENR_FLAG)
    flagged = h[h["flagged"]].copy()
    runs = suspect_runs(h)
    flagged_records = int(flagged["n_exp"].sum())

    # whi2 cluster (highest-confidence, mechanism-anchored)
    c3 = hip[hip["whi2_corr"] >= WHI2_CORR_MIN]
    c3_records = int(c3["n_exp"].sum())

    # enrichment sweep for the tier table
    sweep = {}
    for e in ENR_SWEEP:
        he = flag_runs(hip, e)
        fe = he[he["flagged"]]
        sweep[e] = dict(n_strains=int(len(fe)), n_records=int(fe["n_exp"].sum()))

    mean_hip_cov = hip_r / hip_s  # for the paper-157 estimate
    tiers = [
        dict(short="whi2 cluster\n(mechanism)", n_strains=int(len(c3)), n_records=c3_records,
             pct_all_records=c3_records / total * 100),
        dict(short=f"empirical runs\n(>={ENR_FLAG:g}x)", n_strains=int(len(flagged)),
             n_records=flagged_records, pct_all_records=flagged_records / total * 100),
        dict(short="paper stated\n(157 HIP)", n_strains=157,
             n_records=int(157 * mean_hip_cov),
             pct_all_records=157 * mean_hip_cov / total * 100),
        dict(short="HIP envelope\n(conservative)", n_strains=hip_s, n_records=hip_r,
             pct_all_records=hip_r / total * 100),
    ]

    risk = dict(
        denominators=dict(
            total_records=total, hip_records=hip_r, hop_records=hop_r,
            hip_strains=hip_s, hop_strains=summary["HOP"]["n_strains"],
            hip_pct_records=hip_r / total * 100, hop_pct_records=hop_r / total * 100,
        ),
        baselines=dict(hip_neg_rate_4=base_hip, hop_neg_rate_4=base_hop),
        detector=dict(primary_thresh=PRIMARY_THRESH, roll_window=ROLL_WINDOW,
                      enr_flag=ENR_FLAG, whi2_corr_min=WHI2_CORR_MIN),
        tiers=dict(
            whi2_cluster=dict(n_strains=int(len(c3)), n_records=c3_records,
                              pct_all=c3_records / total * 100,
                              pct_hip=c3_records / hip_r * 100),
            empirical_runs_sweep={f"enr_{e:g}x": dict(
                **v, pct_all=v["n_records"] / total * 100,
                pct_hip=v["n_records"] / hip_r * 100) for e, v in sweep.items()},
            paper_characterized_157=dict(
                n_strains=157, n_records_estimate=int(157 * mean_hip_cov),
                pct_all_estimate=157 * mean_hip_cov / total * 100,
                note="record count ESTIMATED via mean HIP coverage; exact identities need Table S5"),
            conservative_hip_envelope=dict(n_strains=hip_s, n_records=hip_r,
                                           pct_all=hip_r / total * 100),
            hop_not_implicated=dict(n_records=hop_r, pct_all=hop_r / total * 100),
        ),
        suspect_runs=runs,
    )
    with open(osp.join(RESULTS_DIR, "risk_tiers.json"), "w") as fh:
        json.dump(risk, fh, indent=2)
    flagged_out = flagged.sort_values(["chrom_idx", "pos"])[
        ["orf", "gene", "chrom", "pos", "n_exp", "mean_sens", "frac_hyper_4",
         "enr", "roll_enr", "whi2_corr"]
    ]
    flagged_out.to_csv(osp.join(RESULTS_DIR, "flagged_hip_strains.csv"), index=False)

    print(f"HIP share of records: {hip_r/total*100:.1f}%  |  HOP: {hop_r/total*100:.1f}%")
    print(f"flagged HIP strains (>= {ENR_FLAG:g}x runs): {len(flagged)} "
          f"-> {flagged_records:,} records ({flagged_records/total*100:.2f}% of all)")
    print(f"whi2 cluster: {len(c3)} strains -> {c3_records:,} records "
          f"({c3_records/total*100:.2f}% of all)")
    print(f"suspect runs (>=3 adjacent): {len(runs)}")
    for r in runs[:8]:
        print(f"  chr {r['chrom']:<4} {r['pos_start']:>9}-{r['pos_end']:<9} "
              f"n={r['n_strains']:<3} whi2_frac={r['whi2_frac']:.2f}")

    print("\nrendering figures...")
    plot_score_distributions(samples, base_hip, base_hop)
    plot_promiscuity_ecdf(hip, hop)
    plot_genome_map(summary, h, hop)
    plot_whi2(hip, hop)
    plot_hip_vs_hop_specificity(hip, hop, h)
    plot_risk_tiers(tiers, summary)
    plot_composition(summary, flagged_records)
    print("done.")


if __name__ == "__main__":
    main()
