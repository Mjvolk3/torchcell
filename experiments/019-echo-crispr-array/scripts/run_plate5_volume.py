# experiments/019-echo-crispr-array/scripts/run_plate5_volume.py
# [[experiments.019-echo-crispr-array.scripts.run_plate5_volume]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run_plate5_volume
"""End-to-end IMAGE -> RESULTS for Plate 5 (OD1, 2.5 vs 5 nL): quantify the
standardized plate photo, register it to the ECHO picklist via the blank
pattern, normalize, compute per-strain fitness (mutant / BY4741) with SD split
by volume, and -- when reference data is available -- compare to the published
Costanzo 2016 / Kuzmin 2018 single-mutant fitness for the same 12-gene panel.

Assay-development context: first attempt at a CRISPR single-KO fitness assay for
the 12-gene panel, in service of trigenic-interaction quantification (exp 010).
Image-in / results-out.

Run from repo root:
    /Users/michaelvolk/miniconda3/bin/python \
        experiments/019-echo-crispr-array/scripts/run_plate5_volume.py
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
    recommend_volume,
    resolve_orientation,
    score_plate,
    score_table,
    volume_assay_metrics,
    volume_position_confound,
)
from torchcell.sga.assay import shape_by_volume
from torchcell.sga.viz import colony_shape_by_volume, plate_heatmap, value_histogram
from torchcell.utils import PLOT_PALETTE

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = osp.join(EXP_DIR, "data")
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

IMAGE = osp.join(DATA_DIR, "Original.png")
PICKLIST = osp.join(DATA_DIR, "ECHO_picklist_Plate5_384_OD1_2p5-5nL.csv")
REFERENCE = osp.join(RESULTS_DIR, "reference_smf_12panel.csv")  # optional


def save(fig, name):
    """Save as SVG (for the Dendron note) and a PNG (quick view)."""
    fig.savefig(osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.svg/.png")


def build_fitness_df(reports, cfg):
    """Tidy per-strain per-volume fitness with SD."""
    rows = []
    for vol, rep in reports.items():
        for s in rep.strains:
            if s.strain == cfg.blank_name or s.relative_fitness is None:
                continue
            rows.append(
                {
                    "strain": s.strain,
                    "volume_nl": vol,
                    "fitness": s.relative_fitness,
                    "fitness_sd": s.fitness_sd,
                    "n_used": s.n_used,
                    "pvalue": s.pvalue,
                }
            )
    return pd.DataFrame(rows)


def main():
    cfg = NormalizationConfig()

    print("[1] quantify image -> colony grid")
    grid = quantify_plate_image(
        IMAGE, overlay_path=osp.join(EXP_DIR, "quant", "plate5_overlay.png")
    )
    print(
        f"    {(grid['size'] > 0).sum()} colonies, {(grid['size'] == 0).sum()} empty, "
        f"{grid['flags'].str.contains('S').sum()} gash-flagged"
    )

    print("[2] register to picklist via blank pattern")
    layout = read_echo_picklist(PICKLIST)
    merged, op, agree = resolve_orientation(grid, layout)
    print(f"    orientation = {op}  (blank/plated agreement {agree:.1%})")

    print("[3] normalize (whole plate)")
    df = normalize_plate(merged, cfg)
    df.to_csv(osp.join(RESULTS_DIR, "plate5_colonies_registered.csv"), index=False)

    print("[4] fitness = mutant / BY4741 (normalized) with SD, split by volume")
    reports = {}
    for vol, sub in df.groupby("volume_nl"):
        rep = score_plate(sub, cfg, plate_id=f"Plate5_{vol}nL")
        reports[vol] = rep
        score_table(rep).sort_values("relative_fitness").to_csv(
            osp.join(RESULTS_DIR, f"plate5_strain_scores_{vol}nL.csv"), index=False
        )
    fit = build_fitness_df(reports, cfg)
    fit.to_csv(osp.join(RESULTS_DIR, "plate5_fitness_by_volume.csv"), index=False)
    wide = fit.pivot(index="strain", columns="volume_nl", values="fitness")
    wide_sd = fit.pivot(index="strain", columns="volume_nl", values="fitness_sd")
    show = wide.join(wide_sd, lsuffix="_fit", rsuffix="_sd")
    show = show.sort_values(show.columns[0])
    print(show.round(3).to_string())

    print(
        "\n[4b] plating success: attempted wells with NO colony (missing), per "
        "strain x volume (of 11 placed each)"
    )
    plated = df[df["strain"] != cfg.blank_name]
    ps = (
        plated.groupby(["strain", "volume_nl"])
        .agg(placed=("is_missing", "size"), no_colony=("is_missing", "sum"))
        .reset_index()
    )
    ps.to_csv(osp.join(RESULTS_DIR, "plate5_plating_success.csv"), index=False)
    print(ps.pivot(index="strain", columns="volume_nl", values="no_colony").to_string())
    blanks = df[df["strain"] == cfg.blank_name]
    print(
        f"    totals: {int(plated['is_missing'].sum())}/{len(plated)} plated wells "
        f"no colony; blanks {int(blanks['is_missing'].sum())}/{len(blanks)} empty (expected)"
    )
    # attribute losses to the gash: wells whose cell overlaps the detected tear (S flag)
    in_gash = plated["flags"].fillna("").astype(str).str.contains("S")
    n_gash_miss = int((in_gash & plated["is_missing"]).sum())
    print(
        f"    gash: {int(in_gash.sum())} wells touch the tear "
        f"({n_gash_miss} no-colony, {int((in_gash & ~plated['is_missing']).sum())} "
        f"grew-but-excluded); {int(plated['is_missing'].sum()) - n_gash_miss} of the "
        f"{int(plated['is_missing'].sum())} no-colony wells are NOT gash-related"
    )

    print("\n[5] CONFOUND CHECK")
    conf = volume_position_confound(df)
    print(f"    {'!! CONFOUNDED: ' + conf['detail'] if conf['confounded'] else 'ok'}")

    print("\n[6] assay-quality metrics per volume")
    metrics = volume_assay_metrics(df, cfg)
    metrics.to_csv(
        osp.join(RESULTS_DIR, "plate5_volume_assay_metrics.csv"), index=False
    )
    print(metrics.to_string(index=False))
    best_vol, why = recommend_volume(metrics)
    tag = " (UNSAFE - volume confounded with position)" if conf["confounded"] else ""
    print(f"    RECOMMENDATION -> {why}{tag}")

    print("\n[6b] colony SHAPE by volume (abnormal-shape check)")
    shape = shape_by_volume(df, cfg)
    shape.to_csv(osp.join(RESULTS_DIR, "plate5_shape_by_volume.csv"), index=False)
    print(shape.to_string(index=False))

    print("\n[7] figures (SVG + PNG)")
    save(colony_shape_by_volume(df, cfg), "plate5_colony_shape_by_volume")
    # one full-plate heatmap: both volumes, colorbar anchored at 0, divider between
    # the 2.5 nL block (cols 2-12) and the 5 nL block (cols 13-23).
    save(
        plate_heatmap(
            df,
            "norm",
            "Plate5 normalized colony size (1 = plate avg)",
            vmin=0.0,
            divider_after_col=12,
            half_labels=("2.5 nL", "5 nL"),
        ),
        "plate5_heatmap_norm",
    )
    for vol, sub in df.groupby("volume_nl"):
        save(
            value_histogram(sub[~sub["is_blank"]], "norm", f"Plate5 {vol} nL"),
            f"plate5_hist_norm_{vol}nL",
        )
    save(fitness_by_volume_plot(fit, cfg), "plate5_fitness_by_volume")
    save(wt_by_volume_plot(df, cfg), "plate5_wt_distribution_by_volume")
    save(assay_summary_plot(metrics, best_vol), "plate5_assay_quality_summary")

    if osp.exists(REFERENCE):
        ref = pd.read_csv(REFERENCE)
        n_ref = int(ref[["kuzmin_smf", "costanzo_smf"]].notna().any(axis=1).sum())
        print(f"\n[8] reference comparison: SMF available for {n_ref}/12 genes")
        save(fitness_vs_reference_bars(fit, ref, cfg), "plate5_fitness_vs_reference")
        if n_ref >= 3:
            save(
                fitness_vs_reference_scatter(fit, ref, cfg),
                "plate5_fitness_vs_reference_scatter",
            )
        else:
            print(
                "    scatter skipped (need >=3 reference points; Costanzo file absent)"
            )
    else:
        print(f"\n[8] no reference file at {REFERENCE}; skipping SMF comparison")

    print(f"\nDone. CSVs in {RESULTS_DIR}; figures in {IMG_DIR}")


def fitness_by_volume_plot(fit, cfg):
    vols = sorted(fit["volume_nl"].unique())
    strains = fit.groupby("strain")["fitness"].mean().sort_values().index.tolist()
    x = np.arange(len(strains))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(3.4, len(strains) * 0.34), 2.9))
    for i, vol in enumerate(vols):
        sub = fit[fit["volume_nl"] == vol].set_index("strain")
        vals = [sub.loc[s, "fitness"] if s in sub.index else np.nan for s in strains]
        errs = [sub.loc[s, "fitness_sd"] if s in sub.index else np.nan for s in strains]
        ax.bar(
            x + (i - 0.5) * w,
            vals,
            w,
            yerr=errs,
            capsize=1.5,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.5},
            label=f"{vol} nL",
        )
    ax.axhline(1.0, color=PLOT_PALETTE[5], ls="--", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(strains, rotation=90)
    ax.set_ylabel("fitness (mutant / BY4741) ± SD")
    ax.set_title("Plate5 single-KO fitness by volume")
    ax.legend(frameon=False, fontsize=5)
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def wt_by_volume_plot(df, cfg):
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    vols = sorted(df["volume_nl"].dropna().unique())
    data = [
        df[
            (df["volume_nl"] == v)
            & (df["strain"] == cfg.wt_name)
            & ~df["is_missing"]
            & ~df["is_flagged"]
        ]["norm"]
        .dropna()
        .to_numpy()
        for v in vols
    ]
    bp = ax.boxplot(
        data, labels=[f"{v} nL" for v in vols], patch_artist=True, widths=0.6
    )
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=PLOT_PALETTE[i], edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=0.8)
    ax.set_ylabel("BY4741 normalized size")
    ax.set_title("WT reproducibility")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def assay_summary_plot(metrics, best_vol):
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.2))
    vols = metrics["volume_nl"].tolist()
    labels = [f"{v} nL" for v in vols]
    colors = [PLOT_PALETTE[i] for i in range(len(vols))]
    for ax, (col, title) in zip(
        axes,
        [
            ("missing_rate", "missing rate"),
            ("wt_cv", "WT CV (lower=tighter)"),
            ("zfactor_wt_vs_weakest", "Z'-factor (higher=better)"),
        ],
    ):
        bars = ax.bar(
            labels, metrics[col], color=colors, edgecolor="black", linewidth=0.5
        )
        bars[vols.index(best_vol)].set(edgecolor=PLOT_PALETTE[1], linewidth=1.8)
        ax.set_title(title, fontsize=6)
        for s in ax.spines.values():
            s.set_visible(True)
    fig.suptitle(
        f"Plate5 assay quality by volume  (recommended: {best_vol} nL)", fontsize=7
    )
    fig.tight_layout()
    return fig


def _ref_join(fit, ref, cfg):
    """Join our per-strain per-volume fitness + SD to reference SMF via the
    strain<->ORF map carried in the reference file."""
    ours = fit.pivot(index="strain", columns="volume_nl", values="fitness")
    ours.columns = [f"ours_{c}nL" for c in ours.columns]
    sd = fit.pivot(index="strain", columns="volume_nl", values="fitness_sd")
    sd.columns = [f"ours_sd_{c}nL" for c in sd.columns]
    m = ref.set_index("strain").join(ours).join(sd)
    return m


def fitness_vs_reference_bars(fit, ref, cfg):
    m = _ref_join(fit, ref, cfg).sort_values(
        "ours_%snL" % sorted(fit["volume_nl"].unique())[-1]
    )
    genes = m.index.tolist()
    x = np.arange(len(genes))
    series = []
    for c in fit["volume_nl"].unique():
        series.append((f"ours {c} nL", f"ours_{c}nL", f"ours_sd_{c}nL"))
    src_cols = [c for c in ("costanzo_smf", "kuzmin_smf") if c in m.columns]
    sd_map = {"costanzo_smf": "costanzo_sd", "kuzmin_smf": "kuzmin_sd"}
    for c in src_cols:
        series.append((c.replace("_smf", "").capitalize() + " SMF", c, sd_map.get(c)))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(4.0, len(genes) * 0.5), 3.0))
    for i, (lab, col, sdcol) in enumerate(series):
        if col not in m.columns:
            continue
        errs = m[sdcol] if (sdcol and sdcol in m.columns) else None
        ax.bar(
            x + (i - (n - 1) / 2) * w,
            m[col],
            w,
            yerr=errs,
            capsize=1.5,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.5},
            label=lab,
        )
    ax.axhline(1.0, color=PLOT_PALETTE[5], ls="--", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{g}\n{m.loc[g, 'orf'] if 'orf' in m.columns else ''}" for g in genes],
        rotation=90,
        fontsize=5,
    )
    ax.set_ylabel("fitness (WT = 1) ± SD")
    ax.set_ylim(0, 1.45)  # headroom so the legend clears the bars
    ax.set_title("12-panel: our CRISPR fitness vs published SMF")
    ax.legend(
        frameon=False, fontsize=5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0)
    )
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def fitness_vs_reference_scatter(fit, ref, cfg):
    # prefer Costanzo (most genes + has SD); fall back to Kuzmin
    ref_col = "costanzo_smf" if ref["costanzo_smf"].notna().sum() >= 3 else "kuzmin_smf"
    sd_col = "costanzo_sd" if ref_col == "costanzo_smf" else "kuzmin_sd"
    m = _ref_join(fit, ref, cfg)
    our_col = f"ours_{sorted(fit['volume_nl'].unique())[-1]}nL"  # larger volume
    our_sd = f"ours_sd_{sorted(fit['volume_nl'].unique())[-1]}nL"
    d = m[[ref_col, sd_col, our_col, our_sd]].dropna(subset=[ref_col, our_col])
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    # on-brand: amber markers, gray error bars/diagonal (our warm palette)
    ax.errorbar(
        d[ref_col],
        d[our_col],
        xerr=d[sd_col],
        yerr=d[our_sd],
        fmt="o",
        ms=4,
        color=PLOT_PALETTE[0],
        ecolor=PLOT_PALETTE[5],
        elinewidth=0.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        zorder=3,
        capsize=1.5,
    )
    lims = [0, max(1.2, float(np.nanmax(d[[ref_col, our_col]].values)) * 1.1)]
    ax.plot(lims, lims, ls="--", color=PLOT_PALETTE[5], lw=0.6)
    if len(d) >= 3:
        r = float(np.corrcoef(d[ref_col], d[our_col])[0, 1])
        ax.text(
            0.05, 0.92, f"r = {r:.2f}  (n={len(d)})", transform=ax.transAxes, fontsize=6
        )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"published SMF ({ref_col.split('_')[0]}) ± SD")
    ax.set_ylabel(f"our fitness ({our_col.replace('ours_', '')})")
    ax.set_title("assay validation")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
