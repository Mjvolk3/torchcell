# experiments/019-echo-crispr-array/scripts/run2_volume_timepoints.py
# [[experiments.019-echo-crispr-array.scripts.run2_volume_timepoints]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run2_volume_timepoints
"""Run 2 (2026-07-17 plating): a SETTINGS SWEEP, not a batch experiment.

Two 384 plates carry the SAME randomized 13-sample layout (12 CRISPR single-KO
strains + BY4741 wild-type + 6 Blank_media no-cell controls). They differ ONLY in
dispensed volume -- P1 = 2.5 nL, P2 = 5 nL -- and each was imaged at THREE growth
times: ~43.7 h, ~50.3 h, ~72.2 h. That gives six (plate x timepoint) images.

These six are NOT replicate batches and nothing here estimates a "batch effect".
We have not randomized the layout, not replicated a volume, and not repeated a
plate -- so volume, plate, and position are fully confounded with one another and
cannot be separated. This run exists to FIND GOOD ASSAY SETTINGS (volume, growth
time), and every comparison below is read that way:
  * P1 vs P2 at a fixed time  -> the VOLUME contrast (confounded with plate);
  * the same plate across times -> the GROWTH-TIME contrast on identical colonies
    (this one is clean: same physical colonies, imaged repeatedly);
  * the spread of a strain across all six images is dominated by those two
    deliberately-varied factors, so it is NOT a measurement-error bar.

Because the two plates share ONE layout, each strain sits in the SAME wells on
both, so the plates do not resample position: positional bias is reproduced, not
averaged. A second, independently scrambled layout is the fix, and is a
picklist-generation change, not a pipeline change.

GEOMETRY: full 384 (A-P x 1-24, borders used, A1 origin); Plate 5 was a
border-free 14x22 inner block, so geometry is passed explicitly. ORIENTATION is
resolved per image (blanks + internal replicate structure) and asserted; see
``resolve_and_check``. Colony QUANTIFICATION now rejects any cell holding two or
more colonies ('M' flag) -- competing colonies each grow smaller, so the size is
not a faithful readout (torchcell/sga/image.py, multi_area_frac).

Run from repo root:
    /Users/michaelvolk/miniconda3/bin/python \
        experiments/019-echo-crispr-array/scripts/run2_volume_timepoints.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.stats import kruskal, pearsonr, spearmanr

from torchcell.sga import (
    NormalizationConfig,
    normalize_plate,
    quantify_plate_image,
    read_echo_picklist,
    score_plate,
    score_table,
)
from torchcell.sga.viz import plate_heatmap
from torchcell.utils import PLOT_PALETTE

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RUN2_DIR = osp.join(EXP_DIR, "data", "run2_2026-07-17")
RESULTS_DIR = osp.join(EXP_DIR, "results")
QUANT_DIR = osp.join(EXP_DIR, "quant")
PROC_DIR = osp.join(EXP_DIR, "quant", "run2_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")
for d in (RESULTS_DIR, QUANT_DIR, PROC_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

REFERENCE = osp.join(RESULTS_DIR, "reference_smf_12panel.csv")
PLATE5_FITNESS = osp.join(RESULTS_DIR, "plate5_fitness_by_volume.csv")

N_ROWS, N_COLS = 16, 24  # full 384, A1 origin (Plate 5 was 14x22 from B2)

# Growth clock. Plates into the incubator 2026-07-17 15:30 (user-recorded);
# imaging times are EXIF DateTimeOriginal. Elapsed hours use 15:30. The ECHO
# transfer reports timestamp dispensing ~1 h later (an unresolved instrument-clock
# discrepancy); if the instrument clock is right, all elapsed figures shift down
# ~1 h, but the DIFFERENCES between timepoints (from EXIF stamps) are unaffected.
#
# Incubation orientation (user): agar-side-UP from plating until the first assay
# timepoint (t44), agar-side-DOWN thereafter (t50, t72). The concern behind the
# flip was that agar-up colonies might dome/droplet and distort their projected
# footprint; the t44-vs-later comparison is where that would show.
# Images are the full-resolution 12.2 MP originals (the earlier 0.8 MP Photos
# `derivatives/` previews were discarded -- see the discard log). They are
# preprocessed (crop off the metal incubator shelf + downscale to PROCESS_WIDTH)
# before quantification; see _preprocess.
CONDITIONS = [
    dict(
        group="P1_t44",
        plate="P1",
        volume_nl=2.5,
        hours=43.71,
        agar="up",
        image=osp.join(RUN2_DIR, "P1_2p5nL_view_t44.jpg"),
        picklist=osp.join(RUN2_DIR, "P1_2p5nL_cherrypick_13strain.csv"),
    ),
    dict(
        group="P2_t44",
        plate="P2",
        volume_nl=5.0,
        hours=43.71,
        agar="up",
        image=osp.join(RUN2_DIR, "P2_5nL_view_t44.jpg"),
        picklist=osp.join(RUN2_DIR, "P2_5nL_cherrypick_13strain.csv"),
    ),
    dict(
        group="P1_t50",
        plate="P1",
        volume_nl=2.5,
        hours=50.30,
        agar="down",
        image=osp.join(RUN2_DIR, "t50", "P1_2p5nL_view_t50.jpg"),
        picklist=osp.join(RUN2_DIR, "P1_2p5nL_cherrypick_13strain.csv"),
    ),
    dict(
        group="P2_t50",
        plate="P2",
        volume_nl=5.0,
        hours=50.30,
        agar="down",
        image=osp.join(RUN2_DIR, "t50", "P2_5nL_view_t50.jpg"),
        picklist=osp.join(RUN2_DIR, "P2_5nL_cherrypick_13strain.csv"),
    ),
    # t72 = full-resolution (3364x4485) quality-side-UP captures (the canonical
    # correct images; earlier low-res 1499x1999 previews were removed). Same physical
    # plates at ~72 h, so growth-time values are retained (no EXIF timestamp to refine).
    dict(
        group="P1_t72",
        plate="P1",
        volume_nl=2.5,
        hours=72.18,
        agar="down",
        image=osp.join(RUN2_DIR, "t72", "P1_2p5nL_view_t72_up.jpg"),
        picklist=osp.join(RUN2_DIR, "P1_2p5nL_cherrypick_13strain.csv"),
    ),
    dict(
        group="P2_t72",
        plate="P2",
        volume_nl=5.0,
        hours=72.26,
        agar="down",
        image=osp.join(RUN2_DIR, "t72", "P2_5nL_view_t72_up.jpg"),
        picklist=osp.join(RUN2_DIR, "P2_5nL_cherrypick_13strain.csv"),
    ),
]
for _c in CONDITIONS:
    _c.setdefault("reliable", True)
GROUPS = [c["group"] for c in CONDITIONS]
RELIABLE = [c["group"] for c in CONDITIONS if c["reliable"]]

# Human-readable condition label for axes/legends: volume (nL) + growth time,
# instead of the internal P1/P2 codes. P1 = 2.5 nL, P2 = 5 nL throughout.
COND_LABEL = {
    c["group"]: f"{c['volume_nl']:g} nL, {int(round(c['hours']))} h" for c in CONDITIONS
}


# Strain display: the picklist Sample Name (a mix of systematic and common) ->
# "SYSTEMATIC (COMMON)", always leading with the systematic ORF and appending the
# common name where one exists. Built from the reference table (strain, orf,
# common_name). Uncharacterized ORFs (YER079W, YKL033W-A, YLR312C-B) have no common
# name and render as the ORF alone.
def _load_strain_display():
    m = {"BY4741": "BY4741 (WT)", "Blank_media": "Blank_media"}
    if osp.exists(REFERENCE):
        ref = pd.read_csv(REFERENCE)
        for _, r in ref.iterrows():
            orf = str(r["orf"])
            common = r.get("common_name")
            common = (
                None if (pd.isna(common) or str(common).strip() == "") else str(common)
            )
            m[str(r["strain"])] = f"{orf} ({common})" if common else orf
    return m


STRAIN_DISPLAY = _load_strain_display()


def display_gene(strain: str) -> str:
    return STRAIN_DISPLAY.get(strain, strain)


# Process each plate photo at a fixed width: crop to the bright plate (drops the
# dark metal incubator shelf), then downscale. The full 12 MP is too textured for
# the lattice fit (colony interiors fragment; satellite blobs pollute the grid
# fit); ~1400 px averages colony texture into solid blobs so detection +
# registration are robust, while staying ~2x sharper than the old 0.8 MP previews.
PROCESS_WIDTH = 1400


def _preprocess(path: str) -> str:
    """Crop to the plate and downscale to PROCESS_WIDTH; return a temp PNG path."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    g = np.asarray(im.convert("L"), float)
    bright = ndimage.gaussian_filter(g, 40) > 0.80 * np.percentile(g, 99)
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = int(sizes.argmax()) + 1
    ys, xs = np.where(lab == big)
    pad = int(0.02 * max(g.shape))
    r0, r1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + pad)
    c0, c1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + pad)
    crop = im.crop((c0, r0, c1, r1))
    w = PROCESS_WIDTH
    crop = crop.resize((w, max(1, int(crop.height * w / crop.width))))
    out = osp.join(PROC_DIR, osp.splitext(osp.basename(path))[0] + "_proc.png")
    crop.save(out)
    return out


# ORIENTATION -- resolved per image by INTERNAL STRAIN STRUCTURE, then asserted.
#
# The plate can be imaged in any of 4 shape-preserving orientations. The blank
# wells cannot fix it: they sit at rows E,L and 5+12=17=N_ROWS+1, so their pattern
# is symmetric under a vertical flip, AND their purity DEGRADES at long growth --
# by 72 h a blank can pick up a contaminant (P1) or be overrun by a spreading
# neighbour (P2, where all six blanks are occupied). So orientation is decided by
# Kruskal-Wallis H across strain groups: the true labelling makes real replicate
# groups (large H); a flipped labelling scrambles strains (H collapses to noise).
# The chosen op must beat the runner-up by H_MARGIN and clear H_MIN, else we
# genuinely cannot tell and fail loudly. Blank purity is then reported as QC, not
# used to gate. Published data is NOT used, so Costanzo stays independent. t44/t50
# resolve to 'identity' (A1 top-left, H 59-103); t72 resolves to 'identity' too
# (H ~44) despite occupied blanks.
OPS = ("identity", "rot180", "flip_v", "flip_h")
H_MARGIN = 2.5  # best strain-structure H must exceed runner-up by this factor
H_MIN = 20.0  # ...and clear this floor (real between-strain structure exists)


def apply_orientation(grid: pd.DataFrame, op: str) -> pd.DataFrame:
    """Map IMAGE-order (row 1 = top, col 1 = left) to plate coordinates."""
    r, c = grid["row"], grid["col"]
    if op == "identity":
        nr, nc = r, c
    elif op == "rot180":
        nr, nc = N_ROWS + 1 - r, N_COLS + 1 - c
    elif op == "flip_v":
        nr, nc = N_ROWS + 1 - r, c
    elif op == "flip_h":
        nr, nc = r, N_COLS + 1 - c
    else:
        raise ValueError(op)
    out = grid.copy()
    out["row"], out["col"] = nr, nc
    return out


def _strain_structure_H(grid, layout, cfg):
    """Kruskal-Wallis H across strain groups after a full normalize -- the signal
    that separates the true labelling from a flipped one."""
    m = grid.merge(layout, on=["row", "col"], how="inner")
    if not (m["strain"] != cfg.blank_name).any():
        return 0.0
    df = normalize_plate(m, cfg)
    used = df[
        ~df["is_missing"] & ~df["is_flagged"] & ~df["is_blank"] & ~df["is_jackknife"]
    ]
    groups = [
        g["norm"].dropna().to_numpy() for _, g in used.groupby("strain") if len(g) >= 5
    ]
    if len(groups) < 3:
        return 0.0
    return float(kruskal(*groups)[0])


def resolve_and_check(grid, layout, cfg, group):
    """Return (best_op, blanks_empty, diagnostics_df). Pick the op with the
    strongest strain structure; require it to clear H_MIN and beat the runner-up
    by H_MARGIN, else fail loudly (orientation genuinely ambiguous)."""
    rows = []
    for op in OPS:
        m = apply_orientation(grid, op).merge(layout, on=["row", "col"], how="inner")
        is_blank = m["strain"] == cfg.blank_name
        present = m["size"] > cfg.min_size
        rows.append(
            dict(
                group=group,
                op=op,
                n_joined=len(m),
                n_blanks=int(is_blank.sum()),
                blanks_empty=int((is_blank & ~present).sum()),
                plated_grew=int((~is_blank & present).sum()),
                n_plated=int((~is_blank).sum()),
                agreement=float(((is_blank & ~present) | (~is_blank & present)).mean()),
                strain_H=_strain_structure_H(apply_orientation(grid, op), layout, cfg),
            )
        )
    diag = (
        pd.DataFrame(rows)
        .sort_values("strain_H", ascending=False)
        .reset_index(drop=True)
    )
    best, runner = diag.iloc[0], diag.iloc[1]
    if best["strain_H"] < H_MIN or best["strain_H"] < H_MARGIN * max(
        runner["strain_H"], 1e-9
    ):
        raise AssertionError(
            f"{group}: orientation ambiguous -- best op '{best['op']}' has strain-"
            f"structure H={best['strain_H']:.1f}, runner-up '{runner['op']}' "
            f"H={runner['strain_H']:.1f} (need H>={H_MIN} and >{H_MARGIN}x runner-up). "
            f"Inspect the overlay before using any number.\n{diag.to_string(index=False)}"
        )
    be = int(best["blanks_empty"])
    return best["op"], be, diag


def save(fig, name):
    fig.savefig(osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {name}.svg/.png")


def main():
    cfg = NormalizationConfig()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )

    # ---------------------------------------------------------------- quantify
    print("[1] quantify + register + normalize (6 images; multi-colony cells rejected)")
    colonies, reports, orient_rows = {}, {}, []
    for cond in CONDITIONS:
        g = cond["group"]
        # Preprocess (crop off the shelf + downscale to PROCESS_WIDTH), then
        # grid_mode='lattice' for the backlit capture. Overlay goes to the ASSET
        # image dir so it can be embedded next to the heatmaps.
        proc = _preprocess(cond["image"])
        grid = quantify_plate_image(
            proc,
            n_rows=N_ROWS,
            n_cols=N_COLS,
            overlay_path=osp.join(IMG_DIR, f"run2_overlay_{g}.png"),
            grid_mode="lattice",
        )
        layout = read_echo_picklist(cond["picklist"])
        op, blanks_empty, diag = resolve_and_check(grid, layout, cfg, g)
        orient_rows.append(diag)

        merged = apply_orientation(grid, op).merge(
            layout, on=["row", "col"], how="inner"
        )
        df = normalize_plate(merged, cfg)
        for k in ("group", "plate", "volume_nl", "hours", "agar"):
            df[k] = cond[k]
        colonies[g] = df
        rep = score_plate(df, cfg, plate_id=g)
        reports[g] = rep

        n_multi = int(df["flags"].fillna("").str.contains("M").sum())
        grew = int((~df["is_missing"]).sum())
        # Occupied blanks are a QC/overgrowth signal, not an orientation problem.
        blank_note = ""
        if blanks_empty < 6:
            blank_note = (
                f"  !! {6 - blanks_empty}/6 blanks OCCUPIED "
                f"(contamination or neighbour overgrowth)"
            )
        print(
            f"    {g}: op={op}  {grew}/{len(df)} colonies, "
            f"{int(df['is_missing'].sum())} empty, {n_multi} multi-colony rejected, "
            f"blanks empty {blanks_empty}/6, WT median norm {rep.wt_median_norm:.3f}"
            f"{blank_note}"
        )

    all_col = pd.concat(colonies.values(), ignore_index=True)
    all_col.to_csv(osp.join(RESULTS_DIR, "run2_colonies_registered.csv"), index=False)
    pd.concat(orient_rows, ignore_index=True).to_csv(
        osp.join(RESULTS_DIR, "run2_orientation_diagnostics.csv"), index=False
    )

    # ------------------------------------------------------------------- score
    print("\n[2] per-strain fitness (mutant / BY4741) per condition")
    rows = []
    for cond in CONDITIONS:
        g = cond["group"]
        for s in reports[g].strains:
            if s.strain == cfg.blank_name or s.relative_fitness is None:
                continue
            rows.append(
                dict(
                    group=g,
                    plate=cond["plate"],
                    volume_nl=cond["volume_nl"],
                    hours=cond["hours"],
                    strain=s.strain,
                    fitness=s.relative_fitness,
                    fitness_sd=s.fitness_sd,
                    n_used=s.n_used,
                    n_total=s.n_total,
                    pvalue=s.pvalue,
                )
            )
        score_table(reports[g]).sort_values("relative_fitness").to_csv(
            osp.join(RESULTS_DIR, f"run2_strain_scores_{g}.csv"), index=False
        )
    fit = pd.DataFrame(rows)
    fit.to_csv(osp.join(RESULTS_DIR, "run2_fitness_by_condition.csv"), index=False)
    wide = fit.pivot(index="strain", columns="group", values="fitness")[GROUPS]
    print(wide.round(3).to_string())

    # --------------------------------------------------- published SMF compare
    print("\n[3] comparison to published single-mutant fitness")
    ref = pd.read_csv(REFERENCE)
    comp = ref.merge(fit, on="strain", how="inner")
    rr = ref["costanzo_smf"].dropna()
    neutral = rr[rr > 0.9]
    print(
        f"    reference range {rr.min():.3f}..{rr.max():.3f}; excluding the one sick "
        f"gene, {len(neutral)}/{len(rr)} genes span only {neutral.min():.3f}.."
        f"{neutral.max():.3f} (spread {neutral.max() - neutral.min():.3f}) -- so a high "
        f"correlation here rests almost entirely on the one sick gene"
    )
    ref_rows = []
    for g in GROUPS:
        d = (
            comp[comp["group"] == g]
            .dropna(subset=["costanzo_smf", "fitness"])
            .set_index("strain")
        )
        r, p = pearsonr(d["costanzo_smf"], d["fitness"])
        rho, prho = spearmanr(d["costanzo_smf"], d["fitness"])
        loo = {
            s: pearsonr(d.drop(s)["costanzo_smf"], d.drop(s)["fitness"])[0]
            for s in d.index
        }
        worst = min(loo, key=loo.get)
        ref_rows.append(
            dict(
                group=g,
                n=len(d),
                pearson_r=r,
                pearson_p=p,
                spearman_rho=rho,
                spearman_p=prho,
                loo_r_min=min(loo.values()),
                loo_r_max=max(loo.values()),
                most_influential=worst,
                rmse=float(np.sqrt(((d["fitness"] - d["costanzo_smf"]) ** 2).mean())),
                median_bias=float((d["fitness"] - d["costanzo_smf"]).median()),
            )
        )
        print(
            f"    {g}: Pearson r={r:+.3f} (p={p:.1e}), Spearman rho={rho:+.3f}, "
            f"n={len(d)}, RMSE {ref_rows[-1]['rmse']:.3f}, bias {ref_rows[-1]['median_bias']:+.3f}, "
            f"LOO r {min(loo.values()):+.3f}..{max(loo.values()):+.3f} (drop {worst})"
        )
    ref_stats = pd.DataFrame(ref_rows)
    ref_stats.to_csv(osp.join(RESULTS_DIR, "run2_vs_reference_stats.csv"), index=False)
    comp.to_csv(osp.join(RESULTS_DIR, "run2_vs_reference.csv"), index=False)

    # ----------------------------------------------- per-condition quality
    print("\n[4] per-condition quality + dynamic range")
    qual = []
    for cond in CONDITIONS:
        g = cond["group"]
        d = colonies[g]
        used = d[
            ~d["is_missing"] & ~d["is_flagged"] & ~d["is_blank"] & ~d["is_jackknife"]
        ]
        wt = used[used["strain"] == cfg.wt_name]["norm"].dropna()
        plated = d[~d["is_blank"]]
        blanks = d[d["is_blank"]]
        groups_k = [
            x["norm"].dropna().to_numpy()
            for _, x in used.groupby("strain")
            if len(x) >= 5
        ]
        H, pH = kruskal(*groups_k)
        qual.append(
            dict(
                group=g,
                plate=cond["plate"],
                volume_nl=cond["volume_nl"],
                hours=cond["hours"],
                agar=cond["agar"],
                reliable=cond["reliable"],
                n_colonies=len(d),
                missing=int(plated["is_missing"].sum()),
                missing_rate=float(plated["is_missing"].mean()),
                multi_rejected=int(d["flags"].fillna("").str.contains("M").sum()),
                median_size_px=float(d.loc[~d["is_missing"], "size"].median()),
                wt_n=len(wt),
                wt_cv=float(wt.std() / wt.mean()),
                blanks_empty=int(blanks["is_missing"].sum()),
                n_blanks=len(blanks),
                kruskal_H=float(H),
                kruskal_p=float(pH),
            )
        )
    qual = pd.DataFrame(qual)
    ko = fit[fit["strain"] != cfg.wt_name].copy()
    ko["se"] = ko["fitness_sd"] / np.sqrt(ko["n_used"].clip(lower=1))
    dyn = (
        ko.groupby("group")
        .agg(
            fitness_min=("fitness", "min"),
            fitness_max=("fitness", "max"),
            between_strain_sd=("fitness", "std"),
            mean_within_se=("se", "mean"),
        )
        .assign(
            fitness_range=lambda d: d["fitness_max"] - d["fitness_min"],
            discrimination=lambda d: d["between_strain_sd"] / d["mean_within_se"],
        )
    )
    qual = qual.merge(dyn.reset_index(), on="group")
    qual.to_csv(osp.join(RESULTS_DIR, "run2_condition_quality.csv"), index=False)
    print(qual.round(4).to_string(index=False))
    if set(GROUPS) - set(RELIABLE):
        print(
            f"    NOTE: {sorted(set(GROUPS) - set(RELIABLE))} excluded from "
            f"quantitative conclusions (overgrown/misregistered); shown for evidence only."
        )

    # ---------------------------------------------------- what varies across conditions
    print(
        "\n[5] what changes across conditions (this is a settings sweep, not batches)"
    )
    # Same plate across time = clean within-plate growth contrast; different plate
    # = volume contrast confounded with plate. Report both plainly.
    corr = wide.corr(method="pearson")
    corr.to_csv(osp.join(RESULTS_DIR, "run2_condition_correlation.csv"))
    print("    per-strain fitness agreement (Pearson r) between conditions:")
    print(corr.round(3).to_string())

    eff = pd.DataFrame(
        {
            "volume_2p5_minus_5_t44": wide["P1_t44"] - wide["P2_t44"],
            "volume_2p5_minus_5_t50": wide["P1_t50"] - wide["P2_t50"],
            "volume_2p5_minus_5_t72": wide["P1_t72"] - wide["P2_t72"],
            "time_t50_minus_t44_P1": wide["P1_t50"] - wide["P1_t44"],
            "time_t72_minus_t50_P1": wide["P1_t72"] - wide["P1_t50"],
            "time_t50_minus_t44_P2": wide["P2_t50"] - wide["P2_t44"],
            "time_t72_minus_t50_P2": wide["P2_t72"] - wide["P2_t50"],
        }
    )
    eff.to_csv(osp.join(RESULTS_DIR, "run2_volume_time_effects.csv"))
    print("\n    median effect over strains (Δ relative fitness):")
    for c in eff.columns:
        print(
            f"      {c:26s} {eff[c].median():+.4f}  "
            f"(IQR {eff[c].quantile(0.25):+.3f} .. {eff[c].quantile(0.75):+.3f})"
        )

    # ---------------------------------------------------- saturation over time
    # CRITICAL: raw colony pixel-area is NOT comparable across timepoints because
    # the t72 images are a different camera (iPhone 16 Pro, ~3364x4485) vs t44/t50
    # (~768x1024) -- ~19x more pixels per unit area. So we express size as a
    # RESOLUTION-INVARIANT fraction of the well pitch: cell_fill = area_px /
    # pitch_px^2, where pitch_px is recovered per image from the colony-centroid
    # spacing. The 4.5 mm array pitch is physically constant, so cell_fill is
    # comparable across images and proportional to real colony footprint.
    print("\n[6] saturation: resolution-invariant colony footprint vs growth time")

    def pitch_px(d):
        """Median pixel spacing between adjacent plate columns, from centroids."""
        by_col = d.dropna(subset=["cx"]).groupby("col")["cx"].median()
        return (
            float(np.median(np.diff(by_col.sort_index().to_numpy())))
            if len(by_col) > 1
            else np.nan
        )

    sat_rows = []
    for cond in CONDITIONS:
        g = cond["group"]
        d = colonies[g]
        p = pitch_px(d)
        used = d[
            ~d["is_missing"] & ~d["is_flagged"] & ~d["is_blank"] & ~d["is_jackknife"]
        ].copy()
        used["fill"] = used["size"] / (p**2)  # fraction of the well cell filled
        wt = used[used["strain"] == cfg.wt_name]
        sat_rows.append(
            dict(
                group=g,
                plate=cond["plate"],
                volume_nl=cond["volume_nl"],
                hours=cond["hours"],
                pitch_px=p,
                wt_median_fill=float(wt["fill"].median()),
                median_fill=float(used["fill"].median()),
                p10_fill=float(used["fill"].quantile(0.10)),
                p90_fill=float(used["fill"].quantile(0.90)),
                raw_median_size_px=float(used["size"].median()),
            )
        )
    sat = pd.DataFrame(sat_rows)
    sat.to_csv(osp.join(RESULTS_DIR, "run2_saturation_by_time.csv"), index=False)
    # Use the plate-wide MEDIAN colony fill (not WT-specific) for the size trend:
    # it is registration-independent, so it stays valid for P2_t72 whose strain
    # labels are shifted. The right-hand fitness panel IS label-dependent, so
    # P2_t72's fitness point there is flagged unreliable.
    for plate in ("P1", "P2"):
        s = sat[sat["plate"] == plate].sort_values("hours")
        seq = "  ".join(
            f"{h:.0f}h:{v:.1%}" for h, v in zip(s["hours"], s["median_fill"])
        )
        note = ""
        if len(s) == 3:
            g1 = s["median_fill"].iloc[1] - s["median_fill"].iloc[0]
            g2 = s["median_fill"].iloc[2] - s["median_fill"].iloc[1]
            per_h1 = g1 / (s["hours"].iloc[1] - s["hours"].iloc[0])
            per_h2 = g2 / (s["hours"].iloc[2] - s["hours"].iloc[1])
            note = f"  cell-fill growth {per_h1:+.3%}/h then {per_h2:+.3%}/h -> " + (
                "SATURATING" if abs(per_h2) < 0.5 * abs(per_h1) else "still growing"
            )
        print(
            f"    {plate} WT cell-fill (pitch {s['pitch_px'].iloc[0]:.0f}px): {seq}{note}"
        )

    # optional: Plate 5 as an older, separate run (different geometry/design)
    if osp.exists(PLATE5_FITNESS):
        p5 = pd.read_csv(PLATE5_FITNESS)
        p5w = p5.pivot(index="strain", columns="volume_nl", values="fitness")
        p5w.columns = [f"plate5_{c}nL" for c in p5w.columns]
        joint = wide.join(p5w, how="inner")
        joint.corr(method="pearson").to_csv(
            osp.join(RESULTS_DIR, "run2_vs_plate5_correlation.csv")
        )

    # ----------------------------------------------------------------- figures
    print("\n[7] figures")
    save(fitness_by_condition_plot(fit), "run2_fitness_by_condition")
    save(reference_bars(comp, wide), "run2_fitness_vs_reference")
    save(reference_scatter_grid(comp, ref_stats), "run2_fitness_vs_reference_scatter")
    save(condition_corr_plot(corr), "run2_condition_correlation")
    save(quality_plot(colonies, qual, cfg), "run2_condition_quality")
    save(dynamic_range_plot(qual, wide, ref), "run2_dynamic_range")
    save(saturation_plot(sat, wide), "run2_saturation")
    for g in GROUPS:
        save(
            plate_heatmap(
                colonies[g],
                "norm",
                f"run 2 {g}: normalized colony size (1 = plate avg)",
                vmin=0.0,
            ),
            f"run2_heatmap_norm_{g}",
        )

    print(f"\nDone. CSVs in {RESULTS_DIR}; figures + overlays in {IMG_DIR}")


# --------------------------------------------------------------------- figures
def fitness_by_condition_plot(fit):
    order = fit.groupby("strain")["fitness"].mean().sort_values().index.tolist()
    x = np.arange(len(order))
    w = 0.8 / len(GROUPS)
    fig, ax = plt.subplots(figsize=(max(4.5, len(order) * 0.62), 3.0))
    for i, g in enumerate(GROUPS):
        sub = fit[fit["group"] == g].set_index("strain")
        vals = [sub.loc[s, "fitness"] if s in sub.index else np.nan for s in order]
        errs = [sub.loc[s, "fitness_sd"] if s in sub.index else np.nan for s in order]
        ax.bar(
            x + (i - (len(GROUPS) - 1) / 2) * w,
            vals,
            w,
            yerr=errs,
            capsize=1.0,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.3,
            error_kw={"elinewidth": 0.4},
            label=COND_LABEL[g],
        )
    ax.axhline(1.0, color=PLOT_PALETTE[5], ls="--", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([display_gene(s) for s in order], rotation=90)
    ax.set_ylabel("fitness (mutant / BY4741) ± SD")
    ax.set_title(
        "Run 2: single-KO fitness across 6 conditions (2 volumes × 3 timepoints)"
    )
    ax.legend(frameon=False, fontsize=5, ncol=6)
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def reference_bars(comp, wide):
    genes = comp[["strain", "orf"]].drop_duplicates().set_index("strain")
    mean_fit = wide.mean(axis=1)
    sd_fit = wide.std(axis=1)
    m = (
        comp[["strain", "costanzo_smf", "costanzo_sd"]]
        .drop_duplicates()
        .set_index("strain")
        .join(genes)
        .assign(fitness=mean_fit, spread=sd_fit)
        .dropna(subset=["costanzo_smf"])
        .sort_values("costanzo_smf")
    )
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(max(4.0, len(m) * 0.5), 3.0))
    ax.bar(
        x - 0.2,
        m["fitness"],
        0.4,
        yerr=m["spread"],
        capsize=1.5,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.5},
        label="ours (mean of 6 conditions +/- SD across conditions)",
    )
    ax.bar(
        x + 0.2,
        m["costanzo_smf"],
        0.4,
        yerr=m["costanzo_sd"],
        capsize=1.5,
        color=PLOT_PALETTE[4],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.5},
        label="Costanzo 2016 SMF ± SD",
    )
    ax.axhline(1.0, color=PLOT_PALETTE[5], ls="--", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([display_gene(i) for i in m.index], rotation=90, fontsize=5)
    ax.set_ylabel("fitness (WT = 1)")
    ax.set_ylim(0, 1.5)
    ax.set_title("Run 2 vs published single-mutant fitness")
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    return fig


def reference_scatter_grid(comp, ref_stats):
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6), sharex=True, sharey=True)
    for ax, g in zip(axes.ravel(), GROUPS):
        d = comp[comp["group"] == g].dropna(subset=["costanzo_smf", "fitness"])
        st = ref_stats[ref_stats["group"] == g].iloc[0]
        ax.errorbar(
            d["costanzo_smf"],
            d["fitness"],
            xerr=d["costanzo_sd"],
            yerr=d["fitness_sd"],
            fmt="o",
            ms=3,
            color=PLOT_PALETTE[0],
            ecolor=PLOT_PALETTE[5],
            elinewidth=0.4,
            capsize=1.0,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
        lims = [0.3, 1.35]
        ax.plot(lims, lims, ls="--", color=PLOT_PALETTE[5], lw=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # Pearson r = agreement in VALUE; Spearman rho = agreement in ORDER (does the
        # ranking of strains by fitness match Costanzo's, even if the values are off).
        ax.set_title(
            f"{COND_LABEL[g]}\nPearson r={st['pearson_r']:+.2f}   "
            f"Spearman ρ={st['spearman_rho']:+.2f}",
            fontsize=5.5,
        )
        for s in ax.spines.values():
            s.set_visible(True)
    for ax in axes[-1]:
        ax.set_xlabel("Costanzo 2016 SMF")
    for ax in axes[:, 0]:
        ax.set_ylabel("our fitness")
    fig.suptitle("Published-SMF agreement, per condition", fontsize=7)
    fig.tight_layout()
    return fig


def condition_corr_plot(corr):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(corr.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels([COND_LABEL[g] for g in corr.columns], rotation=90)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([COND_LABEL[g] for g in corr.index])
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.to_numpy()[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if v < 0.75 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r")
    ax.set_title("agreement between conditions\n(per-strain fitness)", fontsize=6)
    fig.tight_layout()
    return fig


def quality_plot(colonies, qual, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    colors = [PLOT_PALETTE[i] for i in range(len(GROUPS))]
    wt = [
        colonies[g][
            (colonies[g]["strain"] == cfg.wt_name)
            & ~colonies[g]["is_missing"]
            & ~colonies[g]["is_flagged"]
        ]["norm"]
        .dropna()
        .to_numpy()
        for g in GROUPS
    ]
    bp = axes[0].boxplot(wt, patch_artist=True, widths=0.6)
    for box, col in zip(bp["boxes"], colors):
        box.set(facecolor=col, edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=0.8)
    labels = [COND_LABEL[g] for g in GROUPS]
    axes[0].set_xticks(range(1, len(GROUPS) + 1))
    axes[0].set_xticklabels(labels, rotation=90)
    axes[0].set_ylabel("BY4741 colony size / plate reference")
    # The WT MEDIAN is 1 by construction (fitness is size / WT-median); this box
    # shows the spread of the ~30 individual WT replicate colonies about that 1,
    # i.e. how reproducible a single colony measurement is (tighter = better).
    axes[0].set_title("WT replicate spread (reproducibility)", fontsize=6)

    q = qual.set_index("group").loc[GROUPS]
    axes[1].bar(
        range(len(GROUPS)),
        q["missing_rate"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xticks(range(len(GROUPS)))
    axes[1].set_xticklabels(labels, rotation=90)
    axes[1].set_ylabel("fraction of plated wells with no colony")
    axes[1].set_title("plating failure", fontsize=6)

    axes[2].bar(
        range(len(GROUPS)), q["wt_cv"], color=colors, edgecolor="black", linewidth=0.5
    )
    axes[2].set_xticks(range(len(GROUPS)))
    axes[2].set_xticklabels(labels, rotation=90)
    axes[2].set_ylabel("WT CV (lower = tighter)")
    axes[2].set_title("measurement precision", fontsize=6)
    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(True)
    fig.tight_layout()
    return fig


def dynamic_range_plot(qual, wide, ref, anchor="YJR060W"):
    q = qual.set_index("group").loc[GROUPS]
    colors = [PLOT_PALETTE[i] for i in range(len(GROUPS))]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.2, 2.7))
    labels = [COND_LABEL[g] for g in GROUPS]
    axl.bar(
        range(len(GROUPS)),
        q["fitness_range"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for i, g in enumerate(GROUPS):
        axl.text(
            i,
            q.loc[g, "fitness_range"] + 0.006,
            f"d={q.loc[g, 'discrimination']:.1f}",
            ha="center",
            fontsize=4.5,
        )
    axl.set_xticks(range(len(GROUPS)))
    axl.set_xticklabels(labels, rotation=90)
    axl.set_ylabel("fitness range across strains (max - min)")
    axl.set_title(
        "dynamic range\n(d = between-strain SD / within-strain SE)", fontsize=6
    )

    pub = float(ref.loc[ref["strain"] == anchor, "costanzo_smf"].iloc[0])
    pub_sd = float(ref.loc[ref["strain"] == anchor, "costanzo_sd"].iloc[0])
    axr.bar(
        range(len(GROUPS)),
        [wide.loc[anchor, g] for g in GROUPS],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axr.set_xticks(range(len(GROUPS)))
    axr.set_xticklabels(labels, rotation=90)
    axr.axhline(
        pub, color=PLOT_PALETTE[1], ls="--", lw=0.8, label=f"Costanzo = {pub:.2f}"
    )
    axr.axhspan(pub - pub_sd, pub + pub_sd, color=PLOT_PALETTE[1], alpha=0.15, lw=0)
    axr.axhline(1.0, color=PLOT_PALETTE[5], ls=":", lw=0.6, label="wild-type = 1")
    axr.set_ylim(0, 1.1)
    axr.set_ylabel("measured fitness")
    axr.set_title(f"{anchor}: the sick-strain anchor", fontsize=6)
    axr.legend(frameon=False, fontsize=5, loc="lower right")
    axr.tick_params(axis="x", rotation=90)
    for ax in (axl, axr):
        for s in ax.spines.values():
            s.set_visible(True)
    fig.tight_layout()
    return fig


def saturation_plot(sat, wide):
    """Left: resolution-invariant WT colony footprint (fraction of well cell
    filled) vs time -- does growth level off? Right: the sick strain's relative
    fitness vs time -- does compression worsen? Raw pixel size is NOT used: the
    t72 camera has ~19x the pixel density of t44/t50."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for i, plate in enumerate(("P1", "P2")):
        s = sat[sat["plate"] == plate].sort_values("hours")
        col = PLOT_PALETTE[0] if plate == "P1" else PLOT_PALETTE[4]
        lab = f"{plate} ({s['volume_nl'].iloc[0]:g} nL)"
        axl.plot(
            s["hours"],
            s["median_fill"],
            "o-",
            color=col,
            ms=4,
            markeredgecolor="black",
            markeredgewidth=0.4,
            lw=1.0,
            label=lab,
        )
        axl.fill_between(
            s["hours"], s["p10_fill"], s["p90_fill"], color=col, alpha=0.12, lw=0
        )
    axl.set_xlabel("growth time (h)")
    axl.set_ylabel(
        "colony footprint (fraction of well cell): plate median, band P10–P90"
    )
    axl.set_title("colony growth, resolution-invariant", fontsize=6)
    axl.legend(frameon=False, fontsize=5)

    for plate in ("P1", "P2"):
        cols = [g for g in GROUPS if g.startswith(plate)]
        hrs = [sat[sat["group"] == g]["hours"].iloc[0] for g in cols]
        col = PLOT_PALETTE[0] if plate == "P1" else PLOT_PALETTE[4]
        axr.plot(
            hrs,
            [wide.loc["YJR060W", g] for g in cols],
            "o-",
            color=col,
            ms=4,
            markeredgecolor="black",
            markeredgewidth=0.4,
            lw=1.0,
            label=f"{plate}",
        )
    axr.axhline(0.59, color=PLOT_PALETTE[1], ls="--", lw=0.8, label="Costanzo 0.59")
    axr.set_xlabel("growth time (h)")
    axr.set_ylabel("YJR060W fitness (vs WT)")
    axr.set_title("sick-strain signal vs time", fontsize=6)
    axr.set_ylim(0.5, 1.0)
    axr.legend(frameon=False, fontsize=5)
    for ax in (axl, axr):
        for s in ax.spines.values():
            s.set_visible(True)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
