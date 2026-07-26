# experiments/019-echo-crispr-array/scripts/run3_48h_3rand.py
# [[experiments.019-echo-crispr-array.scripts.run3_48h_3rand]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run3_48h_3rand
"""Run-3: three RE-RANDOMIZED replicate plates of the 12-panel single-KO assay at
5 nL / 48 h -- the proper batch-effect design.

Run 2 estimated the across-plate SE from 6 captures that were really 2 plates re-imaged
at 3 timepoints (correlated), so the "replicates" were not independent. Run 3 fixes that:
P1/P2/P3 are three SEPARATE plates, each with an INDEPENDENT randomized strain layout
(`cherrypick_Plate{1,2,3}`), all imaged 48 h from the start of incubation. Re-randomizing
the position of every strain per plate is what lets the bootstrap-across-plates average out
plate/position batch bias and quantify an HONEST between-plate SE (the batch effect).

Pipeline per plate: full-res crop -> Cellpose-SAM instance seg (homography grid + faint-
colony recovery, `torchcell.sga.cellpose_seg`) -> resolve the 4-way orientation against
THIS plate's layout -> normalize to on-plate BY4741 -> score. Then, across the 3 plates:
bootstrap each strain's mean fitness (plate = the resampling unit) for the SE, and compare
to Costanzo 2016 SMF (correlation + measurement-error SD).

Run from repo root on a GPU node:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run3_48h_3rand.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

from torchcell.sga import (
    CellposeSegConfig,
    NormalizationConfig,
    load_cellpose_model,
    normalize_plate,
    quantify_plate_image_cellpose,
    read_echo_picklist,
    score_plate,
    score_table,
)
from torchcell.sga.viz import label_plate_overlay
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_volume_timepoints as r2  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
DATA_DIR = osp.join(EXP_DIR, "data", "run3_2026-07-23")
QUANT_DIR = osp.join(EXP_DIR, "quant", "run3_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "run3")
for d in (RESULTS_DIR, QUANT_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

N_ROWS, N_COLS = r2.N_ROWS, r2.N_COLS
WT_NAME = "BY4741"
N_BOOT = 4000
SEED = 1234
C_ORANGE, C_RED, C_GRAY = PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[5]

# three re-randomized replicate plates, 5 nL / OD1 / 48 h
CONDITIONS = [
    dict(
        group="P1",
        image=osp.join(DATA_DIR, "P1_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA_DIR, "cherrypick_Plate1_384_5nL.csv"),
    ),
    dict(
        group="P2",
        image=osp.join(DATA_DIR, "P2_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA_DIR, "cherrypick_Plate2_384_5nL.csv"),
    ),
    dict(
        group="P3",
        image=osp.join(DATA_DIR, "P3_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA_DIR, "cherrypick_Plate3_384_5nL.csv"),
    ),
]


def preprocess_fullres(path: str) -> str:
    """Crop to the plate at full resolution (same bright-plate detector as run 2)."""
    out = osp.join(QUANT_DIR, osp.splitext(osp.basename(path))[0] + "_crop.png")
    if osp.exists(out):
        return out
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
    im.crop((c0, r0, c1, r1)).save(out)
    return out


OPS = ["identity", "rot180", "flip_v", "flip_h"]


def _wt_cv(df: pd.DataFrame, wt_name: str) -> float:
    wt = df.loc[(df["strain"] == wt_name) & (~df["is_missing"]), "size"].to_numpy(float)
    return float(np.std(wt) / np.mean(wt)) if wt.size and np.mean(wt) else np.nan


def _strain_fitness(grid, layout, op, cfg):
    """Per-strain median normalized fitness for a candidate orientation (WT/blank
    excluded). Used to break an ambiguous orientation against the replicate consensus.
    """
    m = r2.apply_orientation(grid, op).merge(layout, on=["row", "col"], how="inner")
    df = normalize_plate(m, cfg)
    used = df[
        ~df["is_missing"] & ~df["is_flagged"] & ~df["is_blank"] & ~df["is_jackknife"]
    ]
    return used.groupby("strain")["norm"].median()


def _resolve_op(grid, layout, cfg, group, consensus):
    """Orientation for one plate. Prefer the strain-structure resolver; if it flags the
    plate as ambiguous, break the tie by which orientation's per-strain fitness best
    correlates with the CONSENSUS of the confidently-resolved sibling plates -- the same
    12 strains, so the biological orientation is the one that agrees with the replicates
    (not with the reference, so this stays non-circular).
    """
    try:
        op, be, _diag = r2.resolve_and_check(grid, layout, cfg, group)
        return op, be, "strain-H"
    except AssertionError:
        if consensus is None or consensus.empty:
            raise
        best_op, best_r = "identity", -2.0
        for op in OPS:
            sf = _strain_fitness(grid, layout, op, cfg)
            common = sf.index.intersection(consensus.index)
            if len(common) < 4:
                continue
            r = float(np.corrcoef(sf[common], consensus[common])[0, 1])
            if r > best_r:
                best_r, best_op = r, op
        m = r2.apply_orientation(grid, best_op).merge(
            layout, on=["row", "col"], how="inner"
        )
        is_blank = m["strain"] == cfg.blank_name
        be = int((is_blank & (m["size"] <= cfg.min_size)).sum())
        print(
            f"    {group}: strain-H AMBIGUOUS -> resolved to '{best_op}' by "
            f"cross-replicate corr r={best_r:.2f} vs sibling consensus"
        )
        return best_op, be, f"cross-replicate(r={best_r:.2f})"


def detect_and_score() -> pd.DataFrame:
    """Detect + register + score each plate; return per-plate per-strain fitness.

    Two passes: (1) Cellpose-detect every plate and resolve the orientations that are
    confident; (2) build the replicate consensus and resolve any ambiguous plate against
    it, then normalize + score all.
    """
    cfg = NormalizationConfig()
    seg_cfg = CellposeSegConfig(
        n_rows=N_ROWS,
        n_cols=N_COLS,
        contrast="clahe",
        clahe_clip=0.02,
        cellprob_threshold=-4.0,
        node_tol=0.60,
        edge_margin_frac=0.70,
        multi_min_frac=0.35,
    )
    print("[0] loading Cellpose-SAM (cpsam) on GPU ...")
    model = load_cellpose_model(gpu=True)

    print("[1a] crop -> Cellpose seg (all plates)")
    plates, overlays = {}, {}
    for cond in CONDITIONS:
        g = cond["group"]
        proc = preprocess_fullres(cond["image"])
        res = quantify_plate_image_cellpose(
            proc, model, seg_cfg,
            overlay_path=osp.join(IMG_DIR, f"run3_overlay_{g}.png"),
            return_masks=True,
        )
        np.save(osp.join(QUANT_DIR, f"run3_masks_{g}.npy"), res.masks)
        overlays[g] = (osp.join(IMG_DIR, f"run3_overlay_{g}.png"), res.nodes)
        res.table.to_csv(osp.join(QUANT_DIR, f"run3_grid_{g}.csv"), index=False)
        layout = read_echo_picklist(cond["picklist"])
        plates[g] = dict(grid=res.table, layout=layout)

    print("[1b] resolve orientations (confident first, then ambiguous vs consensus)")
    resolved, confident, pending = {}, {}, []
    for g, p in plates.items():
        try:
            op, be, _d = r2.resolve_and_check(p["grid"], p["layout"], cfg, g)
            resolved[g] = op
            confident[g] = True
            print(f"    {g}: op={op} (strain-H confident) blanks_empty={be}/6")
        except AssertionError:
            pending.append(g)
            confident[g] = False
    consensus = None
    if resolved:
        sfs = [_strain_fitness(plates[g]["grid"], plates[g]["layout"], resolved[g], cfg)
               for g in resolved]
        consensus = pd.concat(sfs, axis=1).mean(axis=1)
    for g in pending:
        op, _be, _how = _resolve_op(plates[g]["grid"], plates[g]["layout"], cfg, g,
                                    consensus)
        resolved[g] = op

    # burn the 384-well address (rows A-P, cols 1-24, all four sides) + grid nodes onto each
    # overlay, using the RESOLVED orientation so the headers are plate wells not image indices
    for g, (path, nodes) in overlays.items():
        label_plate_overlay(path, nodes, resolved[g])

    # QC gate: a plate is usable only if its orientation is confidently resolved AND its
    # on-plate WT reference is consistent (WT_CV below WT_CV_MAX). A plate that is
    # orientation-ambiguous with a noisy WT is not reliably quantifiable -- excluded from
    # the batch-effect analysis (its overlay is still written for transparency).
    WT_CV_MAX = 0.18
    rows, qc = [], []
    print("[1c] normalize + score; QC gate")
    for cond in CONDITIONS:
        g = cond["group"]
        op = resolved[g]
        grid, layout = plates[g]["grid"], plates[g]["layout"]
        merged = r2.apply_orientation(grid, op).merge(
            layout, on=["row", "col"], how="inner"
        )
        df = normalize_plate(merged, cfg)
        rep = score_plate(df, cfg, plate_id=g)
        occ = int((df["size"] > 0).sum())
        mflag = int(df["flags"].fillna("").str.contains("M").sum())
        be = int(((df["strain"] == cfg.blank_name) & df["is_missing"]).sum())
        wtcv = _wt_cv(df, cfg.wt_name)
        qc_pass = bool(confident[g]) and wtcv < WT_CV_MAX
        qc.append(dict(group=g, op=op, orientation_confident=confident[g],
                       wt_cv=round(wtcv, 3), occupied=occ, blanks_empty=be,
                       qc_pass=qc_pass))
        print(
            f"    {g}: op={op} occupied {occ}/384 M={mflag} blanks_empty={be}/6 "
            f"WT_CV={wtcv:.3f} -> QC {'PASS' if qc_pass else 'FAIL'}"
        )
        if not qc_pass:
            continue
        for s in rep.strains:
            if s.strain == cfg.blank_name or s.relative_fitness is None:
                continue
            rows.append(
                dict(
                    group=g,
                    strain=s.strain,
                    fitness=s.relative_fitness,
                    fitness_sd=s.fitness_sd,
                    fitness_se=(
                        s.fitness_sd / (s.n_used**0.5)
                        if s.fitness_sd is not None and s.n_used
                        else None
                    ),
                    n_used=s.n_used,
                    n_total=s.n_total,
                )
            )
        score_table(rep).sort_values("relative_fitness").to_csv(
            osp.join(RESULTS_DIR, f"run3_strain_scores_{g}.csv"), index=False
        )
    fit = pd.DataFrame(rows)
    fit.to_csv(osp.join(RESULTS_DIR, "run3_fitness_by_condition.csv"), index=False)
    qc_df = pd.DataFrame(qc)
    qc_df.to_csv(osp.join(RESULTS_DIR, "run3_plate_qc.csv"), index=False)
    return fit, qc_df


def bootstrap_across_plates(fit: pd.DataFrame) -> pd.DataFrame:
    """Batch effect: resample the 3 plate-level fitnesses per strain (plate = unit)."""
    rng = np.random.default_rng(SEED)
    out = []
    for strain, d in fit[fit["strain"] != WT_NAME].groupby("strain"):
        vals = d["fitness"].to_numpy(float)
        if len(vals) < 2:
            continue
        idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        bm = vals[idx].mean(axis=1)
        out.append(
            dict(
                strain=strain,
                n_plates=len(vals),
                boot_fitness=float(bm.mean()),
                boot_se=float(bm.std(ddof=1)),
                across_plate_sd=float(vals.std(ddof=1)),
                mean_within_plate_sd=float(d["fitness_sd"].mean()),
                # within-plate SE from the ACTUAL replicate count per plate
                # (fitness_sd / sqrt(n_used)), averaged over plates -- n_used varies
                # 20-30 as wells are lost to failed transfers / M-N-C invalidation, so a
                # single hardcoded n misstates this comparison.
                mean_within_plate_se=float(d["fitness_se"].mean()),
                mean_n_used=float(d["n_used"].mean()),
            )
        )
    res = pd.DataFrame(out)
    res.to_csv(osp.join(RESULTS_DIR, "run3_strain_bootstrap.csv"), index=False)
    return res


def compare_reference(boot: pd.DataFrame, n_plates: int) -> pd.DataFrame:
    ref = pd.read_csv(osp.join(RESULTS_DIR, "reference_smf_12panel.csv"))
    m = boot.merge(ref, on="strain", how="left")
    m.to_csv(osp.join(RESULTS_DIR, "run3_vs_reference.csv"), index=False)
    cst = m.dropna(subset=["costanzo_smf"])
    pr, pp = pearsonr(cst["costanzo_smf"], cst["boot_fitness"])
    sr, sp = spearmanr(cst["costanzo_smf"], cst["boot_fitness"])
    print(
        f"\n[3] vs Costanzo (n={len(cst)}): Pearson r={pr:.3f} (p={pp:.3f}) "
        f"Spearman rho={sr:.3f} (p={sp:.3f})"
    )
    print(
        f"    mean bootstrap-across-{n_plates}-plates SE = {boot['boot_se'].mean():.3f} "
        f"(across-plate SD {boot['across_plate_sd'].mean():.3f})"
    )
    return m


def _rc(ax: plt.Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)


def fig_correlation(m: pd.DataFrame, used: list) -> None:
    npl = len(used)
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(88)))
    lim = (0.4, 1.3)
    ax.plot(lim, lim, ls="--", lw=0.5, color=C_GRAY, zorder=1)
    cst = m.dropna(subset=["costanzo_smf"])
    ax.errorbar(
        cst["costanzo_smf"], cst["boot_fitness"], yerr=cst["boot_se"],
        xerr=cst["costanzo_se"], fmt="o", ms=4, mfc=C_ORANGE, mec="black", mew=0.4,
        ecolor="black", elinewidth=0.4, capsize=1.2, zorder=3, label="Costanzo 2016",
    )
    for _, r in cst.iterrows():
        ax.annotate(r["strain"], (r["costanzo_smf"], r["boot_fitness"]), fontsize=3.5,
                    xytext=(3, 2), textcoords="offset points")
    pr, pp = pearsonr(cst["costanzo_smf"], cst["boot_fitness"])
    sr, sp = spearmanr(cst["costanzo_smf"], cst["boot_fitness"])
    ax.text(0.03, 0.97, f"Costanzo (n={len(cst)}): Pearson r={pr:.2f} (p={pp:.3f}), "
            f"Spearman ρ={sr:.2f} (p={sp:.3f})", transform=ax.transAxes, fontsize=4.5,
            va="top", ha="left")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("published single-mutant fitness (Costanzo 2016)")
    ax.set_ylabel(f"CRISPR assay fitness (bootstrap mean, {npl} re-randomized plates)")
    ax.set_title(f"Run 3: assay fitness vs published SMF (48 h, {npl} of 3 plates QC-pass)",
                 fontsize=6)
    ax.legend(fontsize=4.5, loc="lower right", frameon=False)
    _rc(ax)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run3_fitness_correlation_reference.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run3_fitness_correlation_reference.svg"))
    plt.close(fig)


def fig_se(boot: pd.DataFrame, used: list) -> None:
    npl = len(used)
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    d = boot.sort_values("boot_se", ascending=False)
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(72)))
    ax.bar(x - w / 2, d["boot_se"], w, color=C_ORANGE, edgecolor="black", linewidth=0.4,
           label=f"bootstrap-across-plates SE (n={npl} plates)")
    nbar = d["mean_n_used"].mean()
    ax.bar(x + w / 2, d["mean_within_plate_se"], w, color=C_RED,
           edgecolor="black", linewidth=0.4,
           label=f"within-plate SE (colony SD/√n_used, mean n={nbar:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(d["strain"], rotation=90, fontsize=4)
    ax.set_ylim(0, float(max(d["boot_se"].max(),
                d["mean_within_plate_se"].max())) * 1.28)
    ax.set_ylabel("standard error of the mean fitness")
    ax.legend(fontsize=4.5, loc="upper right", frameon=False)
    _rc(ax)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run3_se_batch_effect.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run3_se_batch_effect.svg"))
    plt.close(fig)


def main() -> None:
    fit, qc = detect_and_score()
    used = qc.loc[qc["qc_pass"], "group"].tolist()
    excl = qc.loc[~qc["qc_pass"], "group"].tolist()
    print(f"\n[2] batch effect: bootstrap each strain across QC-PASS plates {used}")
    if excl:
        print(f"    EXCLUDED (QC fail): {excl}")
    boot = bootstrap_across_plates(fit)
    m = compare_reference(boot, n_plates=len(used))
    fig_correlation(m, used)
    fig_se(boot, used)
    print(f"\nwrote results -> {RESULTS_DIR}")
    print(f"wrote figures + overlays -> {IMG_DIR}")


if __name__ == "__main__":
    main()
