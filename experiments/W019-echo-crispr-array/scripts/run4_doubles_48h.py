# experiments/W019-echo-crispr-array/scripts/run4_doubles_48h.py
# [[experiments.W019-echo-crispr-array.scripts.run4_doubles_48h]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/run4_doubles_48h
"""Run-4: the first plating carrying DOUBLE knockouts, so the first data from which a
digenic interaction can be measured rather than only single-mutant fitness.

Design (see data/run4_doubles_2026-08-06/PROVENANCE.md): 26 strains -- WT + 12 singles +
13 doubles -- on three INDEPENDENTLY randomized plates, WT at 28 wells and every mutant at
14 wells per plate, 378 transfers + 6 blanks. Crucially **both constituent singles of every
double sit on the same plate**, so the interaction

    eps_ab = f_ab - f_a * f_b            (multiplicative null, standard SGA)

is computed entirely WITHIN a plate -- no cross-plate normalization enters the interaction
term, only the across-plate bootstrap that puts an error bar on it.

Pipeline per plate is run-3's, unchanged: full-res crop -> Cellpose-SAM instance seg
(homography grid + faint-colony recovery) -> resolve the 4-way orientation against THIS
plate's layout -> normalize to on-plate WT -> score. Then across the 3 plates, bootstrap
each strain's fitness and each double's eps, with the plate as the resampling unit.

Run from repo root on a GPU node:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/run4_doubles_48h.py
"""

from __future__ import annotations

import os
import os.path as osp
import re
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
    score_plate,
    score_table,
)
from torchcell.sga.io import well_to_rowcol
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
REPO = osp.dirname(osp.dirname(EXP_DIR))
RESULTS_DIR = osp.join(EXP_DIR, "results")
DATA_DIR = osp.join(EXP_DIR, "data", "run4_doubles_2026-08-06")
QUANT_DIR = osp.join(EXP_DIR, "quant", "run4_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "W019-echo-crispr-array", "run4")
for d in (RESULTS_DIR, QUANT_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

# Published reference for the doubles: Costanzo DMF + eps + p for all 45 within-10 pairs.
DOUBLES_REF = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv"
)
# Published reference for the singles (Costanzo/Kuzmin SMF), built by build_reference_smf.py.
SINGLES_REF = osp.join(RESULTS_DIR, "reference_smf_12panel.csv")

N_ROWS, N_COLS = r2.N_ROWS, r2.N_COLS
WT_NAME = "WT"
BLANK_NAME = "Blank_media"
N_BOOT = 4000
SEED = 1234
WT_CV_MAX = 0.18
C_ORANGE, C_RED, C_PURPLE, C_GRAY = (
    PLOT_PALETTE[0],
    PLOT_PALETTE[1],
    PLOT_PALETTE[2],
    PLOT_PALETTE[5],
)

PLATES = ["P1", "P2", "P3"]


def _orf(cell: str) -> str:
    """'YDR057W (YOS9)' -> 'YDR057W'; also tolerates the stray trailing spaces in the
    collaborator's strain list ('YER079W '). The systematic ORF is the join key against
    every published reference, so the common name is dropped here, not carried along.
    """
    return re.split(r"\s|\(", cell.strip())[0].strip()


def read_strain_list() -> dict[str, dict]:
    """Map the picklist's sample ids (WT, s1..s12, d1..d13) to systematic ORFs.

    Returns id -> {kind, genes, name} where `name` is the canonical strain label used
    everywhere downstream: the ORF for a single, 'ORF_A+ORF_B' with the two ORFs SORTED for
    a double (matching how gene1/gene2 are ordered in the published reference table, so the
    join needs no orientation handling).
    """
    path = osp.join(DATA_DIR, "Single-and-Double-KO-Strains-List-Order.csv")
    raw = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, r in raw.iterrows():
        sid = str(r["#"]).strip()
        if sid == "WT":
            out[sid] = dict(kind="wt", genes=(), name=WT_NAME)
            continue
        g1 = _orf(str(r["KO1"]))
        g2 = str(r["KO2"])
        if pd.isna(r["KO2"]) or not g2.strip():
            out[sid] = dict(kind="single", genes=(g1,), name=g1)
        else:
            pair = tuple(sorted((g1, _orf(g2))))
            out[sid] = dict(kind="double", genes=pair, name=f"{pair[0]}+{pair[1]}")
    return out


def read_run4_picklist(path: str, strains: dict[str, dict]) -> pd.DataFrame:
    """Read the collaborator's Echo cherry-pick export into the layout frame the rest of
    the pipeline expects (row, col, strain, volume_nl, well).

    Two things differ from run 3's in-repo picklists, both properties of THIS export rather
    than options:

    * the sample column is `Sample`, not `Sample Name`, so torchcell.sga.io.read_echo_picklist
      rejects it. Normalized here, at the experiment level, rather than by teaching the
      shared reader to accept either spelling.
    * the 6 blank wells are ABSENT from the file rather than listed as no-cell controls
      (run 3 carried explicit `Blank_media` rows). They are reconstructed as the complement
      of the transferred wells, which is what they are: 384 wells minus 378 transfers. The
      blanks matter -- the QC gate and the orientation resolver both read them.
    """
    raw = pd.read_csv(path)
    rc = raw["Destination Well"].map(well_to_rowcol)
    layout = pd.DataFrame(
        {
            "row": [r for r, _ in rc],
            "col": [c for _, c in rc],
            "strain": [strains[s.strip()]["name"] for s in raw["Sample"].astype(str)],
            "volume_nl": pd.to_numeric(raw["Transfer Volume"], errors="coerce").values,
            "well": raw["Destination Well"].astype(str).values,
        }
    )
    filled = set(zip(layout["row"], layout["col"], strict=True))
    blanks = [
        dict(
            row=r,
            col=c,
            strain=BLANK_NAME,
            volume_nl=0.0,
            well=f"{chr(ord('A') + r - 1)}{c}",
        )
        for r in range(1, N_ROWS + 1)
        for c in range(1, N_COLS + 1)
        if (r, c) not in filled
    ]
    return pd.concat([layout, pd.DataFrame(blanks)], ignore_index=True)


def preprocess_fullres(path: str) -> str:
    """Crop to the plate at full resolution (same bright-plate detector as runs 2-3)."""
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


def detect_and_score(strains: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect, register and score all three plates. Returns (per-plate per-strain fitness,
    per-plate QC)."""
    cfg = NormalizationConfig(wt_name=WT_NAME, blank_name=BLANK_NAME)
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
    for g in PLATES:
        proc = preprocess_fullres(osp.join(DATA_DIR, f"{g}_sKO-dKO_OD1-5nL.JPG"))
        res = quantify_plate_image_cellpose(
            proc,
            model,
            seg_cfg,
            overlay_path=osp.join(IMG_DIR, f"run4_overlay_{g}.png"),
            return_masks=True,
        )
        np.save(osp.join(QUANT_DIR, f"run4_masks_{g}.npy"), res.masks)
        overlays[g] = (osp.join(IMG_DIR, f"run4_overlay_{g}.png"), res.nodes)
        res.table.to_csv(osp.join(QUANT_DIR, f"run4_grid_{g}.csv"), index=False)
        layout = read_run4_picklist(
            osp.join(DATA_DIR, f"TC_sKO-dKO_OD1_5nL_{g}.csv"), strains
        )
        plates[g] = dict(grid=res.table, layout=layout)

    print("[1b] resolve orientations")
    resolved = {}
    for g in PLATES:
        op, be, _d = r2.resolve_and_check(plates[g]["grid"], plates[g]["layout"], cfg, g)
        resolved[g] = op
        print(f"    {g}: op={op} (strain-H confident) blanks_empty={be}/6")

    for g, (path, nodes) in overlays.items():
        label_plate_overlay(path, nodes, resolved[g])

    print("[1c] normalize + score; QC gate")
    rows, qc = [], []
    for g in PLATES:
        merged = r2.apply_orientation(plates[g]["grid"], resolved[g]).merge(
            plates[g]["layout"], on=["row", "col"], how="inner"
        )
        df = normalize_plate(merged, cfg)
        rep = score_plate(df, cfg, plate_id=g)
        wt = df.loc[(df["strain"] == WT_NAME) & (~df["is_missing"]), "size"].to_numpy(
            float
        )
        wt_cv = float(np.std(wt) / np.mean(wt))
        occ = int((df["size"] > 0).sum())
        be = int(((df["strain"] == BLANK_NAME) & (df["size"] <= cfg.min_size)).sum())
        ok = wt_cv <= WT_CV_MAX
        print(
            f"    {g}: op={resolved[g]} occupied {occ}/384 blanks_empty={be}/6 "
            f"WT_CV={wt_cv:.3f} -> QC {'PASS' if ok else 'FAIL'}"
        )
        qc.append(
            dict(plate=g, op=resolved[g], occupied=occ, blanks_empty=be, wt_cv=wt_cv,
                 qc_pass=ok)
        )
        for s in rep.strains:
            if s.strain == BLANK_NAME or s.relative_fitness is None:
                continue
            rows.append(
                dict(
                    plate=g,
                    strain=s.strain,
                    fitness=s.relative_fitness,
                    fitness_sd=s.fitness_sd,
                    n_used=s.n_used,
                    n_total=s.n_total,
                )
            )
        score_table(rep).sort_values("relative_fitness").to_csv(
            osp.join(RESULTS_DIR, f"run4_strain_scores_{g}.csv"), index=False
        )

    return pd.DataFrame(rows), pd.DataFrame(qc)


def bootstrap_strain_fitness(scores: pd.DataFrame, plates_ok: list[str]) -> pd.DataFrame:
    """Per-strain fitness, bootstrapping ACROSS plates (the plate is the resampling unit,
    so the SE carries the between-plate batch effect rather than only colony scatter)."""
    rng = np.random.default_rng(SEED)
    wide = scores.pivot_table(index="strain", columns="plate", values="fitness")
    wide = wide[[p for p in plates_ok if p in wide.columns]]
    out = []
    for strain, r in wide.iterrows():
        v = r.dropna().to_numpy(float)
        if v.size == 0:
            continue
        draws = rng.choice(v, size=(N_BOOT, v.size), replace=True).mean(axis=1)
        out.append(
            dict(
                strain=strain,
                n_plates=int(v.size),
                fitness=float(v.mean()),
                boot_se=float(draws.std(ddof=1)),
                across_plate_sd=float(v.std(ddof=1)) if v.size > 1 else np.nan,
            )
        )
    return pd.DataFrame(out).sort_values("strain").reset_index(drop=True)


def compute_interactions(
    scores: pd.DataFrame, strains: dict[str, dict], plates_ok: list[str]
) -> pd.DataFrame:
    """Digenic interaction per double, multiplicative null, computed WITHIN each plate then
    bootstrapped across plates.

        eps_ab = f_ab - f_a * f_b

    Both singles are on the same plate as the double, so no cross-plate normalization enters
    eps; the plate-level eps values are independent replicates of the same quantity, and the
    bootstrap over plates is what gives its error bar. With only 3 plates that bootstrap is
    coarse -- it is reported as an SE, not as a p-value.
    """
    rng = np.random.default_rng(SEED + 1)
    fit = scores.pivot_table(index="strain", columns="plate", values="fitness")
    doubles = [v for v in strains.values() if v["kind"] == "double"]
    out = []
    for d in doubles:
        a, b = d["genes"]
        name = d["name"]
        per_plate = {}
        for p in plates_ok:
            if p not in fit.columns:
                continue
            if not {name, a, b} <= set(fit.index):
                continue
            f_ab, f_a, f_b = fit.at[name, p], fit.at[a, p], fit.at[b, p]
            if np.isnan(f_ab) or np.isnan(f_a) or np.isnan(f_b):
                continue
            per_plate[p] = dict(eps=float(f_ab - f_a * f_b), f_ab=float(f_ab),
                                expected=float(f_a * f_b))
        if not per_plate:
            continue
        e = np.array([v["eps"] for v in per_plate.values()])
        draws = rng.choice(e, size=(N_BOOT, e.size), replace=True).mean(axis=1)
        out.append(
            dict(
                double=name,
                gene1=a,
                gene2=b,
                n_plates=int(e.size),
                dmf=float(np.mean([v["f_ab"] for v in per_plate.values()])),
                expected=float(np.mean([v["expected"] for v in per_plate.values()])),
                eps=float(e.mean()),
                eps_se=float(draws.std(ddof=1)),
                eps_sd_across_plates=float(e.std(ddof=1)) if e.size > 1 else np.nan,
            )
        )
    return pd.DataFrame(out).sort_values("eps").reset_index(drop=True)


def _panel(width_key: str, h_mm: float):
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(h_mm))
    )
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
         "axes.linewidth": 0.5}
    )
    for sp in ax.spines.values():
        sp.set_visible(True)
    return fig, ax


def save(fig, name):
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    plt.close(fig)


def plot_singles_vs_reference(boot: pd.DataFrame) -> pd.DataFrame:
    """Our single-mutant fitness vs published Costanzo SMF."""
    # The reference table carries its own `strain` column (the wet-lab plate label, e.g.
    # "YOS9"), which would collide with ours (the systematic ORF) and silently become
    # strain_x/strain_y. Join on the ORF and keep only the reference's value columns.
    ref = pd.read_csv(SINGLES_REF)
    ref["orf"] = ref["orf"].astype(str)
    ref = ref[["orf", "common_name", "costanzo_smf", "costanzo_se", "kuzmin_smf"]]
    m = boot.merge(ref, left_on="strain", right_on="orf", how="inner").dropna(
        subset=["costanzo_smf"]
    )
    if m.empty:
        return m
    pr, pp = pearsonr(m["costanzo_smf"], m["fitness"])
    sr, sp = spearmanr(m["costanzo_smf"], m["fitness"])
    fig, ax = _panel("half_plus", 80)
    lim = [
        min(m["costanzo_smf"].min(), m["fitness"].min()) - 0.08,
        max(m["costanzo_smf"].max(), m["fitness"].max()) + 0.08,
    ]
    ax.plot(lim, lim, ls="--", lw=0.5, color=C_GRAY, zorder=0)
    ax.errorbar(
        m["costanzo_smf"], m["fitness"], yerr=m["boot_se"], xerr=m["costanzo_se"],
        fmt="o", ms=4, mfc=C_ORANGE, mec="black", mew=0.4, ecolor="black",
        elinewidth=0.4, capsize=0, lw=0,
    )
    for _, r in m.iterrows():
        ax.annotate(r["strain"], (r["costanzo_smf"], r["fitness"]), fontsize=3.5,
                    xytext=(3, -1), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("published single-mutant fitness (Costanzo 2016)")
    ax.set_ylabel("run 4 assay fitness (3-plate bootstrap mean)")
    ax.set_title(
        f"Singles, n={len(m)}: r={pr:.2f} (p={pp:.3f}), rho={sr:.2f} (p={sp:.3f})",
        fontsize=6,
    )
    fig.tight_layout()
    save(fig, "run4_singles_vs_reference")
    return m


def plot_doubles_vs_reference(boot: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    """Our double-mutant fitness and interaction vs the published Costanzo values."""
    ref = pd.read_csv(DOUBLES_REF)
    ref["double"] = [
        "+".join(sorted((a, b))) for a, b in zip(ref["gene1"], ref["gene2"], strict=True)
    ]
    m = inter.merge(
        ref[["double", "DmfCostanzo2016_fitness", "DmfCostanzo2016_std", "se", "eps",
             "p", "significant"]].rename(
            columns={"DmfCostanzo2016_fitness": "ref_dmf",
                     "DmfCostanzo2016_std": "ref_dmf_sd", "se": "ref_dmf_se",
                     "eps": "ref_eps", "p": "ref_p"}
        ),
        on="double",
        how="left",
    )

    # --- DMF panel -------------------------------------------------------------------
    d = m.dropna(subset=["ref_dmf"])
    if not d.empty:
        pr, pp = pearsonr(d["ref_dmf"], d["dmf"])
        sr, sp = spearmanr(d["ref_dmf"], d["dmf"])
        fig, ax = _panel("half_plus", 80)
        lim = [min(d["ref_dmf"].min(), d["dmf"].min()) - 0.08,
               max(d["ref_dmf"].max(), d["dmf"].max()) + 0.08]
        ax.plot(lim, lim, ls="--", lw=0.5, color=C_GRAY, zorder=0)
        ax.errorbar(d["ref_dmf"], d["dmf"], xerr=d["ref_dmf_se"], fmt="o", ms=4,
                    mfc=C_RED, mec="black", mew=0.4, ecolor="black", elinewidth=0.4,
                    capsize=0, lw=0)
        for _, r in d.iterrows():
            ax.annotate(r["double"], (r["ref_dmf"], r["dmf"]), fontsize=3,
                        xytext=(3, -1), textcoords="offset points")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("published double-mutant fitness (Costanzo 2016)")
        ax.set_ylabel("run 4 assay DMF (3-plate bootstrap mean)")
        ax.set_title(f"Doubles, n={len(d)}: r={pr:.2f} (p={pp:.3f}), rho={sr:.2f}",
                     fontsize=6)
        fig.tight_layout()
        save(fig, "run4_doubles_vs_reference")

    # --- interaction panel -----------------------------------------------------------
    e = m.dropna(subset=["ref_eps"])
    if not e.empty:
        pr, pp = pearsonr(e["ref_eps"], e["eps"])
        sr, sp = spearmanr(e["ref_eps"], e["eps"])
        fig, ax = _panel("half_plus", 80)
        lo = min(e["ref_eps"].min(), (e["eps"] - e["eps_se"]).min()) - 0.05
        hi = max(e["ref_eps"].max(), (e["eps"] + e["eps_se"]).max()) + 0.05
        ax.axhline(0, lw=0.4, color=C_GRAY, zorder=0)
        ax.axvline(0, lw=0.4, color=C_GRAY, zorder=0)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.5, color=C_GRAY, zorder=0)
        sig = e["significant"].fillna(False).astype(bool)
        ax.errorbar(e.loc[~sig, "ref_eps"], e.loc[~sig, "eps"],
                    yerr=e.loc[~sig, "eps_se"], fmt="o", ms=4, mfc=C_GRAY, mec="black",
                    mew=0.4, ecolor="black", elinewidth=0.4, capsize=0, lw=0,
                    label="not significant in Costanzo")
        ax.errorbar(e.loc[sig, "ref_eps"], e.loc[sig, "eps"], yerr=e.loc[sig, "eps_se"],
                    fmt="o", ms=5, mfc=C_PURPLE, mec="black", mew=0.5, ecolor="black",
                    elinewidth=0.5, capsize=0, lw=0,
                    label="significant in Costanzo")
        for _, r in e.iterrows():
            ax.annotate(r["double"], (r["ref_eps"], r["eps"]), fontsize=3,
                        xytext=(3, -1), textcoords="offset points")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"published interaction $\varepsilon$ (Costanzo 2016)")
        ax.set_ylabel(r"run 4 assay $\varepsilon = f_{ab} - f_a f_b$")
        ax.set_title(f"Interactions, n={len(e)}: r={pr:.2f} (p={pp:.3f}), rho={sr:.2f}",
                     fontsize=6)
        ax.legend(fontsize=4.5, loc="upper left", frameon=False)
        fig.tight_layout()
        save(fig, "run4_interactions_vs_reference")
    return m


def main() -> None:
    strains = read_strain_list()
    n_s = sum(1 for v in strains.values() if v["kind"] == "single")
    n_d = sum(1 for v in strains.values() if v["kind"] == "double")
    print(f"strain list: WT + {n_s} singles + {n_d} doubles")

    scores, qc = detect_and_score(strains)
    qc.to_csv(osp.join(RESULTS_DIR, "run4_plate_qc.csv"), index=False)
    scores.to_csv(osp.join(RESULTS_DIR, "run4_strain_scores_by_plate.csv"), index=False)
    plates_ok = qc.loc[qc["qc_pass"], "plate"].tolist()
    print(f"\n[2] bootstrap across QC-PASS plates {plates_ok}")

    boot = bootstrap_strain_fitness(scores, plates_ok)
    boot.to_csv(osp.join(RESULTS_DIR, "run4_strain_bootstrap.csv"), index=False)

    inter = compute_interactions(scores, strains, plates_ok)
    inter.to_csv(osp.join(RESULTS_DIR, "run4_interactions.csv"), index=False)
    print(f"    {len(inter)} doubles scored for interaction")

    print("\n[3] compare to published")
    singles_m = plot_singles_vs_reference(boot)
    if not singles_m.empty:
        pr, pp = pearsonr(singles_m["costanzo_smf"], singles_m["fitness"])
        print(f"    singles vs Costanzo SMF (n={len(singles_m)}): r={pr:.3f} p={pp:.4f}")
        singles_m.to_csv(
            osp.join(RESULTS_DIR, "run4_singles_vs_reference.csv"), index=False
        )
    doubles_m = plot_doubles_vs_reference(boot, inter)
    doubles_m.to_csv(osp.join(RESULTS_DIR, "run4_doubles_vs_reference.csv"), index=False)
    e = doubles_m.dropna(subset=["ref_eps"])
    if not e.empty:
        pr, pp = pearsonr(e["ref_eps"], e["eps"])
        sr, sp = spearmanr(e["ref_eps"], e["eps"])
        print(f"    eps vs Costanzo eps (n={len(e)}): r={pr:.3f} p={pp:.4f} "
              f"rho={sr:.3f} p={sp:.4f}")
        sig = e[e["significant"].fillna(False).astype(bool)]
        for _, r in sig.iterrows():
            print(f"      [Costanzo-significant] {r['double']}: "
                  f"ours {r['eps']:+.3f} +/- {r['eps_se']:.3f}  "
                  f"published {r['ref_eps']:+.3f}")

    print(f"\nwrote results -> {RESULTS_DIR}")
    print(f"wrote figures + overlays -> {IMG_DIR}")


if __name__ == "__main__":
    main()
