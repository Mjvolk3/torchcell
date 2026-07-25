# experiments/019-echo-crispr-array/scripts/plate_artifacts.py
# [[experiments.019-echo-crispr-array.scripts.plate_artifacts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/plate_artifacts
"""Standard plate-artifact builder for the CRISPR array assay.

Reprocesses a collection of plate captures with the CURRENT pipeline (fresh Cellpose from
the sha256-pinned source images -- deterministic, so it reproduces the committed grids) and
bundles the result into one self-contained artifact per collection:

  * an .xlsx workbook -- README/provenance (image sha256 + git commit + seg-config),
    QC summary, one colony table per plate, and strain scores;
  * an images/ folder of LABELLED detection overlays -- colony boundaries plus the fitted
    grid nodes (cyan) and row/col headers (magenta), so every colony is addressable by
    (row, col) and any mis-registration is visible at a glance.

Non-destructive: writes ONLY under experiments/019-echo-crispr-array/artifacts/, never the
committed paper figures/CSVs -- so it doubles as the "does the current pipeline change an
earlier collection?" check (diff artifacts/run2 vs the committed run2 outputs).

Collections: run2 (P1 2.5 nL / P2 5 nL, OD1, t44/t50/t72) and run3 (P1/P2/P3, 48 h).

Run from repo root on a GPU node:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/plate_artifacts.py [run2|run3|all]
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from torchcell.sga import (
    CellposeSegConfig,
    NormalizationConfig,
    load_cellpose_model,
    normalize_plate,
    quantify_plate_image_cellpose,
    read_echo_picklist,
    score_plate,
)

load_dotenv()
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
SCRIPTS = osp.join(EXP_DIR, "scripts")
ART_ROOT = osp.join(EXP_DIR, "artifacts")
sys.path.insert(0, SCRIPTS)
import run2_cellpose_segmentation as r2cp  # noqa: E402  (preprocess_fullres)
import run2_volume_timepoints as r2  # noqa: E402
import run3_48h_3rand as r3  # noqa: E402

WT_CV_MAX = 0.18
SEG_CFG = dict(
    n_rows=16, n_cols=24, contrast="clahe", clahe_clip=0.02, cellprob_threshold=-4.0,
    node_tol=0.60, edge_margin_frac=0.70, multi_min_frac=0.5,
)

COLLECTIONS = {
    "run2": dict(conditions=r2.CONDITIONS, preprocess=r2cp.preprocess_fullres),
    "run3": dict(conditions=r3.CONDITIONS, preprocess=r3.preprocess_fullres),
}

COLONY_COLS = [
    "row", "col", "strain", "size", "circularity", "flags", "detector",
    "norm", "is_missing", "is_flagged", "is_blank", "is_jackknife", "cx", "cy",
]
SCORE_COLS = [
    "group", "strain", "n_total", "n_used", "median_norm", "sd_norm",
    "relative_fitness", "fitness_sd", "log2_fitness", "pvalue",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", EXP_DIR, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()


def _wt_cv(df: pd.DataFrame, wt_name: str) -> float:
    wt = df.loc[(df["strain"] == wt_name) & (~df["is_missing"]), "size"].to_numpy(float)
    return float(np.std(wt) / np.mean(wt)) if wt.size and np.mean(wt) else np.nan


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if osp.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _plate_labels(op: str, n_rows: int, n_cols: int) -> tuple[list[str], list[str]]:
    """Per-image-index PLATE labels (rows A-P, cols 1-24) under the resolved orientation.

    The overlay is drawn in IMAGE order, but the well a colony belongs to is its PLATE
    address -- ``r2.apply_orientation`` maps image (row, col) -> plate (row, col). All four
    ops keep rows and columns separable (each either preserved or reversed), so the mapping
    collapses to a per-axis label list. Without this the headers would silently show image
    coordinates on any plate that resolves to rot180/flip_v/flip_h.
    """
    rows_rev = op in ("rot180", "flip_v")  # nr = n_rows + 1 - r
    cols_rev = op in ("rot180", "flip_h")  # nc = n_cols + 1 - c
    rows = [
        chr(ord("A") + (n_rows - 1 - ri if rows_rev else ri)) for ri in range(n_rows)
    ]
    cols = [str(n_cols - ci if cols_rev else ci + 1) for ci in range(n_cols)]
    return rows, cols


def _label_overlay(overlay_path: str, nodes: np.ndarray, op: str = "identity") -> None:
    """Draw the fitted grid nodes (cyan crosses) + 384-well headers (black) onto the
    boundary overlay in place. Standard 384-well addressing -- rows A-P, columns 1-24 --
    repeated on ALL FOUR sides (columns top+bottom, rows left+right) so a well can be read
    off from any side. Headers are PLATE addresses under the resolved orientation ``op``
    (see ``_plate_labels``), not raw image indices. A mis-fit node -- sitting off its
    colony -- is visible directly.
    """
    im = Image.open(overlay_path).convert("RGB")
    d = ImageDraw.Draw(im)
    n_rows, n_cols, _ = nodes.shape
    row_lab, col_lab = _plate_labels(op, n_rows, n_cols)
    w, h = im.width, im.height
    fs = max(48, int(w / 26))  # large headers, readable when the plate is zoomed out
    fnt = _font(fs)
    blk = (0, 0, 0)  # black headers -- high contrast on the cream agar margin
    for ri in range(n_rows):
        for ci in range(n_cols):
            y, x = float(nodes[ri, ci, 0]), float(nodes[ri, ci, 1])
            d.line([(x - 5, y), (x + 5, y)], fill=(0, 255, 255), width=1)
            d.line([(x, y - 5), (x, y + 5)], fill=(0, 255, 255), width=1)

    def ctext(cx: float, cy: float, s: str) -> None:
        tw = d.textlength(s, font=fnt)
        d.text((cx - tw / 2, cy - fs / 2), s, fill=blk, font=fnt)

    # columns 1..24 on TOP and BOTTOM
    top_y = min(max(fs, float(nodes[0, :, 0].mean()) - 1.9 * fs), h - fs)
    bot_y = min(float(nodes[-1, :, 0].mean()) + 1.9 * fs, h - fs)
    for ci in range(n_cols):
        ctext(float(nodes[0, ci, 1]), top_y, col_lab[ci])
        ctext(float(nodes[-1, ci, 1]), bot_y, col_lab[ci])
    # rows A..P on LEFT and RIGHT
    left_x = max(fs, float(nodes[:, 0, 1].mean()) - 1.9 * fs)
    right_x = min(float(nodes[:, -1, 1].mean()) + 1.9 * fs, w - fs)
    for ri in range(n_rows):
        ctext(left_x, float(nodes[ri, 0, 0]), row_lab[ri])
        ctext(right_x, float(nodes[ri, -1, 0]), row_lab[ri])
    im.save(overlay_path)


def _resolve_orientation(grid, layout, cfg, g):
    """(op, blanks_empty, confident) via the run-2 strain-structure resolver; a plate it
    flags ambiguous falls back to 'identity', not-confident (QC-excluded).
    """
    try:
        op, be, _diag = r2.resolve_and_check(grid, layout, cfg, g)
        return op, be, True
    except AssertionError:
        m = r2.apply_orientation(grid, "identity").merge(
            layout, on=["row", "col"], how="inner"
        )
        be = int(((m["strain"] == cfg.blank_name) & (m["size"] <= cfg.min_size)).sum())
        return "identity", be, False


def process_collection(name: str, model_holder: dict) -> None:
    spec = COLLECTIONS[name]
    out = osp.join(ART_ROOT, name)
    img_dir = osp.join(out, "images")
    for d in (out, img_dir):
        os.makedirs(d, exist_ok=True)

    cfg = NormalizationConfig()
    seg_cfg = CellposeSegConfig(**SEG_CFG)
    prov, qc, colony_sheets, score_rows = [], [], {}, []

    for cond in spec["conditions"]:
        g = cond["group"]
        src = cond["image"]
        proc = spec["preprocess"](src)
        if model_holder["model"] is None:
            print("    loading Cellpose-SAM (cpsam) on GPU ...")
            model_holder["model"] = load_cellpose_model(gpu=True)
        overlay = osp.join(img_dir, f"{name}_overlay_{g}.png")
        res = quantify_plate_image_cellpose(
            proc, model_holder["model"], seg_cfg, overlay_path=overlay
        )
        grid = res.table
        layout = read_echo_picklist(cond["picklist"])
        op, be, confident = _resolve_orientation(grid, layout, cfg, g)
        # label AFTER the orientation is resolved so the headers carry PLATE addresses
        _label_overlay(overlay, res.nodes, op)
        merged = r2.apply_orientation(grid, op).merge(
            layout, on=["row", "col"], how="inner"
        )
        df = normalize_plate(merged, cfg)
        rep = score_plate(df, cfg, plate_id=g)

        occ = int((df["size"] > 0).sum())
        n_multi = int(df["flags"].fillna("").astype(str).str.contains("M").sum())
        wtcv = _wt_cv(df, cfg.wt_name)
        qc_pass = bool(confident) and wtcv < WT_CV_MAX
        qc.append(dict(
            group=g, orientation=op, orientation_confident=confident,
            occupied=occ, blanks_empty=be, multi_M=n_multi,
            wt_cv=round(wtcv, 3), qc_pass=qc_pass,
        ))
        prov.append(dict(
            group=g, source_image=osp.relpath(src, EXP_DIR),
            sha256=_sha256(src), overlay=f"images/{name}_overlay_{g}.png",
        ))
        cols = [c for c in COLONY_COLS if c in df.columns]
        colony_sheets[g] = df.sort_values(["row", "col"])[cols].reset_index(drop=True)
        for s in rep.strains:
            score_rows.append(dict(
                group=g, strain=s.strain, n_total=s.n_total, n_used=s.n_used,
                median_norm=s.median_norm, sd_norm=s.sd_norm,
                relative_fitness=s.relative_fitness, fitness_sd=s.fitness_sd,
                log2_fitness=s.log2_fitness, pvalue=s.pvalue,
            ))
        print(f"    {g}: op={op} conf={confident} occ={occ}/384 M={n_multi} "
              f"WT_CV={wtcv:.3f} -> QC {'PASS' if qc_pass else 'FAIL'}")

    xlsx = osp.join(out, f"{name}_plate_artifacts.xlsx")
    readme = pd.DataFrame([
        ("collection", name),
        ("generated_by", "experiments/019-echo-crispr-array/scripts/plate_artifacts.py"),
        ("git_commit", _git_commit()),
        ("generated_at_utc", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")),
        ("seg_config", str(SEG_CFG)),
        ("normalization", "NormalizationConfig() defaults (Baryshnikova row/col+spatial)"),
        ("n_captures", str(len(spec["conditions"]))),
        ("qc_wt_cv_max", str(WT_CV_MAX)),
        ("overlays", "labelled: colony boundaries + grid nodes (cyan) + A-P/1-24 headers (black, 4 sides)"),
        ("colour_legend", "green=accepted; red=multi(M) rejected; pink=collider of a multi; "
         "orange=neighbour(N) rejected; blue=grid-recovered faint colony; purple=non-circular(C) rejected"),
    ], columns=["key", "value"])
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(prov).to_excel(xw, sheet_name="provenance", index=False)
        pd.DataFrame(qc).to_excel(xw, sheet_name="QC_summary", index=False)
        pd.DataFrame(score_rows)[SCORE_COLS].to_excel(
            xw, sheet_name="strain_scores", index=False
        )
        for g, sheet in colony_sheets.items():
            sheet.to_excel(xw, sheet_name=g[:31], index=False)

    print(f"  artifact -> {out}")
    print(f"    xlsx    -> {xlsx}")
    print(f"    images  -> {img_dir} ({len(spec['conditions'])} labelled overlays)")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(COLLECTIONS) if which == "all" else [which]
    model_holder = {"model": None}
    for name in names:
        print(f"[{name}] building plate-artifact")
        process_collection(name, model_holder)


if __name__ == "__main__":
    main()
