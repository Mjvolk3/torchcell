# torchcell/sga/cellpose_seg.py
# [[torchcell.sga.cellpose_seg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/cellpose_seg
"""Cellpose-SAM per-colony segmentation for the backlit ECHO CRISPR plates.

A drop-in alternative to the classical per-cell threshold/watershed in
``torchcell.sga.image``. Cellpose returns *instance* masks (one integer id per
colony), so touching/merged/off-grid colonies are separated natively -- which
makes the "invalidate a well that holds a second colony" rule exact rather than a
morphology heuristic.

The lattice fit that assigns colonies to array wells is UNCHANGED: this module
reuses ``image._detect_blobs_backlit`` -> ``_fit_lines`` -> ``_gel_polygon``
exactly as ``quantify_plate_image`` does, then replaces only the per-cell
boundary step. It returns the SAME DataFrame schema
``[row, col, size, circularity, flags, cx, cy]`` so ``normalize_plate`` /
``score_plate`` consume it without change.

Decision + benchmarking rationale: [[experiments.019-echo-crispr-array.cellpose-segmentation-plan]].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from torchcell.sga.image import (
    MAX_ASPECT,
    MIN_COLONY_AREA,
    MIN_EXTENT,
    _detect_blobs_backlit,
    _estimate_angle,
    _fit_lines,
    _gel_polygon,
    _grayscale,
    _perimeter,
    _polarity_is_dark,
    _rotate,
    _signed_dist,
)


class CellposeSegConfig(BaseModel):
    """Parameters for Cellpose-SAM plate segmentation.

    ``node_tol``/``stray_tol`` are fractions of the fitted colony pitch: an
    instance whose centroid is within ``node_tol * pitch`` of a grid node is that
    well's colony; one within ``stray_tol * pitch`` (but farther than
    ``node_tol``) is a near-stray that INVALIDATES the well (flag ``M``); one
    farther than ``stray_tol`` from every node is an off-grid contaminant colony
    (counted, assigned to no well). ``diameter=None`` uses Cellpose-SAM's
    diameter-agnostic default; set it (px) only if faint colonies under-segment.
    """

    n_rows: int = 14
    n_cols: int = 22
    polarity: str = "auto"  # 'auto' | 'bright' | 'dark'
    node_tol: float = 0.55
    stray_tol: float = 1.0
    circularity_flag: float = 0.80
    circularity_reject: float = 0.65  # below this -> invalidate the well (non-circular)
    neighbor_invalidate: bool = (
        True  # invalidate the closest crowded neighbour of a multi
    )
    neighbor_dist: float = (
        0.85  # flag the nearest neighbour if within this*pitch of the multi
    )
    gel_detect: bool = True
    edge_policy: str = "flag"  # 'flag' | 'drop'
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    diameter: float | None = None  # no-op for Cellpose-SAM (cpsam is diameter-agnostic)
    contrast: str = "none"  # 'none' | 'clahe' | 'flatfield' -- boosts faint colonies
    clahe_clip: float = 0.01
    min_instance_area: int = Field(
        default=MIN_COLONY_AREA,
        description="drop Cellpose instances smaller than this (px)",
    )


class PlateSegResult(BaseModel):
    """Return bundle for a segmented plate: the gitter-style per-well table plus
    plate-level counts the classical path cannot report (raw instance count and
    off-grid contaminant colonies).
    """

    model_config = {"arbitrary_types_allowed": True}

    table: pd.DataFrame
    n_instances: int
    n_offgrid: int
    masks: np.ndarray | None = None
    # drawn instance-id -> invalidation category ('' accepted / 'M' multi / 'N'
    # neighbour / 'C' non-circular), for colouring overlays and the montage
    kept_color: dict[int, str] = Field(default_factory=dict)


def load_cellpose_model(gpu: bool = True) -> Any:
    """Load the Cellpose-SAM generalist model (``cpsam``). Weights auto-download
    to ``~/.cellpose/models`` on first use. Returns the live model; kept separate
    from segmentation so one model instance can process many plates.
    """
    from cellpose import models

    return models.CellposeModel(gpu=gpu)


def _fit_lattice(
    g: np.ndarray, n_rows: int, n_cols: int, polarity: str
) -> tuple[np.ndarray, float, bool, float, np.ndarray, tuple[int, int, int, int]]:
    """Reproduce ``quantify_plate_image``'s backlit lattice prologue: returns
    ``(nodes[n_rows,n_cols,2], pitch, invert, theta, center, roi)``. Kept in lockstep
    with ``image.py`` -- if that prologue changes, mirror it here.
    """
    cents, pitch, (r0, r1, c0, c1) = _detect_blobs_backlit(g, n_cols)
    invert = (
        _polarity_is_dark(g, (r0, r1, c0, c1))
        if polarity == "auto"
        else polarity == "dark"
    )
    if len(cents) < n_rows * n_cols * 0.2:
        raise ValueError(
            f"only {len(cents)} colony blobs detected for a {n_rows}x{n_cols} array; "
            f"the lattice fit would be unreliable."
        )
    center = np.array([(r0 + r1) / 2, (c0 + c1) / 2])
    theta = _estimate_angle(cents, pitch)
    cents_rot = _rotate(cents, -theta, center)
    ys_r = _fit_lines(cents_rot[:, 0], n_rows)
    xs_r = _fit_lines(cents_rot[:, 1], n_cols)
    gy, gx = np.meshgrid(ys_r, xs_r, indexing="ij")
    nodes_rot = np.stack([gy.ravel(), gx.ravel()], axis=1)
    nodes = _rotate(nodes_rot, theta, center).reshape(n_rows, n_cols, 2)
    return nodes, pitch, invert, theta, center, (r0, r1, c0, c1)


def _contrast_enhance(img: np.ndarray, method: str, clahe_clip: float) -> np.ndarray:
    """Boost faint-colony contrast BEFORE Cellpose, returning a 3-channel uint8.

    Faint colonies on backlit plates differ only slightly from the bright agar, so
    Cellpose misses them though the eye can see them. Two local operations lift them:
    ``clahe`` (contrast-limited adaptive histogram equalization -- local contrast
    stretch) and ``flatfield`` (divide out the smooth illumination background, which
    also removes the transillumination gradient). Applied to luminance, restacked to
    RGB. Only the image Cellpose SEES is changed; the lattice fit uses the original.
    """
    from skimage import color, exposure
    from skimage.filters import gaussian

    gray = color.rgb2gray(img)  # float [0,1]
    if method == "clahe":
        enh = exposure.equalize_adapthist(gray, clip_limit=clahe_clip)
    elif method == "flatfield":
        bg = gaussian(gray, sigma=max(gray.shape) * 0.03)  # type: ignore[no-untyped-call]
        enh = np.clip(gray / (bg + 1e-6), 0, None)
        enh = exposure.rescale_intensity(enh, out_range=(0.0, 1.0))  # type: ignore[no-untyped-call]
    else:
        raise ValueError(f"contrast must be 'none'|'clahe'|'flatfield', got {method!r}")
    out: np.ndarray = (np.stack([enh] * 3, axis=-1) * 255).astype(np.uint8)
    return out


def _well(
    ri: int, ci: int, size: int, circ: float, flags: str, cx: float, cy: float, iid: int
) -> dict[str, float | str | int]:
    """One per-well record (1-based row/col); ``iid`` is the drawn instance id or -1."""
    return {
        "row": ri + 1,
        "col": ci + 1,
        "size": size,
        "circularity": circ,
        "flags": flags,
        "cx": cx,
        "cy": cy,
        "id": iid,
    }


# invalidation category -> outline colour (green accepted; agar is yellow, so the
# non-circular category is PURPLE, not yellow)
_CATEGORY_COLOR = {
    "": (0, 255, 0),  # accepted -- green
    "M": (255, 0, 0),  # multiple colonies -- red
    "N": (255, 140, 0),  # neighbour of a multi well -- orange
    "C": (170, 0, 255),  # non-circular -- purple
}


def _instance_props(masks: np.ndarray, min_area: int) -> list[dict[str, float]]:
    """Per-instance geometry from a Cellpose integer-label image: area, centroid,
    circularity, bbox aspect, extent. Instances below ``min_area`` are dropped.
    """
    props: list[dict[str, float]] = []
    ids = np.unique(masks)
    for i in ids[ids > 0]:
        blob = masks == i
        area = int(blob.sum())
        if area < min_area:
            continue
        ys, xs = np.where(blob)
        cy, cx = float(ys.mean()), float(xs.mean())
        hb = int(ys.max() - ys.min() + 1)
        wb = int(xs.max() - xs.min() + 1)
        aspect = max(hb, wb) / max(1, min(hb, wb))
        extent = area / float(hb * wb)
        perim = _perimeter(blob)
        circ = float(min(1.0, 4 * np.pi * area / (perim**2))) if perim else 0.0
        props.append(
            {
                "id": int(i),
                "area": area,
                "cy": cy,
                "cx": cx,
                "aspect": aspect,
                "extent": extent,
                "circ": circ,
            }
        )
    return props


# distinct, green-free instance-fill colours (green is reserved for accepted
# outlines); cycled per colony so touching colonies are visually separable.
_INSTANCE_COLORS = np.array(
    [
        [230, 25, 75],
        [245, 130, 48],
        [255, 225, 25],
        [0, 130, 200],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [128, 0, 0],
    ],
    dtype=float,
)


def _draw_cellpose_overlay(
    path: str, masks: np.ndarray, kept_color: dict[int, str], df: pd.DataFrame, out: str
) -> None:
    """Visual-QC overlay: draw each IN-GEL colony's circumference on the original,
    coloured by invalidation category -- green accepted, red multi-colony, orange
    neighbour-of-multi, purple non-circular. Only kept (in-gel, on-grid) instances
    are drawn, so off-plate/frame detections never appear. ``df`` is accepted for a
    stable signature but the colouring comes from ``kept_color`` (instance id ->
    category), which matches exactly what the pipeline invalidates.
    """
    from skimage.segmentation import find_boundaries

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    over = np.asarray(im).copy()
    masks = np.asarray(masks)
    # draw accepted first, then invalidations on top (purple < orange < red priority)
    for cat in ("", "C", "N", "M"):
        ids = [i for i, c in kept_color.items() if c == cat]
        if not ids:
            continue
        sel = np.where(np.isin(masks, ids), masks, 0)
        over[find_boundaries(sel, mode="thick")] = list(_CATEGORY_COLOR[cat])  # type: ignore[no-untyped-call]
    Image.fromarray(over).save(out)


def quantify_plate_image_cellpose(
    path: str,
    model: Any,
    cfg: CellposeSegConfig | None = None,
    overlay_path: str | None = None,
    return_masks: bool = False,
) -> PlateSegResult:
    """Segment a backlit plate with Cellpose-SAM and quantify colonies onto the
    fitted array grid.

    Returns a ``PlateSegResult`` whose ``table`` matches the classical
    ``quantify_plate_image`` schema (``row, col, size, circularity, flags, cx,
    cy``); ``flags`` codes: ``C`` low circularity, ``E`` straddles gel edge,
    ``M`` multiple colonies on the well (rejected downstream).
    """
    cfg = cfg or CellposeSegConfig()
    g = _grayscale(path)
    nodes, pitch, invert, theta, center, _roi = _fit_lattice(
        g, cfg.n_rows, cfg.n_cols, cfg.polarity
    )

    gel_sd = None
    if cfg.gel_detect:
        _gel_poly, gel_mask = _gel_polygon(g, nodes, pitch, theta, center)
        gel_sd = _signed_dist(gel_mask)
    edge_margin = 0.5 * pitch

    img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    if cfg.contrast != "none":
        img = _contrast_enhance(img, cfg.contrast, cfg.clahe_clip)
    masks = model.eval(
        img,
        flow_threshold=cfg.flow_threshold,
        cellprob_threshold=cfg.cellprob_threshold,
        diameter=cfg.diameter,
    )[0]
    props = _instance_props(np.asarray(masks), cfg.min_instance_area)

    node_yx = nodes.reshape(-1, 2)  # (n_rows*n_cols, 2)
    # bucket each in-gel instance to its nearest node; track off-grid contaminants
    buckets: dict[int, list[dict[str, float]]] = {}
    n_offgrid = 0
    for p in props:
        if gel_sd is not None:
            yi = min(max(int(round(p["cy"])), 0), g.shape[0] - 1)
            xi = min(max(int(round(p["cx"])), 0), g.shape[1] - 1)
            if gel_sd[yi, xi] < -edge_margin:
                continue  # outside the gel hexagon -> frame/panel, not a colony
        d = np.hypot(node_yx[:, 0] - p["cy"], node_yx[:, 1] - p["cx"])
        j = int(np.argmin(d))
        dist = float(d[j])
        if dist > cfg.stray_tol * pitch:
            n_offgrid += 1
            continue
        p = {**p, "dist": dist}
        buckets.setdefault(j, []).append(p)

    recs: list[dict[str, float | str | int]] = []
    well_idx: dict[tuple[int, int], int] = {}  # (ri,ci) -> recs index, occupied only
    for ri in range(cfg.n_rows):
        for ci in range(cfg.n_cols):
            j = ri * cfg.n_cols + ci
            yc, xc = float(node_yx[j, 0]), float(node_yx[j, 1])
            insts = buckets.get(j, [])
            on = [p for p in insts if p["dist"] <= cfg.node_tol * pitch]
            near = [
                p
                for p in insts
                if cfg.node_tol * pitch < p["dist"] <= cfg.stray_tol * pitch
            ]
            if not on and not near:
                recs.append(_well(ri, ci, 0, np.nan, "", xc, yc, -1))
                continue
            pool = on if on else near
            best = max(pool, key=lambda p: p["area"])
            size = int(best["area"])
            cym, cxm = best["cy"], best["cx"]
            sd_val = pitch
            if gel_sd is not None:
                syi = min(max(int(round(cym)), 0), g.shape[0] - 1)
                sxi = min(max(int(round(cxm)), 0), g.shape[1] - 1)
                sd_val = float(gel_sd[syi, sxi])
            accepted = (
                size >= MIN_COLONY_AREA
                and best["dist"] <= cfg.node_tol * pitch
                and best["aspect"] <= MAX_ASPECT
                and best["extent"] >= MIN_EXTENT
                and sd_val >= -edge_margin
                and (cfg.edge_policy != "drop" or sd_val >= edge_margin)
            )
            if not accepted:
                recs.append(_well(ri, ci, 0, np.nan, "", cxm, cym, -1))
                continue
            flags = ""
            if best["circ"] < cfg.circularity_reject:
                flags += "C"  # non-circular -> invalidate (purple)
            if -edge_margin <= sd_val < edge_margin:
                flags += "E"
            if len(on) >= 2 or (on and near):
                flags += "M"  # two colonies compete for nutrient -> invalidate (red)
            well_idx[(ri, ci)] = len(recs)
            recs.append(
                _well(
                    ri, ci, size, float(best["circ"]), flags, cxm, cym, int(best["id"])
                )
            )

    # neighbour invalidation: a multi well holds a duplicate that crowds ONE adjacent
    # colony. Flag the single nearest occupied orthogonal neighbour whose colony comes
    # within neighbor_dist*pitch of any of this well's colonies -- the closest crowded
    # neighbour, not all four; normally-spaced neighbours (further than the cutoff) are
    # left accepted.
    if cfg.neighbor_invalidate:
        for (ri, ci), idx in list(well_idx.items()):
            if "M" not in str(recs[idx]["flags"]):
                continue
            j = ri * cfg.n_cols + ci
            mine = [(float(p["cy"]), float(p["cx"])) for p in buckets.get(j, [])]
            best_nb: int | None = None
            best_d: float = 1e18
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = well_idx.get((ri + dr, ci + dc))
                if nb is None:
                    continue
                ny, nx = float(recs[nb]["cy"]), float(recs[nb]["cx"])
                nbd = float(
                    min(((ny - my) ** 2 + (nx - mx) ** 2) ** 0.5 for my, mx in mine)
                )
                if nbd < best_d:
                    best_d = nbd
                    best_nb = nb
            if (
                best_nb is not None
                and best_d < cfg.neighbor_dist * pitch
                and not ({"M", "N"} & set(str(recs[best_nb]["flags"])))
            ):
                recs[best_nb]["flags"] = str(recs[best_nb]["flags"]) + "N"

    kept_color: dict[int, str] = {}
    for r in recs:
        rid = int(r["id"])
        if rid < 0:
            continue
        f = str(r["flags"])
        kept_color[rid] = (
            "M" if "M" in f else "N" if "N" in f else "C" if "C" in f else ""
        )

    cols = ["row", "col", "size", "circularity", "flags", "cx", "cy"]
    df = pd.DataFrame([{k: r[k] for k in cols} for r in recs])
    df.loc[df["size"] < MIN_COLONY_AREA, ["size", "circularity"]] = [0, np.nan]

    masks_arr = np.asarray(masks)
    if overlay_path is not None:
        _draw_cellpose_overlay(path, masks_arr, kept_color, df, overlay_path)
    return PlateSegResult(
        table=df,
        n_instances=len(props),
        n_offgrid=n_offgrid,
        masks=masks_arr if return_masks else None,
        kept_color=kept_color,
    )
