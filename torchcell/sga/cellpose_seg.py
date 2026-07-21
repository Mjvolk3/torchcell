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
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field
from scipy import ndimage

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
    gel_detect: bool = True
    edge_policy: str = "flag"  # 'flag' | 'drop'
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    diameter: float | None = None
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


def _draw_cellpose_overlay(
    path: str, masks: np.ndarray, kept_ids: set[int], df: pd.DataFrame, out: str
) -> None:
    """QC overlay in the project's SGA convention: kept colony boundaries in green,
    a cross at each measured centroid (green clean, magenta 'M' rejected). Mirrors
    ``image._draw_overlay`` so cellpose and classical overlays read the same.
    """
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ov = np.asarray(im).copy()
    det = np.isin(masks, list(kept_ids)) if kept_ids else np.zeros(masks.shape, bool)
    edge = det & ~ndimage.binary_erosion(det, iterations=2)
    ov[edge] = [0, 220, 0]
    pim = Image.fromarray(ov)
    dr = ImageDraw.Draw(pim)
    for _, r in df.iterrows():
        if r["size"] <= 0:
            continue
        x, y = float(r["cx"]), float(r["cy"])
        col = (255, 0, 255) if "M" in str(r["flags"]) else (0, 220, 0)
        dr.line([(x - 5, y), (x + 5, y)], fill=col, width=1)
        dr.line([(x, y - 5), (x, y + 5)], fill=col, width=1)
    pim.save(out)


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

    records: list[tuple[int, int, int, float, str, float, float]] = []
    kept_ids: set[int] = set()
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
                records.append((ri + 1, ci + 1, 0, np.nan, "", xc, yc))
                continue
            pool = on if on else near
            best = max(pool, key=lambda p: p["area"])
            size = int(best["area"])
            cym, cxm = best["cy"], best["cx"]
            flags = ""
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
                records.append((ri + 1, ci + 1, 0, np.nan, "", cxm, cym))
                continue
            if best["circ"] < cfg.circularity_flag:
                flags += "C"
            if -edge_margin <= sd_val < edge_margin:
                flags += "E"
            # two colonies claim this well (a second on-node instance, or a
            # near-stray in the ring): they compete for nutrient -> reject.
            if len(on) >= 2 or (on and near):
                flags += "M"
            kept_ids.add(int(best["id"]))
            records.append((ri + 1, ci + 1, size, float(best["circ"]), flags, cxm, cym))

    df = pd.DataFrame(
        records, columns=["row", "col", "size", "circularity", "flags", "cx", "cy"]
    )
    df.loc[df["size"] < MIN_COLONY_AREA, ["size", "circularity"]] = [0, np.nan]

    masks_arr = np.asarray(masks)
    if overlay_path is not None:
        _draw_cellpose_overlay(path, masks_arr, kept_ids, df, overlay_path)
    return PlateSegResult(
        table=df,
        n_instances=len(props),
        n_offgrid=n_offgrid,
        masks=masks_arr if return_masks else None,
    )
