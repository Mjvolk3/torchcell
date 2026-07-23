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
    tighten_size: bool = Field(
        default=True,
        description="shrink each accepted mask to the dark colony pixels (Otsu within "
        "the mask), dropping the faint halo Cellpose includes -- removes the air gap "
        "between the drawn outline and the colony and makes the stored size the colony "
        "itself. Detection stays as permissive as cellprob/contrast allow.",
    )
    tighten_min_frac: float = Field(
        default=0.20,
        description="keep the original mask if the tightened (dark) region is smaller "
        "than this fraction of it -- guards faint colonies where Otsu over-shrinks.",
    )
    tighten_grow_px: int = Field(
        default=3,
        description="dilate the Otsu colony core back out this many px toward the "
        "visual edge (Otsu splits a hair inside the soft rim), bounded by the original "
        "mask. 0 = raw Otsu core (tightest); larger = looser. 3 lands on the edge.",
    )
    relax_grid: bool = Field(
        default=True,
        description="after the classical even-spacing lattice fit, snap each row/column "
        "line to the median of the Cellpose colony centroids assigned to it, so the grid "
        "FOLLOWS a distorted/perspective plate (rows land on colony rows, not between "
        "them). Empty margin rows are extrapolated from the colony grid. Fixes the "
        "off-center labels + off-gel leak + mis-bucketed multis on skewed captures.",
    )
    homography_refit: bool = Field(
        default=True,
        description="after the even-spacing relax, refit the lattice as a PROJECTIVE "
        "(homography) map from array index (row, col) to pixels, estimated by RANSAC from "
        "the colonies the relaxed grid already places well. A homography is the exact "
        "omni-tray-on-a-table perspective, so it recovers rows a single isotropic pitch "
        "SKIPS on a tilted capture (in-image row pitch != column pitch, i.e. a trapezoid) "
        "-- while leaving correctly-fit plates unchanged. Deterministic (seeded RANSAC).",
    )
    edge_margin_frac: float = Field(
        default=0.5,
        description="gel-edge gate half-width in pitch units. A colony is dropped as "
        "off-gel only if it sits more than this*pitch outside the fitted gel hexagon. "
        "Larger keeps more true edge-row (row A / row P) colonies.",
    )
    multi_min_frac: float = Field(
        default=0.5,
        description="a second colony on/near a well triggers the multi ('M') flag only "
        "if its area is at least this fraction of the primary colony's -- so a Cellpose "
        "over-split fragment of one colony no longer falsely rejects the well.",
    )
    recover_missed_wells: bool = Field(
        default=True,
        description="after assignment, recover colonies Cellpose MISSED: at every empty "
        "in-gel grid node, probe the intensity depression vs the local agar and, if it "
        "reads as a real colony (depth > recover_depth_thresh), threshold it in (Otsu "
        "core + tighten_grow_px, the SAME size basis as a detected colony -- faintness is "
        "a contrast property, not a size one, so a recovered colony is scored normally). "
        "Cellpose misses the FAINTEST colonies at the corners/edges (backlight falloff) "
        "on the less-grown plates; the correct grid tells us exactly which wells to check. "
        "Recovered wells carry detector='recovered' (provenance only, not a quality flag).",
    )
    recover_depth_thresh: float = Field(
        default=12.0,
        description="minimum agar-minus-colony depression (gray levels) for an empty node "
        "to be recovered as a colony. Empty wells sit near 0; real colonies are far above "
        "this -- so it separates faint colonies from bare agar without recovering blanks.",
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


def _relax_lattice(
    nodes: np.ndarray,
    cents: np.ndarray,
    n_rows: int,
    n_cols: int,
    theta: float,
    center: np.ndarray,
    iters: int = 4,
) -> np.ndarray:
    """Snap the even-spacing lattice to the colony centroids so it follows plate
    distortion. Working in the de-rotated frame, each row line is moved to the median
    y of the centroids nearest it (each column line to the median x), iterated a few
    times; rows/columns with too few colonies (empty margins) are linearly
    extrapolated from the ones that snapped, so they sit on the colony grid rather
    than drifting into the lid/frame. Returns nodes in the original frame.
    """
    cr = _rotate(cents, -theta, center)
    nrot = _rotate(nodes.reshape(-1, 2), -theta, center).reshape(n_rows, n_cols, 2)

    def _snap(
        base: np.ndarray, n: int, coord: np.ndarray, min_count: int
    ) -> np.ndarray:
        vals = base.copy()
        snapped = np.zeros(n, dtype=bool)
        for _ in range(iters):
            idx = np.abs(coord[:, None] - vals[None, :]).argmin(axis=1)
            for k in range(n):
                m = idx == k
                if int(m.sum()) >= min_count:
                    vals[k] = float(np.median(coord[m]))
                    snapped[k] = True
        if snapped.sum() >= 2:  # extrapolate empty lines from the snapped grid
            ks = np.where(snapped)[0]
            a, b = np.polyfit(ks, vals[ks], 1)
            miss = np.where(~snapped)[0]
            vals[miss] = a * miss + b
        return vals

    ys = _snap(nrot[:, :, 0].mean(axis=1), n_rows, cr[:, 0], 4)
    xs = _snap(nrot[:, :, 1].mean(axis=0), n_cols, cr[:, 1], 3)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    nodes_rot = np.stack([gy.ravel(), gx.ravel()], axis=1)
    return _rotate(nodes_rot, theta, center).reshape(n_rows, n_cols, 2)


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography ``H`` mapping ``src`` -> ``dst`` (both ``(N, 2)``) by the direct
    linear transform, with isotropic (Hartley) normalization for conditioning. In
    homogeneous coords ``dst ~ H @ [src; 1]``. Used to model the array as a projective
    grid: index space ``(col, row)`` -> pixels, exact for a plate photographed at an
    angle (perspective/trapezoid).
    """

    def _norm(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = p.mean(0)
        s = np.sqrt(2) / (np.sqrt(((p - c) ** 2).sum(1)).mean() + 1e-12)
        t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = (t @ np.c_[p, np.ones(len(p))].T).T
        return ph[:, :2], t

    sn, ts = _norm(src)
    dn, td = _norm(dst)
    n = len(src)
    a = np.zeros((2 * n, 9))
    for k in range(n):
        u, v = sn[k]
        x, y = dn[k]
        a[2 * k] = [-u, -v, -1, 0, 0, 0, x * u, x * v, x]
        a[2 * k + 1] = [0, 0, 0, -u, -v, -1, y * u, y * v, y]
    _, _, vt = np.linalg.svd(a)
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    out: np.ndarray = h / h[2, 2]
    return out


def _apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography ``h`` to ``(N, 2)`` points, returning ``(N, 2)``. A degenerate
    RANSAC sample can map a point onto the line at infinity (``w`` -> 0); clamp ``w`` so
    such a point becomes a large finite coordinate (rejected as an outlier) rather than
    a divide-by-zero NaN.
    """
    ph = (h @ np.c_[pts, np.ones(len(pts))].T).T
    w = ph[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out: np.ndarray = ph[:, :2] / w
    return out


def _homography_lattice(
    nodes: np.ndarray,
    cents: np.ndarray,
    n_rows: int,
    n_cols: int,
    pitch: float,
    iters: int = 4,
    ransac_iters: int = 200,
    thresh: float = 8.0,
) -> np.ndarray:
    """Refit the lattice as a PROJECTIVE grid and return the corrected nodes.

    The even-spacing relax carries a single isotropic pitch, so on a plate shot at a
    tilt -- where the in-image row pitch differs from the column pitch (a trapezoid) --
    the 16-row lattice is too short to span the rows and SKIPS the faintest interior
    rows. Here we instead assign colonies to the current grid to label them with array
    indices ``(row, col)``, RANSAC-fit a homography ``index -> pixel`` (the exact
    perspective of an omni tray on a light box), regenerate all nodes from it, and
    iterate. RANSAC discards the mislabeled colonies of any skipped row, so the fit
    recovers those rows; on a correctly-fit plate the homography reproduces the grid.
    Deterministic: fixed RNG seed.
    """
    rng = np.random.default_rng(0)
    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    idx_all = np.stack([jj.ravel(), ii.ravel()], axis=1).astype(float)  # (col, row)
    grid = nodes
    for _ in range(iters):
        nyx = grid.reshape(-1, 2)
        src, dst = [], []
        for cy, cx in cents:
            d = np.hypot(nyx[:, 0] - cy, nyx[:, 1] - cx)
            k = int(np.argmin(d))
            if d[k] <= 0.5 * pitch:
                i, j = divmod(k, n_cols)
                src.append([j, i])
                dst.append([cx, cy])
        if len(src) < 12:
            return grid
        src_a = np.array(src, float)
        dst_a = np.array(dst, float)
        best_h: np.ndarray | None = None
        best_inl = -1
        best_mask = np.ones(len(src_a), bool)
        for _r in range(ransac_iters):
            sel = rng.choice(len(src_a), 4, replace=False)
            try:
                h = _fit_homography(src_a[sel], dst_a[sel])
            except np.linalg.LinAlgError:
                continue
            err = np.sqrt(((_apply_homography(h, src_a) - dst_a) ** 2).sum(1))
            mask = err < thresh
            if int(mask.sum()) > best_inl:
                best_inl = int(mask.sum())
                best_h = h
                best_mask = mask
        if best_h is None:
            return grid
        h = _fit_homography(src_a[best_mask], dst_a[best_mask])  # refit on inliers
        px = _apply_homography(h, idx_all)  # (x, y)
        grid = np.stack([px[:, 1], px[:, 0]], axis=1).reshape(n_rows, n_cols, 2)
    return grid


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


def _tighten_instance(
    masks: np.ndarray,
    iid: int,
    g: np.ndarray,
    invert: bool,
    min_frac: float,
    grow_px: int = 0,
) -> int:
    """Shrink instance ``iid`` in ``masks`` to its dark-colony core and return the
    tightened area (px). Cellpose masks on backlit plates include a faint halo of
    near-agar pixels; Otsu within the mask splits colony (dark, if ``invert``) from
    halo (bright), we keep the largest colony component and ZERO the halo pixels of
    this id in ``masks`` (so drawn contours + montage tighten too). The core is then
    dilated ``grow_px`` px back out toward the visual edge (Otsu splits a hair inside
    the soft colony rim), bounded by the original mask so it never re-enters the halo
    or a neighbour. If the colony core is < ``min_frac`` of the mask (a faint colony
    Otsu over-shrinks), the mask is left untouched and the original area returned.
    """
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as _label
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    ys, xs = np.where(masks == iid)
    if ys.size == 0:
        return 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    subm = masks[y0:y1, x0:x1] == iid
    subg = g[y0:y1, x0:x1]
    vals = subg[subm]
    if vals.size < 10 or vals.min() == vals.max():
        return int(subm.sum())
    t = threshold_otsu(vals)  # type: ignore[no-untyped-call]
    colony = (subg <= t) if invert else (subg >= t)
    colony &= subm
    lab, n = _label(colony)
    if n == 0:
        return int(subm.sum())
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    cc = lab == int(counts.argmax())
    if grow_px > 0:  # grow back to the visual edge, bounded by the original mask
        cc = binary_dilation(cc, structure=disk(grow_px)) & subm  # type: ignore[no-untyped-call]
    if cc.sum() < max(min_frac * subm.sum(), MIN_COLONY_AREA):
        return int(subm.sum())  # over-shrunk faint colony -> keep original mask
    masks[y0:y1, x0:x1][subm & ~cc] = 0  # zero the halo pixels of this id
    return int(cc.sum())


def _well(
    ri: int,
    ci: int,
    size: int,
    circ: float,
    flags: str,
    cx: float,
    cy: float,
    iid: int,
    detector: str = "",
) -> dict[str, float | str | int]:
    """One per-well record (1-based row/col); ``iid`` is the drawn instance id or -1.
    ``detector`` records provenance: '' empty, 'cellpose' detected, 'recovered' filled in
    by the grid-guided miss recovery.
    """
    return {
        "row": ri + 1,
        "col": ci + 1,
        "size": size,
        "circularity": circ,
        "flags": flags,
        "cx": cx,
        "cy": cy,
        "id": iid,
        "detector": detector,
    }


def _recover_colony(
    g: np.ndarray,
    ny: int,
    nx: int,
    pitch: float,
    invert: bool,
    depth_thresh: float,
    grow_px: int,
) -> tuple[int, float, np.ndarray, int, int] | None:
    """Recover a Cellpose-missed colony at an empty grid node from image intensity.

    Cellpose misses the faintest colonies (low contrast vs agar, worst at the backlight-
    falloff corners); the correct grid says a well SHOULD sit here, so we look directly.
    Measure the agar-minus-colony depression (median annulus minus 25th-pct core); if it
    clears ``depth_thresh`` the well holds a colony. Threshold it with Otsu inside a disk
    and grow the core ``grow_px`` px -- the SAME size basis as ``_tighten_instance`` -- so
    a faint colony gets its TRUE size (faintness is contrast, not growth). Returns
    ``(size, circularity, colony_bbox_mask, y0, x0)`` or None if no colony.
    """
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as _label
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk as _disk

    r_out = int(0.78 * pitch)
    y0, y1 = max(0, ny - r_out), min(g.shape[0], ny + r_out + 1)
    x0, x1 = max(0, nx - r_out), min(g.shape[1], nx + r_out + 1)
    sub = g[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(yy - ny, xx - nx)
    core = sub[rr <= 0.28 * pitch]
    ann = sub[(rr >= 0.55 * pitch) & (rr <= r_out)]
    if core.size < 20 or ann.size < 20:
        return None
    bg = float(np.median(ann))
    depth = (
        bg - float(np.percentile(core, 25))
        if invert
        else float(np.percentile(core, 75)) - bg
    )
    if depth <= depth_thresh:
        return None
    win = rr <= 0.46 * pitch
    vals = sub[win]
    if vals.size < 20 or vals.min() == vals.max():
        return None
    t = threshold_otsu(vals)  # type: ignore[no-untyped-call]
    colony = ((sub <= t) if invert else (sub >= t)) & win
    lab, n = _label(colony)
    if n == 0:
        return None
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    cc = lab == int(counts.argmax())
    if grow_px > 0:
        cc = binary_dilation(cc, structure=_disk(grow_px)) & win  # type: ignore[no-untyped-call]
    size = int(cc.sum())
    if size < MIN_COLONY_AREA:
        return None
    perim = _perimeter(cc)
    circ = float(min(1.0, 4 * np.pi * size / (perim**2))) if perim else 0.0
    return size, circ, cc, y0, x0


# invalidation category -> outline colour (green accepted; agar is yellow, so the
# non-circular category is PURPLE, not yellow)
_CATEGORY_COLOR = {
    "": (0, 255, 0),  # accepted -- green
    "M": (255, 0, 0),  # multiple colonies -- red
    "N": (255, 140, 0),  # neighbour of a multi well -- orange
    "C": (170, 0, 255),  # non-circular -- purple
    "R": (0, 128, 255),  # grid-recovered Cellpose miss -- blue
    "X": (255, 20, 147),  # extra colliding colony (cause of a multi) -- deep pink
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
    # draw accepted first, then recovered, then invalidations + collision partners on top
    for cat in ("", "R", "X", "C", "N", "M"):
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
    precomputed_masks: np.ndarray | None = None,
) -> PlateSegResult:
    """Segment a backlit plate with Cellpose-SAM and quantify colonies onto the
    fitted array grid.

    Returns a ``PlateSegResult`` whose ``table`` matches the classical
    ``quantify_plate_image`` schema (``row, col, size, circularity, flags, cx,
    cy``); ``flags`` codes: ``C`` low circularity, ``E`` straddles gel edge,
    ``M`` multiple colonies on the well (rejected downstream).

    ``precomputed_masks`` skips the Cellpose forward pass and re-runs only the
    grid/gel/assignment/scoring on a given instance-mask array (``model`` may be
    ``None``) -- for re-deriving the grid (e.g. relaxation) from cached masks
    without a GPU. If those masks are already Otsu-tightened, pass
    ``tighten_size=False`` so they are not tightened twice.
    """
    cfg = cfg or CellposeSegConfig()
    g = _grayscale(path)
    nodes, pitch, invert, theta, center, _roi = _fit_lattice(
        g, cfg.n_rows, cfg.n_cols, cfg.polarity
    )
    edge_margin = cfg.edge_margin_frac * pitch

    if precomputed_masks is not None:
        masks = np.asarray(precomputed_masks)
    else:
        img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        if cfg.contrast != "none":
            img = _contrast_enhance(img, cfg.contrast, cfg.clahe_clip)
        masks = np.asarray(
            model.eval(
                img,
                flow_threshold=cfg.flow_threshold,
                cellprob_threshold=cfg.cellprob_threshold,
                diameter=cfg.diameter,
            )[0]
        )
    props = _instance_props(masks, cfg.min_instance_area)

    # relax the grid onto the Cellpose colony centroids (follows plate distortion),
    # then derive the gel polygon from the corrected grid so its boundary tracks the
    # true colony extent (no lid/frame leak). Drop NN-isolated specks first so a lone
    # lid/frame detection cannot bias the snap.
    if cfg.relax_grid and len(props) >= 0.5 * cfg.n_rows * cfg.n_cols:
        cents = np.array([[p["cy"], p["cx"]] for p in props])
        dd = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        cents = cents[dd.min(axis=1) < 1.8 * pitch]
        if len(cents) >= 0.5 * cfg.n_rows * cfg.n_cols:
            nodes = _relax_lattice(nodes, cents, cfg.n_rows, cfg.n_cols, theta, center)
            if cfg.homography_refit:
                nodes = _homography_lattice(nodes, cents, cfg.n_rows, cfg.n_cols, pitch)

    gel_sd = None
    if cfg.gel_detect:
        _gel_poly, gel_mask = _gel_polygon(g, nodes, pitch, theta, center)
        gel_sd = _signed_dist(gel_mask)

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
    empty_wells: list[tuple[int, int, int]] = []  # (ri,ci,recs_idx) with no instance
    competitor_ids: set[int] = set()  # extra colliding colonies (cause of a multi flag)
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
                empty_wells.append((ri, ci, len(recs)))
                recs.append(_well(ri, ci, 0, np.nan, "", xc, yc, -1))
                continue
            pool = on if on else near
            best = max(pool, key=lambda p: p["area"])
            orig_area = int(best["area"])
            cym, cxm = best["cy"], best["cx"]
            sd_val = pitch
            if gel_sd is not None:
                syi = min(max(int(round(cym)), 0), g.shape[0] - 1)
                sxi = min(max(int(round(cxm)), 0), g.shape[1] - 1)
                sd_val = float(gel_sd[syi, sxi])
            accepted = (
                orig_area >= MIN_COLONY_AREA
                and best["dist"] <= cfg.node_tol * pitch
                and best["aspect"] <= MAX_ASPECT
                and best["extent"] >= MIN_EXTENT
                and sd_val >= -edge_margin
                and (cfg.edge_policy != "drop" or sd_val >= edge_margin)
            )
            if not accepted:
                recs.append(_well(ri, ci, 0, np.nan, "", cxm, cym, -1))
                continue
            # tighten the stored size to the dark colony core (drop the halo); detection
            # already happened, so this only sharpens the size/outline, never recall.
            size = (
                _tighten_instance(
                    masks,
                    int(best["id"]),
                    g,
                    invert,
                    cfg.tighten_min_frac,
                    cfg.tighten_grow_px,
                )
                if cfg.tighten_size
                else orig_area
            )
            flags = ""
            if best["circ"] < cfg.circularity_reject:
                flags += "C"  # non-circular -> invalidate (purple)
            if -edge_margin <= sd_val < edge_margin:
                flags += "E"
            # multi ('M') fires only on a SECOND real colony (area >= multi_min_frac of
            # the primary) on/near the well -- a Cellpose over-split fragment no longer
            # falsely rejects a clean single colony.
            competitors = [
                p
                for p in (on + near)
                if int(p["id"]) != int(best["id"])
                and p["area"] >= cfg.multi_min_frac * orig_area
            ]
            if competitors:
                flags += "M"  # two colonies compete for nutrient -> invalidate (red)
                # the extra colliding colonies are not grid wells; mark them so the
                # collision shows BOTH colonies, not just the invalidated well.
                competitor_ids.update(int(p["id"]) for p in competitors)
            well_idx[(ri, ci)] = len(recs)
            recs.append(
                _well(
                    ri,
                    ci,
                    size,
                    float(best["circ"]),
                    flags,
                    cxm,
                    cym,
                    int(best["id"]),
                    "cellpose",
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

    # grid-guided recovery of Cellpose MISSES: at each empty in-gel node, look directly at
    # the image for a colony Cellpose failed to detect (faint corners / backlight falloff)
    # and fill it in with a true-size threshold. New instance ids are written into `masks`
    # so the overlay draws them; recovered wells are scored like any colony (detector tag
    # is provenance only).
    if cfg.recover_missed_wells:
        next_id = int(masks.max()) + 1
        for ri, ci, idx in empty_wells:
            yc = int(round(float(recs[idx]["cy"])))
            xc = int(round(float(recs[idx]["cx"])))
            if gel_sd is not None:
                yi = min(max(yc, 0), g.shape[0] - 1)
                xi = min(max(xc, 0), g.shape[1] - 1)
                if gel_sd[yi, xi] < -edge_margin:
                    continue  # off-gel -- do not invent colonies on the frame/bevel
            rec = _recover_colony(
                g, yc, xc, pitch, invert, cfg.recover_depth_thresh, cfg.tighten_grow_px
            )
            if rec is None:
                continue
            size, circ, cc, y0, x0 = rec
            masks[y0 : y0 + cc.shape[0], x0 : x0 + cc.shape[1]][cc] = next_id
            ys, xs = np.where(cc)
            recs[idx] = _well(
                ri,
                ci,
                size,
                circ,
                "",
                float(x0 + xs.mean()),
                float(y0 + ys.mean()),
                next_id,
                "recovered",
            )
            well_idx[(ri, ci)] = idx
            next_id += 1

    kept_color: dict[int, str] = {}
    for r in recs:
        rid = int(r["id"])
        if rid < 0:
            continue
        f = str(r["flags"])
        kept_color[rid] = (
            "R"
            if r.get("detector") == "recovered"
            else "M"
            if "M" in f
            else "N"
            if "N" in f
            else "C"
            if "C" in f
            else ""
        )
    # the extra colliding colonies (not grid wells) drawn distinctly from the red well
    for cid in competitor_ids:
        if cid not in kept_color:
            kept_color[cid] = "X"

    cols = ["row", "col", "size", "circularity", "flags", "cx", "cy", "detector"]
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
