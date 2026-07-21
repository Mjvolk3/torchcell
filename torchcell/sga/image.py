# torchcell/sga/image.py
# [[torchcell.sga.image]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/image
"""Image analysis: standardized plate photo -> per-colony size grid (the gitter
stage, done here so the pipeline is image-in / results-out).

Target input is a standardized dark-field capture (e.g. a pixl imager): bright
colonies on mid-gray agar, near-black background, fixed backlight bars/screws
outside the plate. We KNOW the array is a regular ``n_rows x n_cols`` block
(default 14x22 = the inner block of a border-free 384 plate), which makes a
projection-profile grid fit robust to missing colonies and to a dropped-gel
gash. Disrupted colonies are FLAGGED, not silently trusted:

  'C' - low circularity (mis-shapen; e.g. torn by the gash)
  'S' - spill / touches a detected dark agar-tear (gash) region

Output columns match gitter / ``read_gitter_dat``: row col size circularity flags
(row/col are IMAGE order: row 1 = top, col 1 = left; plate A1 orientation is
resolved downstream against the layout's blank pattern).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

# A detected blob is only accepted as a colony (and only then drawn / counted) if
# it clears these. Shared by the DataFrame and the overlay ``det`` mask so the two
# can never diverge (the old bug: ``det`` drew boundaries around blobs the size
# filter had already thrown away). ``MAX_ASPECT``/``MIN_EXTENT`` reject thin bars
# and jagged arrows (plate-frame fragments) that are never real colonies.
MIN_COLONY_AREA = 20  # px: smaller than this is not a colony
MAX_ASPECT = 2.5  # bbox long/short: above this is a stripe, not a colony
MIN_EXTENT = 0.45  # area / bbox area: below this is jagged/hollow, not a colony


def _grayscale(path: str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(im, dtype=float)


def _disk(r: int) -> np.ndarray:
    """Boolean disk structuring element of radius r for morphology."""
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    disk: np.ndarray = (x * x + y * y) <= r * r
    return disk


def _plate_roi(g: np.ndarray, pad_frac: float = 0.06) -> tuple[int, int, int, int]:
    """Bounding box of the agar plate (largest mid-bright component), shrunk by
    ``pad_frac`` on each side to drop the bright plastic frame.
    """
    lab, n = ndimage.label(g > 40)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = int(sizes.argmax()) + 1
    ys, xs = np.where(lab == big)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    dh, dw = int((r1 - r0) * pad_frac), int((c1 - c0) * pad_frac)
    return r0 + dh, r1 - dh, c0 + dw, c1 - dw


def _polarity_is_dark(g: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    """True if colonies are DARKER than agar (transillumination), False if
    brighter (dark-field). Decided by the agar background level: a bright plate
    (backlit, agar ~200) has dark colonies; a dark plate (dark-field, agar ~60)
    has bright colonies.
    """
    r0, r1, c0, c1 = roi
    return float(np.median(g[r0:r1, c0:c1])) > 128.0


def _colony_signal(g: np.ndarray, invert: bool = False) -> np.ndarray:
    """Flatten lighting and keep colony signal (>=0). ``invert`` for dark
    colonies on bright agar (transillumination).
    """
    bg = ndimage.gaussian_filter(g, sigma=60)
    enh = (bg - g) if invert else (g - bg)
    signal: np.ndarray = np.clip(enh, 0, None)
    return signal


def _detect_blobs(
    g: np.ndarray,
    enh: np.ndarray,
    roi: tuple[int, int, int, int],
    invert: bool = False,
    n_cols: int = 22,
) -> tuple[np.ndarray, float]:
    """Find colony centroids (roughly round blobs) inside the ROI.

    Returns an (N, 2) array of (y, x) centroids and the estimated grid pitch
    (median nearest-neighbour spacing). ``invert`` handles dark-on-bright.
    """
    r0, r1, c0, c1 = roi
    inside = np.zeros_like(g, bool)
    inside[r0:r1, c0:c1] = True
    agar_med = float(np.median(g[inside]))
    if invert:  # dark colonies: signal where g is well below agar
        mask = inside & (g < min(agar_med * 0.85, np.percentile(g[inside], 15)))
    else:  # bright colonies
        mask = inside & (g > max(agar_med * 2.0, np.percentile(g[inside], 90)))
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2)), 60.0
    # keep round, well-filled, colony-sized blobs; reject lid-reflection STREAKS
    # (thin/elongated) and glare (poorly filled) that would skew the lattice fit.
    lo, hi = 25, (0.6 * (c1 - c0) / n_cols) ** 2 * np.pi * 4
    slices = ndimage.find_objects(lab)
    keep = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        area = int(comp.sum())
        if not (lo < area < hi):
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        aspect = max(h, w) / max(1, min(h, w))
        extent = area / (h * w)
        if aspect <= 2.2 and extent >= 0.5:  # round-ish and filled
            keep.append(i)
    cents = np.array(ndimage.center_of_mass(mask, lab, keep))
    # drop blobs hugging the ROI wall: those are lid/wall reflections, not
    # colonies (the plated array is inset ~1 well from the wall). They otherwise
    # anchor the lattice fit onto a phantom edge column.
    m = 0.03 * (c1 - c0)
    on = (
        (cents[:, 0] > r0 + m)
        & (cents[:, 0] < r1 - m)
        & (cents[:, 1] > c0 + m)
        & (cents[:, 1] < c1 - m)
    )
    cents = cents[on]
    # pitch = median nearest-neighbour distance
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1)))
    return cents, pitch


def _detect_blobs_backlit(
    g: np.ndarray, n_cols: int = 24
) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    """Detect colonies on a BACKLIT plate, and derive the ROI from them.

    ``_plate_roi`` assumes the plate is the brightest thing on a dark field. That
    fails for a plate resting on a light PANEL: panel and agar sit at the same
    level (~213 here), so ``g > 40`` fuses them into one component and the ROI
    becomes the panel. A ROI that large also drags the 15th-percentile threshold
    in ``_detect_blobs`` down into the dark surround, at which point the mildly
    darker colonies never clear it and detection returns nothing at all.

    So invert the dependency: the colonies are the only regular DARK lattice on a
    bright field, so let them define the region. Detection is relative to a
    heavily blurred local background rather than to an absolute cut, which is
    what makes it work at the low contrast (colonies only ~6-20 gray levels below
    agar) that transillumination produces.

    Returns (centroids, pitch, roi) with the ROI padded to enclose the outermost
    colony cells.
    """
    bg = ndimage.gaussian_filter(g, sigma=60)
    bright = bg > 0.80 * np.percentile(bg, 99)  # agar + light panel, not surround
    depth = bg - g  # colonies are darker than their local background
    thr = max(6.0, float(np.percentile(depth[bright], 88)))
    mask = ndimage.binary_opening(bright & (depth > thr), iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2)), 60.0, (0, g.shape[0], 0, g.shape[1])

    hi = (0.6 * g.shape[1] / n_cols) ** 2 * np.pi * 4
    keep = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if not (20 < area < hi):
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) / max(1, min(h, w)) <= 2.2 and area / (h * w) >= 0.5:
            keep.append(i)
    if not keep:
        return np.empty((0, 2)), 60.0, (0, g.shape[0], 0, g.shape[1])

    cents = np.array(ndimage.center_of_mass(mask, lab, keep)).reshape(-1, 2)
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    pitch = float(np.median(nn))
    # a real colony always has a neighbour ~1 pitch away; isolated specks (dust,
    # plate-edge glints) do not, and would otherwise stretch the lattice fit.
    cents = cents[nn < 1.8 * pitch]
    if len(cents) == 0:
        return cents, pitch, (0, g.shape[0], 0, g.shape[1])
    pad = 0.7 * pitch
    r0 = max(0, int(cents[:, 0].min() - pad))
    r1 = min(g.shape[0], int(cents[:, 0].max() + pad))
    c0 = max(0, int(cents[:, 1].min() - pad))
    c1 = min(g.shape[1], int(cents[:, 1].max() + pad))
    return cents, pitch, (r0, r1, c0, c1)


def _estimate_angle(cents: np.ndarray, pitch: float) -> float:
    """Estimate grid rotation (radians) from near-horizontal neighbour vectors."""
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    angles = []
    for i in range(len(cents)):
        for j in range(len(cents)):
            if i == j or not (0.6 * pitch < d[i, j] < 1.4 * pitch):
                continue
            dy, dx = cents[j] - cents[i]
            a = np.arctan2(dy, dx)
            if abs(a) < np.radians(25):  # near-horizontal row neighbours
                angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def _rotate(points: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    """Rotate (y, x) points by ``theta`` about ``center``."""
    ct, st = np.cos(theta), np.sin(theta)
    rel = points - center
    y = rel[:, 0] * ct - rel[:, 1] * st
    x = rel[:, 0] * st + rel[:, 1] * ct
    rotated: np.ndarray = np.stack([y, x], axis=1) + center
    return rotated


def _fit_lines(coords: np.ndarray, n: int) -> np.ndarray:
    """Fit ``n`` evenly-spaced line centers to 1-D lattice points (colony
    coordinates along one axis).

    Uses a lattice fit: for a candidate (offset, pitch), snap each coordinate to
    its nearest integer index and score by residual PLUS a penalty for any point
    whose index falls outside ``0..n-1``. The out-of-range penalty is what forces
    the pitch to stretch to cover every real column/row -- a too-small pitch that
    would drop the outer columns is rejected. Robust to missing colonies and to a
    few glare outliers.
    """
    lo, hi = np.percentile(coords, 1), np.percentile(coords, 99)
    p_ideal = (hi - lo) / (n - 1)
    best: np.ndarray | None = None
    best_cost = np.inf
    for pitch in np.linspace(0.85 * p_ideal, 1.25 * p_ideal, 120):
        for x0 in np.linspace(lo - 0.5 * pitch, lo + 0.5 * pitch, 45):
            idx = np.round((coords - x0) / pitch)
            inb = (idx >= 0) & (idx <= n - 1)
            if inb.sum() < 0.6 * len(coords):
                continue
            pred = x0 + idx * pitch
            resid = float(np.mean(((coords - pred)[inb]) ** 2))
            cost = resid + 3.0 * pitch**2 * float((~inb).mean())
            if cost < best_cost:
                best_cost = cost
                best = x0 + pitch * np.arange(n)
    assert best is not None, "lattice fit found no candidate line set"
    return best


def _signed_dist(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask boundary: >0 inside, <0 outside (px). One array
    lookup then tells us how far a colony centroid is inside/outside the gel.
    """
    sd: np.ndarray = ndimage.distance_transform_edt(
        mask
    ) - ndimage.distance_transform_edt(~mask)
    return sd


def _gel_polygon(
    g: np.ndarray,
    nodes: np.ndarray,
    pitch: float,
    theta: float,
    center: np.ndarray,
    margin_frac: float = 0.6,
    chamfer_pitch: float = 1.3,
    chamfer_corners: tuple[int, ...] = (2, 3),
) -> tuple[np.ndarray, np.ndarray]:
    """Six-sided gel boundary (a chamfered rectangle) derived from the fitted
    colony lattice. Returns ``(poly (K,2) image (y,x) vertices, gel_mask HxW bool)``.

    The rectangle is fit in the grid's OWN rotated frame over ALL nodes, so a colony
    row that pokes past the plate edge cannot bulge it -- this is the fix for the
    off-centre t72 captures where the top row overlaps the edge (a global fit over
    hundreds of nodes is barely moved by one row).

    ``chamfer_corners`` indexes the rotated-frame corners ``[tl, tr, br, bl]`` (0..3)
    to cut; the default ``(2, 3)`` is the two BOTTOM corners, matching the imaging SOP
    of loading the plate chamfers-down (per-image darkness auto-detection proved
    framing-dependent and unreliable). ``chamfer_pitch`` is the chamfer edge length in
    pitch units (short: ~1 well). Anything outside this polygon is frame / panel, not
    gel, and its detections are rejected in ``quantify_plate_image``.
    """
    from skimage.draw import polygon as _sk_polygon

    rot = _rotate(nodes.reshape(-1, 2), -theta, center)  # axis-aligned lattice frame
    m = margin_frac * pitch
    y0, y1 = float(rot[:, 0].min() - m), float(rot[:, 0].max() + m)
    x0, x1 = float(rot[:, 1].min() - m), float(rot[:, 1].max() + m)
    ch = chamfer_pitch * pitch
    corners = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])  # CW: tl tr br bl
    cut = set(int(i) % 4 for i in chamfer_corners)
    verts = []
    for i in range(4):
        c = corners[i]
        if i in cut:  # replace the corner with the two chamfer endpoints
            v_prev = corners[(i - 1) % 4] - c
            v_next = corners[(i + 1) % 4] - c
            verts.append(c + v_prev / (np.linalg.norm(v_prev) + 1e-9) * ch)
            verts.append(c + v_next / (np.linalg.norm(v_next) + 1e-9) * ch)
        else:
            verts.append(c)
    poly_img = _rotate(np.array(verts), theta, center)
    rr, cc = _sk_polygon(poly_img[:, 0], poly_img[:, 1], g.shape)
    mask = np.zeros(g.shape, bool)
    mask[rr, cc] = True
    return poly_img, mask


def _segment_border(
    cell: np.ndarray,
    invert: bool,
    pitch: float,
    close_elem: np.ndarray,
    open_elem: np.ndarray,
    method: str = "threshold",
) -> np.ndarray:
    """Segment one lattice cell against the BRIGHT (agar) reference. ``method``
    selects the intensity-cut threshold (default) or a marker-controlled
    watershed; see ``quantify_plate_image``'s ``seg_method``.
    """
    if method == "watershed":
        return _segment_watershed(cell, invert, pitch)
    if method != "threshold":
        raise ValueError(
            f"seg_method must be 'threshold' or 'watershed', got {method!r}"
        )
    # Reference agar to the BRIGHT pixels of the cell (a high percentile for
    # dark-on-bright transillumination), not the border ring: near the plate frame
    # the border picks up dark frame shadow, which corrupts a border-median estimate
    # and drops the outermost colonies. The bright percentile is the agar level
    # regardless of a dark frame intruding on one edge or of colony fill fraction.
    if invert:  # dark colonies on bright agar
        # p90 agar stays ABOVE the colony even when the colony fills the whole cell
        # (p80 would drop into colony pixels and under-cut a big colony); a dark
        # frame edge is low, so it never pulls p90 down.
        agar = float(np.percentile(cell, 90))
        ref = cell[cell > np.percentile(cell, 70)]
        spread = float(np.median(np.abs(ref - np.median(ref)))) if ref.size else 1.0
        # 2.5 sigma matches the detector/watershed marker so a colony that earned a
        # lattice node clears the cut; the absolute 6-level floor stops pure agar
        # noise from firing on an empty cell.
        k = max(6.0, 2.5 * 1.4826 * spread)
        fg = cell < agar - k
    else:  # bright colonies on dark background
        agar = float(np.percentile(cell, 20))
        ref = cell[cell < np.percentile(cell, 45)]
        spread = float(np.median(np.abs(ref - np.median(ref)))) if ref.size else 1.0
        fg = cell > agar + 4.0 * 1.4826 * spread
    fg = ndimage.binary_closing(fg, close_elem)
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_opening(fg, open_elem)
    seg: np.ndarray = fg
    return seg


def _segment_watershed(cell: np.ndarray, invert: bool, pitch: float) -> np.ndarray:
    """Marker-controlled watershed with INTENSITY (not fixed-radius) markers.

    Colony marker = clearly-dark pixels in the central region (dark -> colony;
    central -> a dark plate-frame stripe at the cell edge is excluded). Agar marker
    = clearly-bright pixels in the outer region. The rim and the glare band on the
    colony dome are left unmarked; watershed floods the intensity gradient to assign
    them, so the boundary lands on the true rim and the bright glare stays inside the
    colony basin. Unlike a fixed-radius agar ring, this does not clip big colonies
    that fill their cell (the ring would land on the colony and pull the edge in).
    """
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    h, w = cell.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    central = rad <= 0.34 * pitch
    outer = rad >= 0.34 * pitch

    smooth = ndimage.gaussian_filter(cell, 1.0)
    agar = float(np.percentile(cell, 80)) if invert else float(np.percentile(cell, 20))
    ref = (
        cell[cell > np.percentile(cell, 55)]
        if invert
        else cell[cell < np.percentile(cell, 45)]
    )
    spread = float(np.median(np.abs(ref - np.median(ref)))) or 1.0
    depth = 3.0 * 1.4826 * spread  # agar-to-colony contrast scale

    markers = np.zeros(cell.shape, dtype=np.int32)
    if invert:  # dark colonies on bright agar
        colony = central & (smooth <= agar - depth)
        agar_m = outer & (smooth >= agar - 0.3 * depth)
    else:  # bright colonies on dark background
        colony = central & (smooth >= agar + depth)
        agar_m = outer & (smooth <= agar + 0.3 * depth)
    if not colony.any():
        return np.zeros(cell.shape, bool)  # empty well -> no colony
    markers[colony] = 1
    markers[agar_m] = 2

    labels = watershed(sobel(smooth), markers)
    filled: np.ndarray = ndimage.binary_fill_holes(labels == 1)
    return filled


def quantify_plate_image(
    path: str,
    n_rows: int = 14,
    n_cols: int = 22,
    overlay_path: str | None = None,
    circularity_flag: float = 0.80,
    polarity: str = "auto",
    grid_mode: str = "roi",
    multi_area_frac: float = 0.30,
    multi_min_area: int = 25,
    seg_method: str = "threshold",
    return_masks: bool = False,
    gel_detect: bool = True,
    edge_policy: str = "flag",
    node_tol: float = 0.55,
) -> pd.DataFrame:
    """Quantify colonies on a standardized plate image into a gitter-style table.

    ``polarity``: 'auto' (default) detects bright-colony dark-field vs
    dark-colony transillumination from the agar level; force with 'bright'/'dark'.

    ``grid_mode``: 'roi' (default) segments the plate first and looks for
    colonies inside it — correct for a dark-field capture where the plate is the
    brightest object. Use 'lattice' for a BACKLIT capture (plate on a light
    panel), where plate and panel are equally bright and plate segmentation
    fails; the colony lattice then defines the region itself. See
    ``_detect_blobs_backlit``.

    ``multi_area_frac`` / ``multi_min_area``: a cell is flagged 'M' (multiple
    colonies -> rejected downstream) when it holds a second blob of at least
    ``multi_min_area`` px AND at least ``multi_area_frac`` of the kept blob's
    area. Competing colonies both grow smaller, so such a cell is not a faithful
    fitness readout.

    ``seg_method`` (backlit ``grid_mode='lattice'`` only): 'threshold' (default) is
    the border/bright-referenced intensity cut; 'watershed' is a marker-controlled
    watershed (colony-center seed + agar seed, flooding the intensity gradient) that
    keeps a glare-lit dome inside the colony and separates touching colonies, at the
    cost of leaking where a dark plate-frame stripe abuts a colony with no agar gap.
    The dark-field ``grid_mode='roi'`` path always uses the median threshold.

    Returns columns: row, col, size, circularity, flags, cx, cy — where cx, cy
    are the MEASURED colony centroids (not the grid node), so an overlay drawn at
    them lands on the real colony. ``flags`` is a string of single-char codes:
    'C' low circularity, 'S' spill/gash, 'M' multiple colonies in the cell.
    """
    g = _grayscale(path)
    if grid_mode == "lattice":
        cents, pitch, (r0, r1, c0, c1) = _detect_blobs_backlit(g, n_cols)
        invert = (
            _polarity_is_dark(g, (r0, r1, c0, c1))
            if polarity == "auto"
            else polarity == "dark"
        )
    elif grid_mode == "roi":
        r0, r1, c0, c1 = _plate_roi(g)
        invert = (
            _polarity_is_dark(g, (r0, r1, c0, c1))
            if polarity == "auto"
            else polarity == "dark"
        )
        enh = _colony_signal(g, invert)
        cents, pitch = _detect_blobs(g, enh, (r0, r1, c0, c1), invert, n_cols)
    else:
        raise ValueError(f"grid_mode must be 'roi' or 'lattice', got {grid_mode!r}")
    if len(cents) < n_rows * n_cols * 0.2:
        raise ValueError(
            f"only {len(cents)} colony blobs detected for a {n_rows}x{n_cols} array "
            f"in {path!r}; the lattice fit would be unreliable. Try "
            f"grid_mode='lattice' for a backlit capture."
        )
    center = np.array([(r0 + r1) / 2, (c0 + c1) / 2])
    theta = _estimate_angle(cents, pitch)
    cents_rot = _rotate(cents, -theta, center)
    ys_r = _fit_lines(cents_rot[:, 0], n_rows)
    xs_r = _fit_lines(cents_rot[:, 1], n_cols)
    # lattice nodes in rotated space -> rotate back to image coordinates
    gy, gx = np.meshgrid(ys_r, xs_r, indexing="ij")
    nodes_rot = np.stack([gy.ravel(), gx.ravel()], axis=1)
    nodes = _rotate(nodes_rot, theta, center).reshape(n_rows, n_cols, 2)
    half = int(pitch * 0.45)

    # --- agar-tear (gash) mask: extreme non-colony deviation inside the plate.
    # Dark-field: the tear reads dark; transillumination: it reads bright. ---
    inside = np.zeros_like(g, bool)
    inside[r0:r1, c0:c1] = True
    agar_med = float(np.median(g[inside]))
    gash = inside & (g > min(255, agar_med * 1.30) if invert else g < agar_med * 0.55)
    gash = ndimage.binary_opening(gash, iterations=2)

    # per-cell colony segmentation. Two thresholding modes:
    #   * BORDER-referenced (grid_mode='lattice'): the cell's border ring is
    #     always agar, so thresholding against it captures the whole colony even
    #     when the colony fills most of the cell and its domed top carries a bright
    #     backlight glare band (the median-of-cell threshold would keep only the
    #     darker bottom -> half-moon). Morphological closing bridges the glare
    #     band; fill_holes solidifies. This is what makes full-circumference
    #     measurement robust on the backlit plates.
    #   * MEDIAN-of-cell (grid_mode='roi'): the original SGA-style threshold, kept
    #     byte-for-byte for the dark-field Plate-5 path.
    use_border_seg = grid_mode == "lattice"
    close_r = max(2, int(round(0.07 * pitch)))
    close_elem, open_elem = _disk(close_r), _disk(2)
    # 6-sided gel boundary: any detection outside it is frame/panel, not colony, and
    # is rejected. Lattice/backlit only; the dark-field roi path keeps its behaviour.
    gel_poly, gel_mask, gel_sd = None, None, None
    if use_border_seg and gel_detect:
        gel_poly, gel_mask = _gel_polygon(g, nodes, pitch, theta, center)
        gel_sd = _signed_dist(gel_mask)
    edge_margin = 0.5 * pitch  # a centroid this far outside the gel is rejected
    det = np.zeros_like(g, bool)  # global detected-colony mask (for the overlay)
    records = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            yc, xc = nodes[ri, ci]
            y0, y1 = int(yc - half), int(yc + half)
            x0, x1 = int(xc - half), int(xc + half)
            cell = g[y0:y1, x0:x1]
            cell_gash = gash[y0:y1, x0:x1]
            if cell.size == 0 or min(cell.shape) < 4:
                records.append((ri + 1, ci + 1, 0, np.nan, "S", float(xc), float(yc)))
                continue
            # gel gate: a NODE sitting well outside the gel is frame/panel -- do not
            # even segment it (this is where the phantom frame boundaries came from).
            if gel_sd is not None:
                nyi = min(max(int(round(yc)), 0), g.shape[0] - 1)
                nxi = min(max(int(round(xc)), 0), g.shape[1] - 1)
                if gel_sd[nyi, nxi] < -edge_margin:
                    records.append(
                        (ri + 1, ci + 1, 0, np.nan, "", float(xc), float(yc))
                    )
                    continue
            size, circ, flags = 0, np.nan, ""
            cxm, cym = float(xc), float(yc)
            if use_border_seg:
                fg = _segment_border(
                    cell, invert, pitch, close_elem, open_elem, seg_method
                )
                lab, ncomp = ndimage.label(fg)
                if ncomp:
                    cyl, cxl = (y1 - y0) / 2, (x1 - x0) / 2
                    coms = ndimage.center_of_mass(fg, lab, range(1, ncomp + 1))
                    areas = np.asarray(ndimage.sum(fg, lab, range(1, ncomp + 1)))
                    dists = np.array([np.hypot(p[0] - cyl, p[1] - cxl) for p in coms])
                    # keep the LARGEST blob near the node, not the strictly nearest:
                    # a noise speck at the node used to steal the cell from an
                    # off-centre real colony, dropping it entirely (a Class-B miss).
                    central = np.where(dists <= node_tol * pitch)[0]
                    if len(central):
                        keep = int(central[int(np.argmax(areas[central]))]) + 1
                    else:
                        keep = int(np.argmin(dists)) + 1
                    blob = lab == keep
                    size = int(blob.sum())
                    ys_b, xs_b = np.where(blob)
                    hb = int(ys_b.max() - ys_b.min() + 1)
                    wb = int(xs_b.max() - xs_b.min() + 1)
                    aspect = max(hb, wb) / max(1, min(hb, wb))
                    extent = size / float(hb * wb)
                    perim = _perimeter(blob)
                    circ = (
                        float(min(1.0, 4 * np.pi * size / (perim**2))) if perim else 0.0
                    )
                    by, bx = coms[keep - 1]  # measured centroid, image coords
                    cym, cxm = y0 + by, x0 + bx
                    dist_node = float(np.hypot(by - cyl, bx - cxl))
                    if gel_sd is not None:
                        syi = min(max(int(round(cym)), 0), g.shape[0] - 1)
                        sxi = min(max(int(round(cxm)), 0), g.shape[1] - 1)
                        sd_val = float(gel_sd[syi, sxi])
                    else:
                        sd_val = pitch
                    # ONE acceptance predicate, shared by BOTH the DataFrame and the
                    # overlay `det`, so they can never diverge (the old artefact bug):
                    # colony-sized, round-ish/solid (not a bar/arrow), on its own node
                    # (not neighbour bleed or empty-agar noise), and inside the gel.
                    accepted = (
                        size >= MIN_COLONY_AREA
                        and dist_node <= node_tol * pitch
                        and aspect <= MAX_ASPECT
                        and extent >= MIN_EXTENT
                        and sd_val >= -edge_margin
                        and (edge_policy != "drop" or sd_val >= edge_margin)
                    )
                    if accepted:
                        if circ < circularity_flag:
                            flags += "C"
                        if -edge_margin <= sd_val < edge_margin:
                            flags += "E"  # colony straddles the gel edge
                        det[y0:y1, x0:x1] |= blob
                        # Multiple colonies in one cell: a SECOND colony-sized blob
                        # (>= multi_min_area px and >= multi_area_frac of the kept
                        # blob) means two colonies shared this position; they compete
                        # for nutrient so neither is a faithful readout. Flag 'M'.
                        thresh = max(multi_min_area, multi_area_frac * size)
                        kc = coms[keep - 1]
                        extra = [
                            j
                            for j in range(1, ncomp + 1)
                            if j != keep
                            and areas[j - 1] >= thresh
                            and np.hypot(coms[j - 1][0] - kc[0], coms[j - 1][1] - kc[1])
                            > 0.4 * pitch
                        ]
                        if extra and size >= multi_min_area:
                            flags += "M"
                            for j in extra:
                                det[y0:y1, x0:x1] |= lab == j
                    else:  # not a colony -> record as empty, draw nothing
                        size, circ, cxm, cym = 0, np.nan, float(xc), float(yc)
            else:  # dark-field roi path (Plate 5): original behaviour, unchanged
                med = float(np.median(cell))
                mad = float(np.median(np.abs(cell - med))) or 1.0
                k = 3.0 * 1.4826 * mad
                fg = (cell < med - k) if invert else (cell > med + k)
                fg = ndimage.binary_opening(fg, iterations=1)
                lab, ncomp = ndimage.label(fg)
                if ncomp:
                    cyl, cxl = (y1 - y0) / 2, (x1 - x0) / 2
                    coms = ndimage.center_of_mass(fg, lab, range(1, ncomp + 1))
                    areas = ndimage.sum(fg, lab, range(1, ncomp + 1))
                    dists = np.array(
                        [((p[0] - cyl) ** 2 + (p[1] - cxl) ** 2) for p in coms]
                    )
                    keep = int(np.argmin(dists)) + 1
                    blob = lab == keep
                    size = int(blob.sum())
                    perim = _perimeter(blob)
                    circ = (
                        float(min(1.0, 4 * np.pi * size / (perim**2))) if perim else 0.0
                    )
                    if circ < circularity_flag:
                        flags += "C"
                    by, bx = coms[keep - 1]
                    cym, cxm = y0 + by, x0 + bx
                    det[y0:y1, x0:x1] |= blob
                    thresh = max(multi_min_area, multi_area_frac * size)
                    kc = coms[keep - 1]
                    extra = [
                        j
                        for j in range(1, ncomp + 1)
                        if j != keep
                        and areas[j - 1] >= thresh
                        and (
                            (coms[j - 1][0] - kc[0]) ** 2
                            + (coms[j - 1][1] - kc[1]) ** 2
                        )
                        ** 0.5
                        > 0.4 * pitch
                    ]
                    if extra and size >= multi_min_area:
                        flags += "M"
                        for j in extra:
                            det[y0:y1, x0:x1] |= lab == j
            if cell_gash.mean() > 0.05:
                flags += "S"
            records.append((ri + 1, ci + 1, size, circ, flags, cxm, cym))

    # belt-and-suspenders: nothing outside the gel hexagon or inside a tear can
    # survive in the overlay mask, independent of any per-cell bug (lattice only).
    if use_border_seg:
        det &= ~gash
        if gel_mask is not None:
            det &= gel_mask

    df = pd.DataFrame(
        records, columns=["row", "col", "size", "circularity", "flags", "cx", "cy"]
    )
    # a colony that is essentially absent -> size 0 (missing)
    df.loc[df["size"] < MIN_COLONY_AREA, ["size", "circularity"]] = [0, np.nan]

    if overlay_path is not None:
        _draw_overlay(path, df, det, overlay_path, (r0, r1, c0, c1), gash, gel_poly)
    if return_masks:
        return df, det
    return df


def _perimeter(blob: np.ndarray) -> float:
    er = ndimage.binary_erosion(blob)
    return float((blob & ~er).sum())


def _draw_overlay(
    path: str,
    df: pd.DataFrame,
    det: np.ndarray,
    out: str,
    roi: tuple[int, int, int, int],
    gash: np.ndarray,
    gel_poly: np.ndarray | None = None,
) -> None:
    """Draw the detected colony boundaries (green) + a cross at each measured
    centroid, the gash region (blue), the ROI (yellow), and the 6-sided gel
    boundary (cyan) when detected. Cross colour: green clean, red flagged (C/S),
    magenta 'M' (multiple colonies -> rejected, also boxed). Green is the SGA/gitter
    convention and is kept consistent across all overlays in this project.
    """
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ov = np.asarray(im).copy()
    ov[gash] = [0, 90, 255]  # gash region in blue
    edge = det & ~ndimage.binary_erosion(det, iterations=2)  # 2-px colony boundary
    ov[edge] = [0, 255, 0]  # detected-colony outline in green
    im = Image.fromarray(ov)
    d = ImageDraw.Draw(im)
    r0, r1, c0, c1 = roi
    d.rectangle([c0, r0, c1, r1], outline=(255, 255, 0), width=2)
    if gel_poly is not None:  # the 6-sided gel boundary; detections outside are cut
        d.polygon([(float(x), float(y)) for y, x in gel_poly], outline=(0, 200, 220))
    for _, r in df.iterrows():
        if r["size"] > 0:
            x, y = r["cx"], r["cy"]
            if "M" in str(r["flags"]):
                col = (255, 0, 255)  # magenta: multiple colonies, rejected
                d.rectangle([x - 7, y - 7, x + 7, y + 7], outline=col, width=2)
            else:
                col = (255, 60, 60) if r["flags"] else (0, 255, 0)  # red if flagged
            d.line([x - 5, y, x + 5, y], fill=col, width=1)
            d.line([x, y - 5, x, y + 5], fill=col, width=1)
    im.save(out)
