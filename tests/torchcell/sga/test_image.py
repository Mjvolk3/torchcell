# tests/torchcell/sga/test_image.py
# [[tests.torchcell.sga.test_image]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_image.py
"""Classical plate quantification on the synthetic 3x4 plates of ``conftest.py``.

Hand-derived expectations, all on a 60 px pitch with node rows y = 90, 150, 210 and
node columns x = 110, 170, 230, 290:

* Dark-field (``grid_mode="roi"``) per-cell segmentation opens with the 3x3 cross, which
  removes the four corner pixels of a square: 7x7 -> 45 px, 9x9 -> 77 px. The U loses
  its six convex corners (195 -> 189 px, boundary 82 -> 76), so its circularity is
  4*pi*189/76^2 = 0.4112 (< 0.80, flag ``C``) and its centroid moves to 0.2646 px below
  the node ((1590 - 2*14) / 189 - 8 = 0.26455 rows from the bounding-box top row 8).
* Backlit (``grid_mode="lattice"``) segmentation closes with disk(4) and opens with
  disk(2) (the 13-pixel disk x^2 + y^2 <= 4); a 7x7 square keeps rows of
  3, 5, 7, 7, 7, 5, 3 = 37 px and a 9x9 keeps 5, 7, 9, 9, 9, 9, 9, 7, 5 = 69 px. The
  U's morphology result (193 px, circularity 0.592117, centroid 0.466321 px below the
  node) is pinned numerically, not by hand.
* Multi-colony ``M``: the secondary 7x7 sits 18*sqrt(2) = 25.46 px from the primary,
  above the 0.4 * 60 = 24 px separation, and its area clears max(25, 0.3 * primary).
* Gash ``S`` (dark-field): the 14x14 patch opened twice with the cross keeps
  196 - 4*3 = 184 px, 184 / 54^2 = 6.3% of the 54x54 cell (> 5%).
* Gel polygon: margin 0.6 * 60 = 36 px beyond the outer nodes, chamfer 1.3 * 60 = 78 px
  on the two bottom corners. A bottom-corner node is (1.3 - 2*0.6) / sqrt(2) * 60 =
  4.24 px OUTSIDE the chamfer line, inside the 0.5-pitch edge band, so those two
  colonies carry ``E`` and are dropped under ``edge_policy="drop"``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from PIL import Image

from torchcell.sga.image import (
    MAX_ASPECT,
    MIN_COLONY_AREA,
    MIN_EXTENT,
    _colony_signal,
    _detect_blobs,
    _detect_blobs_backlit,
    _disk,
    _estimate_angle,
    _fit_lines,
    _gel_polygon,
    _grayscale,
    _perimeter,
    _plate_roi,
    _polarity_is_dark,
    _rotate,
    _segment_border,
    _segment_watershed,
    _signed_dist,
    quantify_plate_image,
)

ROWS_Y = (90, 150, 210)
COLS_X = (110, 170, 230, 290)
PITCH = 60.0
# the 12 blob centroids both detectors return (raster order of the label image is not
# guaranteed, so tests sort); the U's centroid is 0.264550 px below its node
U_CY = 150 + (1590 - 2 * 14) / 189 - 8
EXPECTED_CENTS = sorted(
    [(90.0, 110.0), (90.0, 170.0), (90.0, 230.0), (90.0, 290.0), (108.0, 308.0)]
    + [(150.0, 110.0), (150.0, 290.0), (U_CY, 170.0)]
    + [(210.0, x) for x in (110.0, 170.0, 230.0, 290.0)]
)
BLUE = (0, 90, 255)
GREEN = (0, 255, 0)
RED = (255, 60, 60)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 200, 220)


def _rgb(path: str) -> NDArray[Any]:
    return np.asarray(Image.open(path).convert("RGB"))


def _where_color(ov: NDArray[Any], color: tuple[int, int, int]) -> NDArray[Any]:
    """(N, 2) array of (y, x) pixels equal to ``color``."""
    mask = np.all(ov == np.array(color, dtype=ov.dtype), axis=-1)
    return np.argwhere(mask)


def _cell(ri: int, ci: int) -> tuple[slice, slice]:
    """The 54x54 window (half = int(0.45 * 60) = 27) around a 1-based well."""
    y, x = ROWS_Y[ri - 1], COLS_X[ci - 1]
    return slice(y - 27, y + 27), slice(x - 27, x + 27)


# --- primitives -------------------------------------------------------------------


def test_constants_are_the_documented_acceptance_gates() -> None:
    """The three shared gates: 20 px, aspect 2.5, extent 0.45."""
    assert (MIN_COLONY_AREA, MAX_ASPECT, MIN_EXTENT) == (20, 2.5, 0.45)


def test_grayscale_reads_the_saved_bytes_as_float(dark_plate_path: str) -> None:
    """An 8-bit PNG round-trips exactly: agar 60 inside the plate, 0 outside, 200 on a
    colony, as float64.
    """
    g = _grayscale(dark_plate_path)
    assert g.dtype == np.float64
    assert g.shape == (300, 400)
    assert (g[0, 0], g[50, 50], g[90, 110]) == (0.0, 60.0, 200.0)


def test_disk_structuring_elements() -> None:
    """r=1 is the 3x3 cross (5 px); r=2 is the 13-pixel digital disk."""
    assert_array_equal(
        _disk(1), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    )
    d2 = _disk(2)
    assert d2.shape == (5, 5)
    assert int(d2.sum()) == 13
    assert not d2[0, 0] and not d2[0, 1] and d2[0, 2] and d2[1, 1]


def test_perimeter_counts_boundary_pixels_under_the_cross() -> None:
    """A 3x3 block has one interior pixel (8 boundary); a 3x9 rectangle has a 1x7
    interior (27 - 7 = 20); a lone pixel is all boundary (1).
    """
    blk = np.zeros((7, 7), bool)
    blk[2:5, 2:5] = True
    assert _perimeter(blk) == 8.0
    rect = np.zeros((7, 13), bool)
    rect[2:5, 2:11] = True
    assert _perimeter(rect) == 20.0
    one = np.zeros((3, 3), bool)
    one[1, 1] = True
    assert _perimeter(one) == 1.0


def test_plate_roi_is_the_largest_bright_component_shrunk_six_percent() -> None:
    """Plate at rows 10..89 / cols 20..79 (value 60): dh = int(79 * 0.06) = 4,
    dw = int(59 * 0.06) = 3 -> (14, 85, 23, 76). A brighter but smaller 5x5 component
    elsewhere does not win.
    """
    g = np.zeros((100, 100), float)
    g[10:90, 20:80] = 60.0
    g[2:7, 90:95] = 200.0
    assert tuple(int(v) for v in _plate_roi(g)) == (14, 85, 23, 76)


def test_polarity_is_decided_by_the_roi_median() -> None:
    """Median 60 inside the ROI -> bright colonies (False); median 213 -> dark (True).
    The pixels outside the ROI do not vote.
    """
    g = np.zeros((40, 40), float)
    g[10:30, 10:30] = 60.0
    assert _polarity_is_dark(g, (10, 30, 10, 30)) is False
    g[10:30, 10:30] = 213.0
    assert _polarity_is_dark(g, (10, 30, 10, 30)) is True


def test_colony_signal_is_the_positive_part_of_the_high_pass() -> None:
    """On a constant 100 image with one 200 spike, ``g - blur(g)`` is positive only at
    the spike (the blur is above 100 everywhere else), so exactly one pixel is nonzero
    and its value is 200 minus a blur just above 100 (the sigma-60 kernel spreads the
    spike's 100 excess over the whole 50x50 field, so under 0.1 of it lands back on the
    spike); ``invert`` flips the sign, so the spike is the only zero.
    """
    g = np.full((50, 50), 100.0)
    g[25, 25] = 200.0
    s = _colony_signal(g)
    assert int(np.count_nonzero(s)) == 1
    assert 99.9 < s[25, 25] < 100.0
    inv = _colony_signal(g, invert=True)
    assert inv[25, 25] == 0.0
    assert int(np.count_nonzero(inv)) == 50 * 50 - 1


def test_rotate_convention_and_inverse() -> None:
    """``_rotate`` maps (y, x) by y' = y cos - x sin, x' = y sin + x cos: the point
    (0, 1) rotated by pi/2 about the origin lands on (-1, 0). Rotating back by -theta
    about an arbitrary center restores the points.
    """
    out = _rotate(np.array([[0.0, 1.0]]), np.pi / 2, np.array([0.0, 0.0]))
    assert_allclose(out, [[-1.0, 0.0]], atol=1e-12)
    pts = np.array([[3.0, 4.0], [-2.0, 7.5], [10.0, 10.0]])
    c = np.array([1.0, 2.0])
    back = _rotate(_rotate(pts, 0.7, c), -0.7, c)
    assert_allclose(back, pts, atol=1e-12)
    # the center is a fixed point
    assert_allclose(_rotate(c[None], 1.1, c), c[None], atol=1e-12)


def test_estimate_angle_measures_the_row_direction() -> None:
    """Every near-horizontal neighbor vector of a tilted lattice has the same angle, so
    the median is that angle to floating precision: rows along (sin a, cos a) with
    a = 10 degrees give exactly a. A single point has no neighbor pair and yields 0.0.
    """
    a = np.radians(10.0)
    yy, xx = np.meshgrid([0.0, 10.0], [0.0, 10.0, 20.0], indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], 1)
    tilted = np.stack(
        [
            pts[:, 0] * np.cos(a) + pts[:, 1] * np.sin(a),
            -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a),
        ],
        1,
    )
    assert _estimate_angle(tilted, 10.0) == pytest.approx(a, abs=1e-12)
    assert _estimate_angle(np.array([[0.0, 0.0]]), 10.0) == 0.0


def test_untilt_doubles_the_lattice_rotation() -> None:
    """Finding: ``_rotate(v, t)`` maps a direction at angle ``a`` to ``a - t`` (y' =
    y cos t - x sin t, x' = y sin t + x cos t), while ``_estimate_angle`` returns ``a``
    itself, so the ``quantify_plate_image`` prologue ``_rotate(cents, -theta)`` sends the
    rows to angle ``a + theta = 2a`` instead of 0. A lattice built with
    ``_rotate(grid, 0.05)`` has rows at a = -0.05; ``_estimate_angle`` returns -0.05; a
    row of 8 colonies 420 px long has y-spread 420 sin(0.05) = 20.99 px before and
    420 sin(0.10) = 41.93 px after the "un-rotation" (exactly double, not zero).
    """
    yy, xx = np.meshgrid(np.arange(6) * 60.0, np.arange(8) * 60.0, indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], 1)
    c = grid.mean(0)
    cents = _rotate(grid, 0.05, c)
    theta = _estimate_angle(cents, 60.0)
    assert theta == pytest.approx(-0.05, abs=1e-12)
    unrot = _rotate(cents, -theta, c)
    assert np.ptp(cents[:8, 0]) == pytest.approx(420 * np.sin(0.05), abs=1e-9)
    assert np.ptp(unrot[:8, 0]) == pytest.approx(420 * np.sin(0.10), abs=1e-9)


def test_quantify_loses_colonies_on_a_tilted_plate(tmp_path: Path) -> None:
    """Finding (plate-level consequence of the sign mismatch above): a 5x7 backlit array
    of 9x9 colonies is fully recovered upright (35 wells at 69 px) but, rotated 6 degrees
    with nearest-neighbor resampling, the doubled tilt misfits the lattice and fewer than
    35 wells receive a colony. The exact count, 16, is pinned numerically (it depends on
    the resampled pixel geometry); the behavioral claim is the strict shortfall, which a
    sign fix must remove.
    """
    g = np.full((400, 500), 213, np.uint8)
    for ri in range(5):
        for ci in range(7):
            g[
                80 + 60 * ri - 4 : 80 + 60 * ri + 5, 70 + 60 * ci - 4 : 70 + 60 * ci + 5
            ] = 190
    im0 = Image.fromarray(g, "L")
    upright = str(tmp_path / "rot0.png")
    im0.save(upright)
    tilted = str(tmp_path / "rot6.png")
    im0.rotate(6, resample=Image.Resampling.NEAREST, fillcolor=213).save(tilted)
    df0 = quantify_plate_image(upright, 5, 7, grid_mode="lattice")
    assert df0["size"].tolist() == [69] * 35
    df6 = quantify_plate_image(tilted, 5, 7, grid_mode="lattice")
    n_found = int((df6["size"] > 0).sum())
    assert n_found < 35
    assert n_found == 16  # pinned numerically


def test_fit_lines_recovers_an_even_lattice_with_a_gap() -> None:
    """Coordinates 0, 10, 20, 30 fit four lines at those positions; with the 20 line
    absent the out-of-range penalty still stretches the pitch to cover 30, so the same
    four lines return. The search grid is 120 pitches x 45 offsets, so the result is
    exact to a fraction of a pixel, not to machine precision.
    """
    assert_allclose(
        _fit_lines(np.array([0.0, 10.0, 20.0, 30.0]), 4), [0, 10, 20, 30], atol=0.2
    )
    assert_allclose(
        _fit_lines(np.array([0.0, 10.0, 30.0]), 4), [0, 10, 20, 30], atol=0.2
    )


def test_signed_distance_is_positive_inside_negative_outside() -> None:
    """5x5 block in a 9x9 field: center (4, 4) is 3 px from the nearest outside pixel
    (1, 4); (0, 4) is 2 px from the nearest inside pixel (2, 4); a block edge pixel is
    +1.
    """
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    sd = _signed_dist(m)
    assert (sd[4, 4], sd[0, 4], sd[2, 4]) == (3.0, -2.0, 1.0)


def test_gel_polygon_chamfers_the_two_bottom_corners() -> None:
    """2x2 lattice at y 10/20, x 10/30, pitch 10, theta 0: margin 6 -> rectangle
    y 4..26, x 4..36; chamfer 13 px replaces the bottom-right corner (26, 36) by
    (13, 36) and (26, 23) and the bottom-left (26, 4) by (26, 17) and (13, 4). The
    mask is the inclusive 23x33 rectangle (759 px) minus two corner triangles of
    1 + 2 + ... + 13 = 91 px each = 577 px.
    """
    nodes = np.array([[[10.0, 10.0], [10.0, 30.0]], [[20.0, 10.0], [20.0, 30.0]]])
    poly, mask = _gel_polygon(
        np.zeros((60, 60)), nodes, 10.0, 0.0, np.array([15.0, 20.0])
    )
    assert_allclose(
        poly, [[4, 4], [4, 36], [13, 36], [26, 23], [26, 17], [13, 4]], atol=1e-6
    )
    assert int(mask.sum()) == 577
    assert mask[15, 20] and mask[4, 4] and mask[4, 36] and mask[26, 20]
    assert not mask[26, 36] and not mask[26, 4]


def test_gel_polygon_can_chamfer_the_top_corners_instead() -> None:
    """``chamfer_corners=(0, 1)`` cuts tl and tr: the top corners are replaced and the
    bottom ones kept, so the top-left corner pixel is outside and the bottom-left inside.
    """
    nodes = np.array([[[10.0, 10.0], [10.0, 30.0]], [[20.0, 10.0], [20.0, 30.0]]])
    poly, mask = _gel_polygon(
        np.zeros((60, 60)),
        nodes,
        10.0,
        0.0,
        np.array([15.0, 20.0]),
        chamfer_corners=(0, 1),
    )
    assert_allclose(
        poly, [[17, 4], [4, 17], [4, 23], [17, 36], [26, 36], [26, 4]], atol=1e-6
    )
    assert not mask[4, 4] and mask[26, 4] and mask[26, 36]


# --- per-cell segmentation ------------------------------------------------------------


def _backlit_cell() -> NDArray[Any]:
    """54x54 agar at 213 with a 7x7 colony at 190 centered at (26, 26)."""
    cell = np.full((54, 54), 213.0)
    cell[23:30, 23:30] = 190.0
    return cell


def test_segment_border_threshold_keeps_37_of_a_7x7_colony() -> None:
    """Agar p90 = 213, the ``cell > p70`` reference is empty so the spread falls back
    to 1.0 and k = max(6, 3.7) = 6: the cut is < 207, selecting the 49 colony pixels;
    disk(2) opening then trims the corners to 37 px (rows 3, 5, 7, 7, 7, 5, 3).
    """
    seg = _segment_border(_backlit_cell(), True, PITCH, _disk(4), _disk(2))
    assert seg.dtype == np.bool_
    assert int(seg.sum()) == 37
    assert seg[23:30, 23:30].sum() == 37  # nothing outside the square
    assert not seg[23, 23] and not seg[23, 24] and seg[23, 25]


def test_segment_border_bright_colony_branch() -> None:
    """Dark-field: agar p20 = 60, the ``cell < p45`` reference is empty, spread 1.0,
    cut > 60 + 4 * 1.4826 = 65.9: the 200-valued colony is selected, again 37 px.
    """
    cell = np.full((54, 54), 60.0)
    cell[23:30, 23:30] = 200.0
    assert int(_segment_border(cell, False, PITCH, _disk(4), _disk(2)).sum()) == 37


def test_segment_border_rejects_an_unknown_method() -> None:
    """The error names both valid methods and the offending value."""
    with pytest.raises(
        ValueError, match="seg_method must be 'threshold' or 'watershed', got 'foo'"
    ):
        _segment_border(_backlit_cell(), True, PITCH, _disk(4), _disk(2), method="foo")


def test_segment_border_watershed_dispatches_to_segment_watershed() -> None:
    """The two entry points return identical masks for the same cell."""
    cell = _backlit_cell()
    cell[0, 0] = 214.0
    via_border = _segment_border(
        cell, True, PITCH, _disk(4), _disk(2), method="watershed"
    )
    assert_array_equal(via_border, _segment_watershed(cell, True, PITCH))


def test_segment_watershed_returns_empty_mask_on_constant_agar() -> None:
    """Finding: on a cell whose upper 45% of intensities are all equal (flat synthetic
    agar at 213), ``cell > percentile(cell, 55)`` is empty, ``np.median`` of it is NaN,
    and ``float(nan) or 1.0`` keeps the NaN because NaN is truthy. The colony marker
    ``smooth <= agar - nan`` is then all False and the function returns no colony even
    though a clear 190-valued 7x7 colony is present.
    """
    seg = _segment_watershed(_backlit_cell(), True, PITCH)
    assert int(seg.sum()) == 0


def test_segment_watershed_floods_one_pixel_past_the_step() -> None:
    """With a single 214 agar pixel the reference set is non-empty (spread exactly 0,
    so the ``or 1.0`` fallback fires) and the watershed runs. The basin is 77 px in the
    bounding box rows/cols 22..30, one pixel outside the 7x7 step on every side minus the
    four corners; the value is pinned numerically, not derived: ``sobel(smooth)`` is
    exactly equal on both sides of the blurred step (10.3983 at (26, 22) and (26, 23)),
    so which side the ridge pixels fall on is a flooding-order tie-break inside
    ``skimage.segmentation.watershed``. An empty cell gives no marker and an all-False
    mask.
    """
    cell = _backlit_cell()
    cell[0, 0] = 214.0
    seg = _segment_watershed(cell, True, PITCH)
    assert int(seg.sum()) == 77
    ys, xs = np.where(seg)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (22, 30, 22, 30)
    empty = np.full((54, 54), 213.0)
    empty[0, 0] = 214.0
    assert int(_segment_watershed(empty, True, PITCH).sum()) == 0


# --- blob detection -------------------------------------------------------------------


def test_detect_blobs_dark_field_centroids_and_pitch(dark_plate_path: str) -> None:
    """All 12 bright blobs are found (the U and the M secondary included; the missing
    well is not), centroids exact for the symmetric squares, and the median
    nearest-neighbor distance is exactly 60 (the two 25.46 px M-pair distances and the
    one 45.7 px secondary-to-(2,4) distance are below the median position).
    """
    g = _grayscale(dark_plate_path)
    roi = _plate_roi(g)
    assert tuple(int(v) for v in roi) == (35, 264, 41, 358)
    cents, pitch = _detect_blobs(g, np.zeros_like(g), roi, False, 4)
    assert pitch == 60.0
    assert_allclose(sorted(map(tuple, cents.tolist())), EXPECTED_CENTS, atol=1e-9)


def test_detect_blobs_ignores_the_enh_argument(dark_plate_path: str) -> None:
    """Finding: ``_detect_blobs`` takes ``enh`` (the flattened colony signal that
    ``quantify_plate_image`` computes with ``_colony_signal``) but never reads it; the
    threshold is on ``g`` alone. Random noise for ``enh`` gives byte-identical output.
    """
    g = _grayscale(dark_plate_path)
    roi = _plate_roi(g)
    a, pa = _detect_blobs(g, np.zeros_like(g), roi, False, 4)
    noise = np.random.default_rng(0).normal(size=g.shape) * 1e6
    b, pb = _detect_blobs(g, noise, roi, False, 4)
    assert_array_equal(a, b)
    assert pa == pb


def test_detect_blobs_drops_wall_hugging_and_elongated_blobs(
    dark_plate_array: NDArray[Any], save_plate: Callable[[str, NDArray[Any]], str]
) -> None:
    """A 7x7 blob whose centroid is within 0.03 * (358 - 41) = 9.5 px of the ROI's
    left wall (x = 45 < 41 + 9.5) is a lid reflection and is dropped; a 3x15 bar
    (aspect 5 > 2.2) at the empty well is not a colony. The 12 base centroids remain.
    """
    g = dark_plate_array
    g[87:94, 42:49] = 200
    g[149:152, 223:238] = 200
    gg = _grayscale(save_plate("extras.png", g))
    cents, pitch = _detect_blobs(gg, np.zeros_like(gg), _plate_roi(gg), False, 4)
    assert pitch == 60.0
    assert_allclose(sorted(map(tuple, cents.tolist())), EXPECTED_CENTS, atol=1e-9)


def test_detect_blobs_empty_plate() -> None:
    """No blob -> an empty (0, 2) array and the 60 px default pitch."""
    g = np.zeros((300, 400), float)
    g[20:280, 20:380] = 60.0
    cents, pitch = _detect_blobs(g, np.zeros_like(g), (35, 264, 41, 358), False, 4)
    assert cents.shape == (0, 2)
    assert pitch == 60.0


def test_detect_blobs_backlit_centroids_pitch_and_roi(backlit_plate_path: str) -> None:
    """The dark colonies define the region: same 12 centroids as the dark-field
    detector, pitch 60, and the ROI is the centroid bounding box padded by
    0.7 * 60 = 42 px: (int(90 - 42), int(210 + 42), int(110 - 42), int(308 + 42)) =
    (48, 252, 68, 350).
    """
    g = _grayscale(backlit_plate_path)
    cents, pitch, roi = _detect_blobs_backlit(g, 4)
    assert pitch == 60.0
    assert roi == (48, 252, 68, 350)
    assert_allclose(sorted(map(tuple, cents.tolist())), EXPECTED_CENTS, atol=1e-9)
    assert _polarity_is_dark(g, roi) is True


def test_detect_blobs_backlit_drops_an_isolated_speck(
    backlit_plate_array: NDArray[Any], save_plate: Callable[[str, NDArray[Any]], str]
) -> None:
    """A lone 7x7 speck at (20, 385) is 118 px from the nearest colony (90, 290), more
    than 1.8 * 60 = 108, so it is removed before the ROI is derived: the ROI stays
    (48, 252, 68, 350) and the 12 lattice centroids are unchanged.
    """
    g = backlit_plate_array
    g[17:24, 382:389] = 190
    gg = _grayscale(save_plate("speck.png", g))
    cents, pitch, roi = _detect_blobs_backlit(gg, 4)
    assert pitch == 60.0
    assert roi == (48, 252, 68, 350)
    assert_allclose(sorted(map(tuple, cents.tolist())), EXPECTED_CENTS, atol=1e-9)


def test_detect_blobs_backlit_empty_field() -> None:
    """A uniform image has no depression: empty centroids, pitch 60, whole-frame ROI."""
    g = np.full((300, 400), 213.0)
    cents, pitch, roi = _detect_blobs_backlit(g, 4)
    assert cents.shape == (0, 2)
    assert pitch == 60.0
    assert roi == (0, 300, 0, 400)


# --- quantify_plate_image: dark-field roi path ---------------------------------------


def test_quantify_dark_field_table(dark_plate_path: str) -> None:
    """Sizes 45 (7x7 minus corners), 77 (9x9 minus corners, flag M), 189 (U, flag C,
    circularity 0.4112), 0 for the empty well (circularity NaN, flags ''), and S on
    the gash cell; measured centroids equal the shape centroids.
    """
    df = quantify_plate_image(dark_plate_path, 3, 4, grid_mode="roi")
    assert list(df.columns) == [
        "row",
        "col",
        "size",
        "circularity",
        "flags",
        "cx",
        "cy",
    ]
    assert len(df) == 12
    assert df["row"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert df["col"].tolist() == [1, 2, 3, 4] * 3
    assert df["size"].tolist() == [45, 45, 45, 77, 45, 189, 0, 45, 45, 45, 45, 45]
    assert df["flags"].tolist() == ["", "", "", "M", "", "C", "", "", "", "S", "", ""]
    circ = df["circularity"].to_numpy()
    assert np.isnan(circ[6])
    assert circ[5] == pytest.approx(4 * np.pi * 189 / 76**2, rel=1e-12)
    assert_allclose(np.delete(circ, [5, 6]), 1.0)
    by = {
        (int(r["row"]), int(r["col"])): (float(r["cx"]), float(r["cy"]))
        for r in df.to_dict("records")
    }
    for ri, y in enumerate(ROWS_Y, 1):
        for ci, x in enumerate(COLS_X, 1):
            if (ri, ci) in {(2, 2), (2, 3)}:
                continue
            assert by[(ri, ci)] == (float(x), float(y))
    assert by[(2, 2)] == pytest.approx((170.0, U_CY), abs=1e-9)
    # the empty well reports its lattice node, which the coarse fit places within 3 px
    assert by[(2, 3)] == pytest.approx((230.0, 150.0), abs=3.0)


def test_quantify_dark_field_masks_and_overlay(
    dark_plate_path: str, tmp_path: Path
) -> None:
    """``return_masks`` yields (table, det) with det the union of every drawn blob:
    9 * 45 + 77 + 45 (the M extra) + 189 = 716 px. The overlay is the same size as the
    input; the opened gash is painted blue (184 px), the ROI rectangle yellow at its
    corner, green marks the colony outline and cross, red crosses (21 px each) sit only
    at the C and S wells, and a magenta box plus cross (56 + 48 + 21 = 125 px) at the M
    well.
    """
    ov_path = str(tmp_path / "ov.png")
    result: object = quantify_plate_image(
        dark_plate_path, 3, 4, grid_mode="roi", overlay_path=ov_path, return_masks=True
    )
    assert isinstance(result, tuple)
    df, det = result
    assert det.dtype == np.bool_ and det.shape == (300, 400)
    assert int(det.sum()) == 716
    assert int(det[_cell(1, 4)].sum()) == 77 + 45
    assert int(det[_cell(2, 3)].sum()) == 0
    ov = _rgb(ov_path)
    assert ov.shape == (300, 400, 3)
    blue = _where_color(ov, BLUE)
    assert len(blue) == 184
    assert tuple(blue.min(0)) == (220, 150) and tuple(blue.max(0)) == (233, 163)
    assert tuple(ov[35, 41]) == YELLOW
    assert tuple(ov[90, 107]) == GREEN and tuple(ov[90, 110]) == GREEN
    red = _where_color(ov, RED)
    assert len(red) == 42
    near_c = np.hypot(red[:, 0] - U_CY, red[:, 1] - 170) <= 6
    near_s = np.hypot(red[:, 0] - 210, red[:, 1] - 170) <= 6
    assert bool(np.all(near_c | near_s)) and int(near_c.sum()) == 21
    magenta = _where_color(ov, MAGENTA)
    assert len(magenta) == 125
    assert tuple(magenta.min(0)) == (83, 283) and tuple(magenta.max(0)) == (97, 297)
    assert len(_where_color(ov, CYAN)) == 0  # no gel polygon on the roi path
    assert df["size"].tolist()[3] == 77


def test_quantify_multi_min_area_gates_the_m_flag(dark_plate_path: str) -> None:
    """The secondary is 45 px after opening; a ``multi_min_area`` of 46 means no second
    blob clears the threshold and the well is a plain 77 px colony.
    """
    df = quantify_plate_image(dark_plate_path, 3, 4, grid_mode="roi", multi_min_area=46)
    assert df["flags"].tolist()[3] == ""
    assert df["size"].tolist()[3] == 77


def test_quantify_forced_polarity_and_bad_grid_mode(dark_plate_path: str) -> None:
    """Forcing ``polarity="dark"`` on the bright-colony plate finds no dark blobs and
    fails the 20% coverage check; an unknown ``grid_mode`` is rejected by name.
    """
    with pytest.raises(ValueError, match="colony blobs detected for a 3x4 array"):
        quantify_plate_image(dark_plate_path, 3, 4, grid_mode="roi", polarity="dark")
    with pytest.raises(
        ValueError, match="grid_mode must be 'roi' or 'lattice', got 'x'"
    ):
        quantify_plate_image(dark_plate_path, 3, 4, grid_mode="x")


def test_quantify_too_few_blobs_names_the_path(tmp_path: Path) -> None:
    """A featureless backlit image has 0 blobs for a 3x4 array; the message carries the
    count, the array shape, the repr of the path and the lattice hint.
    """
    path = str(tmp_path / "flat.png")
    Image.fromarray(np.full((300, 400), 213, np.uint8), "L").save(path)
    with pytest.raises(ValueError) as ei:
        quantify_plate_image(path, 3, 4, grid_mode="lattice")
    assert str(ei.value) == (
        f"only 0 colony blobs detected for a 3x4 array in {path!r}; the lattice fit "
        "would be unreliable. Try grid_mode='lattice' for a backlit capture."
    )


# --- quantify_plate_image: backlit lattice path --------------------------------------


def test_quantify_backlit_table(backlit_plate_path: str) -> None:
    """Sizes 37 (7x7 under disk morphology), 69 (9x9, flag M), 193 (U, flag C), 0 for
    the empty well; the two bottom-corner colonies carry E (their nodes lie 4.24 px
    outside the chamfered gel edge, inside the 30 px band). The 255-valued gash patch
    raises no S (see the dedicated finding test).
    """
    df = quantify_plate_image(backlit_plate_path, 3, 4, grid_mode="lattice")
    assert df["size"].tolist() == [37, 37, 37, 69, 37, 193, 0, 37, 37, 37, 37, 37]
    assert df["flags"].tolist() == ["", "", "", "M", "", "C", "", "", "E", "", "", "E"]
    circ = df["circularity"].to_numpy()
    assert np.isnan(circ[6])
    assert circ[5] == pytest.approx(0.592117, abs=1e-6)  # numerically pinned
    assert_allclose(np.delete(circ, [5, 6]), 1.0)
    by = {
        (int(r["row"]), int(r["col"])): (float(r["cx"]), float(r["cy"]))
        for r in df.to_dict("records")
    }
    for ri, y in enumerate(ROWS_Y, 1):
        for ci, x in enumerate(COLS_X, 1):
            if (ri, ci) in {(2, 2), (2, 3)}:
                continue
            assert by[(ri, ci)] == (float(x), float(y))
    assert by[(2, 2)][0] == 170.0
    assert by[(2, 2)][1] == pytest.approx(150.466321, abs=1e-6)  # numerically pinned
    assert by[(2, 3)] == pytest.approx((230.0, 150.0), abs=3.0)


def test_quantify_backlit_gash_flag_cannot_fire(backlit_plate_path: str) -> None:
    """Finding: in the inverted (dark-colony) branch the tear threshold is
    ``g > min(255, 1.30 * agar_median)``. With agar at 213 that is ``g > 255``, which
    no 8-bit pixel satisfies (any agar median above 255 / 1.3 = 196.2 disables the
    gate), so the bright 255 tear patch in the (3, 2) cell (6.3% of the cell) never
    produces an S flag on the backlit plate, while the same geometry on the dark-field
    plate does flag S (``test_quantify_dark_field_table``).
    """
    df = quantify_plate_image(backlit_plate_path, 3, 4, grid_mode="lattice")
    assert df["flags"].tolist()[9] == ""
    assert not any("S" in f for f in df["flags"])
    assert _grayscale(backlit_plate_path)[220:234, 150:164].min() == 255.0


def test_quantify_backlit_edge_policy_drop_removes_the_corner_colonies(
    backlit_plate_path: str,
) -> None:
    """Finding: with the default gel geometry (margin 0.6 pitch, bottom chamfers of
    1.3 pitch) the bottom-corner nodes sit 0.0707 pitch outside the gel polygon. Under
    ``edge_policy="drop"`` a colony needs ``sd >= +0.5 pitch``, so the two bottom-corner
    colonies of any lattice-mode plate are recorded as absent (size 0, no flag) while
    every other well is unchanged.
    """
    df = quantify_plate_image(
        backlit_plate_path, 3, 4, grid_mode="lattice", edge_policy="drop"
    )
    assert df["size"].tolist() == [37, 37, 37, 69, 37, 193, 0, 37, 0, 37, 37, 0]
    assert df["flags"].tolist() == ["", "", "", "M", "", "C", "", "", "", "", "", ""]
    assert np.isnan(df["circularity"].to_numpy()[[6, 8, 11]]).all()


def test_quantify_backlit_without_gel_has_no_edge_flags(
    backlit_plate_path: str,
) -> None:
    """``gel_detect=False`` skips the polygon: same sizes, no E anywhere, and the
    overlay carries no cyan gel outline.
    """
    df = quantify_plate_image(
        backlit_plate_path, 3, 4, grid_mode="lattice", gel_detect=False
    )
    assert df["size"].tolist() == [37, 37, 37, 69, 37, 193, 0, 37, 37, 37, 37, 37]
    assert df["flags"].tolist() == ["", "", "", "M", "", "C", "", "", "", "", "", ""]


def test_quantify_backlit_masks_diverge_from_table_at_the_gel_edge(
    backlit_plate_path: str, tmp_path: Path
) -> None:
    """Finding: the docstring says one acceptance predicate keeps ``det`` and the table
    from diverging, but the final ``det &= gel_mask`` clips the E-flagged corner
    colonies to the pixels inside the polygon while the table keeps size 37. At (3, 1)
    none of the 37 px survive: the chamfer line passes through the removed corner pixel
    of the opened square, and the nearest surviving pixel is 3.6 px from the node while
    the line is 4.24 px out. At (3, 4) 7 px survive (pinned numerically): the coarse
    even-spacing fit places the node columns 3.66 px right of the true columns, so the
    right-hand chamfer is offset from its colony and clips it less. Interior wells agree
    exactly (37; M well 69 + 37).
    """
    ov_path = str(tmp_path / "ovl.png")
    result: object = quantify_plate_image(
        backlit_plate_path,
        3,
        4,
        grid_mode="lattice",
        overlay_path=ov_path,
        return_masks=True,
    )
    assert isinstance(result, tuple)
    df, det = result
    assert det.dtype == np.bool_ and det.shape == (300, 400)
    assert int(det[_cell(1, 1)].sum()) == 37
    assert int(det[_cell(1, 4)].sum()) == 69 + 37
    assert int(det[_cell(2, 2)].sum()) == 193
    assert int(det[_cell(2, 3)].sum()) == 0
    sizes = df["size"].tolist()
    assert (sizes[8], sizes[11]) == (37, 37)
    assert int(det[_cell(3, 1)].sum()) == 0
    assert int(det[_cell(3, 4)].sum()) == 7  # pinned numerically
    ov = _rgb(ov_path)
    assert len(_where_color(ov, BLUE)) == 0  # no gash region in inverted mode
    assert len(_where_color(ov, CYAN)) > 0  # the gel hexagon is drawn
    assert tuple(ov[48, 68]) == YELLOW  # ROI corner from the backlit detector
    red = _where_color(ov, RED)
    assert len(red) == 63  # C + two E crosses, 21 px each
    assert len(_where_color(ov, MAGENTA)) == 125


def test_quantify_backlit_watershed_finding(backlit_plate_path: str) -> None:
    """Finding: ``seg_method="watershed"`` on flat synthetic agar returns size 0 for
    every colony except the gash cell (3, 2): only there does the 255 patch make the
    ``cell > p55`` reference non-empty (spread 0 -> the ``or 1.0`` fallback fires), and
    the watershed basin of the 7x7 colony is 77 px (pinned numerically: the sobel ridge
    of the blurred step is a two-pixel tie, so the basin edge is a flooding-order
    tie-break; see ``test_segment_watershed_floods_one_pixel_past_the_step``).
    Everywhere else the NaN spread
    (``test_segment_watershed_returns_empty_mask_on_constant_agar``) silently empties
    the well.
    """
    df = quantify_plate_image(
        backlit_plate_path, 3, 4, grid_mode="lattice", seg_method="watershed"
    )
    assert df["size"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0]


def test_quantify_backlit_disk_colony_is_preserved(
    backlit_disk_plate_path: str,
) -> None:
    """The radius-15 digital disk (709 px) is a union of disk(2) translates, so the
    disk(4) closing and disk(2) opening leave it intact: size 709, circularity clamped
    to 1.0 (4*pi*709/84^2 = 1.26), centroid exactly on the node.
    """
    df = quantify_plate_image(backlit_disk_plate_path, 3, 4, grid_mode="lattice")
    row = df.iloc[10]
    assert (int(row["row"]), int(row["col"])) == (3, 3)
    assert int(row["size"]) == 709
    assert float(row["circularity"]) == 1.0
    assert (float(row["cx"]), float(row["cy"])) == (230.0, 210.0)


# --- 2026.10.06 (Phase 21): blob-filter edge cases and the watershed bright branch ---


def test_detect_blobs_crashes_when_every_blob_is_filtered(tmp_path: Path) -> None:
    """Finding: ``_detect_blobs`` has no guard for an empty ``keep`` list (the backlit
    detector has one). Two 4x4 specks clear the threshold (n = 2) but after the cross
    opening each is 12 px (16 minus 4 corners), below the 25 px floor, so ``center_of_mass(..., [])`` gives a 1-D empty array and
    ``cents[:, 0]`` raises ``IndexError`` instead of returning no blobs; through
    ``quantify_plate_image`` this replaces the actionable "only 0 colony blobs" refusal
    with a numpy error. Pinned until an empty ``keep`` returns ``(empty (0, 2), 60.0)``
    (image.py:126-132). Reach: an empty or speck-only plate in the default
    ``grid_mode="roi"`` (the W019 callers) crashes; no wrong number is produced.
    """
    g = np.zeros((300, 400), float)
    g[20:280, 20:380] = 60.0
    g[100:104, 150:154] = 200.0
    g[200:204, 250:254] = 200.0
    roi = (35, 264, 41, 358)
    with pytest.raises(
        IndexError,
        match=re.escape(
            "too many indices for array: array is 1-dimensional, but 2 were indexed"
        ),
    ):
        _detect_blobs(g, np.zeros_like(g), roi, False, 4)
    path = str(tmp_path / "specks.png")
    Image.fromarray(g.astype(np.uint8), "L").save(path)
    with pytest.raises(
        IndexError,
        match=re.escape(
            "too many indices for array: array is 1-dimensional, but 2 were indexed"
        ),
    ):
        quantify_plate_image(path, 3, 4, grid_mode="roi")


def test_detect_blobs_crashes_when_every_kept_blob_hugs_the_wall() -> None:
    """Finding: one 7x7 colony whose centroid (y = 39) is within the 0.03 * 317 = 9.5 px
    wall band of the ROI top (35) is dropped, leaving a (0, 2) array whose
    nearest-neighbour ``min`` has no identity: ``ValueError`` rather than an empty
    detection. Pinned with the empty-``keep`` Finding (image.py:137-141). Same
    reach: a crash in ``grid_mode="roi"``, never a wrong number.
    """
    g = np.zeros((300, 400), float)
    g[20:280, 20:380] = 60.0
    g[36:43, 150:157] = 200.0
    with pytest.raises(
        ValueError,
        match=re.escape(
            "zero-size array to reduction operation minimum which has no identity"
        ),
    ):
        _detect_blobs(g, np.zeros_like(g), (35, 264, 41, 358), False, 4)


def test_detect_blobs_backlit_drops_blobs_below_the_area_floor() -> None:
    """A 4x4 depression (12 px after the cross opening, <= 20) is detected by the mask but filtered by area:
    no kept blob -> empty centroids, the 60 px default pitch and the whole frame.
    """
    g = np.full((300, 400), 213.0)
    g[150:154, 200:204] = 150.0
    cents, pitch, roi = _detect_blobs_backlit(g, 4)
    assert cents.shape == (0, 2)
    assert (pitch, roi) == (60.0, (0, 300, 0, 400))


def test_detect_blobs_backlit_single_colony_has_infinite_pitch() -> None:
    """One 7x7 colony: its nearest-neighbour distance is +inf (diagonal filled), so the
    median pitch is +inf and the ``nn < 1.8 * pitch`` filter (inf < inf) drops the only
    colony: an empty array, pitch inf, whole-frame ROI.
    """
    g = np.full((300, 400), 213.0)
    g[147:154, 197:204] = 150.0
    cents, pitch, roi = _detect_blobs_backlit(g, 4)
    assert cents.shape == (0, 2)
    assert pitch == float("inf")
    assert roi == (0, 300, 0, 400)


def test_segment_watershed_bright_branch_mirrors_the_dark_branch() -> None:
    """Structural identity: every quantity is symmetric under ``x -> 255 - x``
    (percentiles 80/20 and 55/45 swap, the MAD and the Sobel magnitude are unchanged),
    so a bright colony on dark agar segments to exactly the dark-branch mask of the
    mirrored cell (77 px, rows/cols 22..30).
    """
    cell = _backlit_cell()
    cell[0, 0] = 214.0
    dark = _segment_watershed(cell, True, PITCH)
    bright = _segment_watershed(255.0 - cell, False, PITCH)
    assert_array_equal(bright, dark)
    assert int(bright.sum()) == 77
