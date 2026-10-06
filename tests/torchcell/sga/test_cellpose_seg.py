# tests/torchcell/sga/test_cellpose_seg.py
# [[tests.torchcell.sga.test_cellpose_seg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_cellpose_seg.py
"""Cellpose-SAM plate quantification without a Cellpose model.

``quantify_plate_image_cellpose`` takes ``precomputed_masks`` and ``model=None``, so the
grid fit, gel gate, well assignment, multi/neighbor invalidation, tightening and
grid-guided recovery all run on the synthetic backlit plate of ``conftest.py`` with the
instance labels Cellpose would have returned (``plate_masks``). No weights are loaded;
``load_cellpose_model`` is exercised against a monkeypatched ``CellposeModel``.

Hand-derived expectations on the 60 px pitch (nodes y = 90, 150, 210; x = 110, 170,
230, 290):

* raw instance areas are the drawn shapes: 7x7 = 49, 9x9 = 81 (boundary 32, circularity
  4*pi*81/32^2 = 0.99402), U = 195 (boundary 82, circularity 0.36443 < 0.65 -> ``C``);
* the M secondary (49 px) is 25.46 px from the 81 px primary: above the 0.30 * 60 = 18
  px separation and above 0.35 * 81 = 28.35 px in area -> ``M``; its id is colored
  ``X``;
* neighbor invalidation: the well below the M well, (2, 4) at (150, 290), is
  sqrt(42^2 + 18^2) = 45.7 px from the secondary at (108, 308), under 0.85 * 60 = 51 px,
  so it is flagged ``N``; the well to the left is 60 px away and stays clean;
* bottom-corner nodes lie 0.0707 pitch outside the chamfered gel edge -> ``E``;
* recovery: the radius-15 digital disk (709 px) at an empty node has a depression of
  213 - 190 = 23 gray levels > 12; Otsu inside the 0.46-pitch window recovers the disk
  and ``tighten_grow_px=3`` dilates it to 1001 px (pinned numerically).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from PIL import Image

from torchcell.sga.cellpose_seg import (
    _CATEGORY_COLOR,
    _INSTANCE_COLORS,
    CellposeSegConfig,
    PlateSegResult,
    _apply_homography,
    _contrast_enhance,
    _draw_cellpose_overlay,
    _fit_homography,
    _fit_lattice,
    _homography_lattice,
    _instance_props,
    _recover_colony,
    _relax_lattice,
    _snap_edge_row,
    _tighten_instance,
    _well,
    load_cellpose_model,
    quantify_plate_image_cellpose,
)
from torchcell.sga.image import MIN_COLONY_AREA, _grayscale

ROWS_Y = (90, 150, 210)
COLS_X = (110, 170, 230, 290)
TRUE_NODES = np.stack(np.meshgrid(ROWS_Y, COLS_X, indexing="ij"), -1).astype(float)
CFG = CellposeSegConfig(n_rows=3, n_cols=4)
U_CY = 150 + 1590 / 195 - 8  # raw U centroid: 0.153846 px below its node
EXPECTED_SIZES = [49, 49, 49, 81, 49, 195, 0, 49, 49, 49, 49, 49]
EXPECTED_FLAGS = ["", "", "", "M", "", "C", "", "N", "E", "", "", "E"]


def _rgb(path: str) -> NDArray[Any]:
    return np.asarray(Image.open(path).convert("RGB"))


def _count_color(ov: NDArray[Any], color: tuple[int, int, int]) -> int:
    return int(np.all(ov == np.array(color, dtype=ov.dtype), axis=-1).sum())


# --- config + records -----------------------------------------------------------------


def test_config_defaults_match_the_classical_gates() -> None:
    """The defaults that the assignment logic reads: a 14x22 array, node tolerance
    0.55 pitch, stray 1.0 pitch, circularity flag 0.80 / reject 0.65, and the minimum
    instance area is the shared 20 px colony floor.
    """
    cfg = CellposeSegConfig()
    assert (cfg.n_rows, cfg.n_cols) == (14, 22)
    assert (cfg.node_tol, cfg.stray_tol) == (0.55, 1.0)
    assert (cfg.circularity_flag, cfg.circularity_reject) == (0.80, 0.65)
    assert cfg.min_instance_area == MIN_COLONY_AREA == 20
    assert (cfg.multi_min_frac, cfg.multi_min_sep_frac) == (0.35, 0.30)
    assert (cfg.tighten_grow_px, cfg.recover_depth_thresh) == (3, 12.0)
    assert cfg.polarity == "auto" and cfg.contrast == "none" and cfg.diameter is None


def test_plate_seg_result_holds_arrays() -> None:
    """Arbitrary types are allowed: the table and the mask array round-trip by identity
    and the id->category map defaults to empty.
    """
    import pandas as pd

    tbl = pd.DataFrame({"row": [1]})
    masks = np.zeros((2, 2), np.int32)
    res = PlateSegResult(table=tbl, n_instances=3, n_offgrid=1, masks=masks)
    assert res.table is tbl and res.masks is masks
    assert (res.n_instances, res.n_offgrid, res.kept_color, res.nodes) == (
        3,
        1,
        {},
        None,
    )


def test_well_record() -> None:
    """1-based row/col, the nine keys in schema order, defaults '' detector."""
    rec = _well(0, 3, 49, 1.0, "M", 290.0, 90.0, 4)
    assert rec == {
        "row": 1,
        "col": 4,
        "size": 49,
        "circularity": 1.0,
        "flags": "M",
        "cx": 290.0,
        "cy": 90.0,
        "id": 4,
        "detector": "",
    }
    assert (
        _well(2, 2, 0, float("nan"), "", 0.0, 0.0, -1, "recovered")["detector"]
        == "recovered"
    )


def test_category_and_instance_colors() -> None:
    """Six categories with the documented RGB values (green accepted, red multi, orange
    neighbor, purple non-circular, blue recovered, deep pink collision partner); twelve
    green-free instance fills.
    """
    assert _CATEGORY_COLOR == {
        "": (0, 255, 0),
        "M": (255, 0, 0),
        "N": (255, 140, 0),
        "C": (170, 0, 255),
        "R": (0, 128, 255),
        "X": (255, 20, 147),
    }
    assert _INSTANCE_COLORS.shape == (12, 3)
    assert not any(tuple(c) == (0.0, 255.0, 0.0) for c in _INSTANCE_COLORS)


def test_load_cellpose_model_constructs_cpsam(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader is ``models.CellposeModel(gpu=gpu)`` and nothing else; a stand-in
    class records the call so no weights are touched. Skipped where ``cellpose`` is
    not installed (the CI runner); every other test in this file runs without it.
    """
    cm = pytest.importorskip("cellpose.models")

    calls: list[dict[str, Any]] = []

    class Fake:
        def __init__(self, **kw: Any) -> None:
            calls.append(kw)

    monkeypatch.setattr(cm, "CellposeModel", Fake)
    model = load_cellpose_model(gpu=False)
    assert isinstance(model, Fake)
    assert calls == [{"gpu": False}]


# --- geometry helpers -----------------------------------------------------------------


def test_fit_lattice_prologue(backlit_plate_path: str) -> None:
    """Pitch 60, dark colonies, no tilt, center = ROI center ((48 + 252) / 2,
    (68 + 350) / 2) = (150, 209), nodes within 4 px of the true grid (the coarse
    even-spacing search is pulled by the off-lattice M secondary).
    """
    g = _grayscale(backlit_plate_path)
    nodes, pitch, invert, theta, center, roi = _fit_lattice(g, 3, 4, "auto")
    assert (pitch, invert, theta) == (60.0, True, 0.0)
    assert_array_equal(center, [150.0, 209.0])
    assert roi == (48, 252, 68, 350)
    assert nodes.shape == (3, 4, 2)
    assert np.abs(nodes - TRUE_NODES).max() < 4.0
    assert _fit_lattice(g, 3, 4, "bright")[2] is False


def test_fit_lattice_rejects_a_featureless_plate() -> None:
    """Zero blobs for a 3x4 array (< 20% of 12) is an error naming the count."""
    with pytest.raises(
        ValueError, match="only 0 colony blobs detected for a 3x4 array"
    ):
        _fit_lattice(np.full((300, 400), 213.0), 3, 4, "auto")


def test_relax_lattice_snaps_lines_to_centroid_medians() -> None:
    """Even grid rows 10/20/30, cols 5/15/25/35; colonies on rows 10/21/33 (4 per row
    >= the row minimum of 4, 3 per column >= 3): every row line moves to the colony
    median exactly and the columns stay. With no colony on the third row it is
    extrapolated from the two snapped rows: 10, 21 -> 32.
    """
    ii, jj = np.meshgrid([10.0, 20.0, 30.0], [5.0, 15.0, 25.0, 35.0], indexing="ij")
    nodes = np.stack([ii, jj], -1)
    ci, cj = np.meshgrid([10.0, 21.0, 33.0], [5.0, 15.0, 25.0, 35.0], indexing="ij")
    cents = np.stack([ci.ravel(), cj.ravel()], 1)
    out = _relax_lattice(nodes, cents, 3, 4, 0.0, np.array([20.0, 20.0]))
    assert_allclose(out[:, :, 0], np.tile([[10.0], [21.0], [33.0]], (1, 4)), atol=1e-9)
    assert_allclose(out[:, :, 1], np.tile([5.0, 15.0, 25.0, 35.0], (3, 1)), atol=1e-9)
    out2 = _relax_lattice(
        nodes, cents[cents[:, 0] < 30], 3, 4, 0.0, np.array([20.0, 20.0])
    )
    assert_allclose(out2[:, 0, 0], [10.0, 21.0, 32.0], atol=1e-9)


def test_fit_homography_recovers_a_scale_and_shift() -> None:
    """Points mapped by [[2, 0, 5], [0, 2, 7], [0, 0, 1]] fit that matrix back exactly
    (normalized so h[2, 2] = 1); a general projective matrix round-trips too.
    """
    h_true = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0], [0.0, 0.0, 1.0]])
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.2]])
    dst = _apply_homography(h_true, src)
    expected = np.array([[5.0, 7.0], [7.0, 7.0], [7.0, 9.0], [5.0, 9.0], [6.0, 7.4]])
    assert_allclose(dst, expected, atol=1e-12)
    assert_allclose(_fit_homography(src, dst), h_true, atol=1e-9)
    hp = np.array([[1.0, 0.1, 3.0], [0.05, 1.2, -2.0], [0.001, 0.002, 1.0]])
    assert_allclose(_fit_homography(src, _apply_homography(hp, src)), hp, atol=1e-8)


def test_apply_homography_clamps_the_point_at_infinity() -> None:
    """A third row [1, 0, 0] sends x = 0 to w = 0; the clamp to 1e-12 keeps the result
    finite (and huge) instead of NaN.
    """
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    out = _apply_homography(h, np.array([[0.0, 5.0], [2.0, 4.0]]))
    assert np.isfinite(out).all()
    assert out[0, 1] == 5.0 / 1e-12
    assert_allclose(out[1], [1.0, 2.0])


def test_homography_lattice_reproduces_a_projective_grid() -> None:
    """A 4x4 index grid mapped by x = 100 + 30 j + 3 i, y = 100 + 30 i (a shear the
    isotropic pitch cannot express) is recovered to 1e-9 from the even 30 px grid: all
    16 colonies are inliers of the exact fit. With only 9 colonies (< 12) the grid is
    returned untouched.
    """
    ii, jj = np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij")
    nodes = np.stack([100 + 30 * ii, 100 + 30 * jj], -1)
    idx = np.stack([jj.ravel(), ii.ravel()], 1)
    h = np.array([[30.0, 3.0, 100.0], [0.0, 30.0, 100.0], [0.0, 0.0, 1.0]])
    px = _apply_homography(h, idx)
    cents = np.stack([px[:, 1], px[:, 0]], 1)
    out = _homography_lattice(nodes, cents, 4, 4, 30.0)
    assert_allclose(out.reshape(-1, 2), cents, atol=1e-9)
    small = _homography_lattice(nodes[:3, :3], cents[:9], 3, 3, 30.0)
    assert_array_equal(small, nodes[:3, :3])


def test_snap_edge_row_shifts_only_when_a_full_lane_is_gained() -> None:
    """3x8 grid at 10 px pitch. Colonies one row DOWN (y 10/20/30): the as-is grid
    assigns 16, the -1 shift 24, gain 8 >= max(0.25 * 8, 6) -> shifted grid with the
    extrapolated bottom row at y = 30. Colonies one row UP -> the +1 shift (top row
    extrapolated to -10). Colonies on the grid -> unchanged. On a 3x4 grid the same
    slip gains only 4 < 6 and is refused.
    """
    ii, jj = np.meshgrid(np.arange(3.0), np.arange(8.0), indexing="ij")
    n38 = np.stack([10 * ii, 10 * jj], -1)
    down = (n38 + np.array([10.0, 0.0])).reshape(-1, 2)
    assert_allclose(_snap_edge_row(n38, down, 3, 8, 10.0, 0.55)[:, 0, 0], [10, 20, 30])
    up = (n38 - np.array([10.0, 0.0])).reshape(-1, 2)
    assert_allclose(_snap_edge_row(n38, up, 3, 8, 10.0, 0.55)[:, 0, 0], [-10, 0, 10])
    assert_array_equal(_snap_edge_row(n38, n38.reshape(-1, 2), 3, 8, 10.0, 0.55), n38)
    ii, jj = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    n34 = np.stack([10 * ii, 10 * jj], -1)
    down34 = (n34 + np.array([10.0, 0.0])).reshape(-1, 2)
    assert_array_equal(_snap_edge_row(n34, down34, 3, 4, 10.0, 0.55), n34)


def test_snap_edge_row_column_shift() -> None:
    """8x3 grid, colonies one column right: the -1 column shift is accepted (gain 8)
    and the new right edge is extrapolated to x = 30.
    """
    ii, jj = np.meshgrid(np.arange(8.0), np.arange(3.0), indexing="ij")
    n83 = np.stack([10 * ii, 10 * jj], -1)
    right = (n83 + np.array([0.0, 10.0])).reshape(-1, 2)
    assert_allclose(_snap_edge_row(n83, right, 8, 3, 10.0, 0.55)[0, :, 1], [10, 20, 30])


# --- pixel helpers --------------------------------------------------------------------


def test_instance_props_geometry() -> None:
    """A 3x9 rectangle: area 27, centroid (2, 5), aspect 3, extent 1, circularity
    4*pi*27/20^2 = 0.84823 (boundary 27 - 7 interior = 20). A 1x2 instance (2 px) is
    below ``min_area`` 3 and dropped; a 3x3 block has circularity clamped to 1.0.
    """
    lab = np.zeros((10, 12), np.int32)
    lab[1:4, 1:10] = 1
    lab[6, 2:4] = 2
    lab[6:9, 7:10] = 3
    props = _instance_props(lab, 3)
    assert [p["id"] for p in props] == [1, 3]
    p1 = props[0]
    assert (p1["area"], p1["cy"], p1["cx"], p1["aspect"], p1["extent"]) == (
        27,
        2.0,
        5.0,
        3.0,
        1.0,
    )
    assert p1["circ"] == pytest.approx(4 * np.pi * 27 / 400, rel=1e-12)
    assert (props[1]["area"], props[1]["cy"], props[1]["cx"], props[1]["circ"]) == (
        9,
        7.0,
        8.0,
        1.0,
    )


def test_tighten_instance_keeps_the_dark_core() -> None:
    """11x11 mask (121 px) whose inner 7x7 is 100 (colony) and halo 200: Otsu splits
    the two levels, the 49 px core clears max(0.2 * 121, 20) = 24.2 and the halo is
    zeroed. ``grow_px=1`` dilates the core by the cross (49 + 4 * 7 = 77). With
    ``min_frac=0.5`` the core (49 < 60.5) counts as over-shrunk: 121 returned, mask
    untouched.
    """
    masks = np.zeros((20, 20), np.int32)
    masks[3:14, 3:14] = 1
    g = np.full((20, 20), 200.0)
    g[5:12, 5:12] = 100.0
    m0 = masks.copy()
    assert _tighten_instance(m0, 1, g, True, 0.2, 0) == 49
    assert int((m0 == 1).sum()) == 49 and m0[5:12, 5:12].all()
    m1 = masks.copy()
    assert _tighten_instance(m1, 1, g, True, 0.2, 1) == 77
    assert int((m1 == 1).sum()) == 77
    m2 = masks.copy()
    assert _tighten_instance(m2, 1, g, True, 0.5, 0) == 121
    assert_array_equal(m2, masks)


def test_tighten_instance_degenerate_cases() -> None:
    """Absent id -> 0; fewer than 10 pixels -> area unchanged; a constant-intensity
    instance (no Otsu split) -> area unchanged, mask untouched.
    """
    masks = np.zeros((10, 10), np.int32)
    masks[1:4, 1:4] = 1
    masks[5:9, 5:9] = 2
    g = np.full((10, 10), 150.0)
    assert _tighten_instance(masks.copy(), 7, g, True, 0.2) == 0
    assert _tighten_instance(masks.copy(), 1, g, True, 0.2) == 9
    m = masks.copy()
    assert _tighten_instance(m, 2, g, True, 0.2) == 16
    assert_array_equal(m, masks)


def test_recover_colony_finds_a_filled_well() -> None:
    """Agar 213 with a radius-15 disk at 190 centered on the node, pitch 60: r_out =
    int(0.78 * 60) = 46 so the crop starts at 100 - 46 = 54 and is 93x93; the annulus
    median is 213 and the core 25th percentile 190 -> depth 23 > 12; Otsu on the
    0.46-pitch window selects the 709 px digital disk (circularity clamped: 4*pi*709/84^2
    = 1.26). A 205-valued disk (depth 8) and bare agar return None.
    """
    g = np.full((200, 200), 213.0)
    y, x = np.ogrid[-100:100, -100:100]
    disk = (x * x + y * y) <= 225
    g[disk] = 190.0
    rec = _recover_colony(g, 100, 100, 60.0, True, 12.0, 0)
    assert rec is not None
    size, circ, cc, y0, x0 = rec
    assert (size, circ, y0, x0, cc.shape) == (709, 1.0, 54, 54, (93, 93))
    assert_array_equal(cc, disk[54:147, 54:147])
    shallow = np.full((200, 200), 213.0)
    shallow[disk] = 205.0
    assert _recover_colony(shallow, 100, 100, 60.0, True, 12.0, 0) is None
    assert (
        _recover_colony(np.full((200, 200), 213.0), 100, 100, 60.0, True, 12.0, 0)
        is None
    )


def test_recover_colony_grow_dilates_the_core() -> None:
    """``grow_px=3`` dilates the 709 px disk by disk(3) inside the window: 1001 px
    (pinned numerically; the window radius 27.6 does not bind).
    """
    g = np.full((200, 200), 213.0)
    y, x = np.ogrid[-100:100, -100:100]
    g[(x * x + y * y) <= 225] = 190.0
    rec = _recover_colony(g, 100, 100, 60.0, True, 12.0, 3)
    assert rec is not None
    assert (rec[0], rec[1]) == (1001, 1.0)


def test_contrast_enhance_restacks_luminance() -> None:
    """Both methods return an HxWx3 uint8 with identical channels; the 150-valued spot
    stays darker than the 213 field after either stretch; an unknown method is named.
    """
    img = np.full((40, 40, 3), 213, np.uint8)
    img[18:22, 18:22] = 150
    for method in ("clahe", "flatfield"):
        out = _contrast_enhance(img, method, 0.01)
        assert out.shape == (40, 40, 3) and out.dtype == np.uint8
        assert_array_equal(out[..., 0], out[..., 1])
        assert_array_equal(out[..., 0], out[..., 2])
        assert int(out[20, 20, 0]) < int(out[0, 0, 0])
    fe = _contrast_enhance(img, "flatfield", 0.01)
    assert (int(fe.min()), int(fe.max())) == (
        0,
        255,
    )  # rescale_intensity spans the range
    with pytest.raises(
        ValueError, match="contrast must be 'none'\\|'clahe'\\|'flatfield', got 'x'"
    ):
        _contrast_enhance(img, "x", 0.01)


def test_draw_cellpose_overlay_colors_thick_boundaries(tmp_path: Path) -> None:
    """A 7x7 instance drawn 'thick' colors its own 24 border pixels plus the 28
    4-adjacent outside pixels = 52; the category picks the color (green accepted, red
    M); an instance absent from ``kept_color`` is not drawn; interior pixels keep the
    original gray.
    """
    src = str(tmp_path / "src.png")
    Image.fromarray(np.full((20, 20), 128, np.uint8), "L").save(src)
    masks = np.zeros((20, 20), np.int32)
    masks[5:12, 5:12] = 1
    masks[14:17, 14:17] = 2
    import pandas as pd

    out = str(tmp_path / "out.png")
    _draw_cellpose_overlay(src, masks, {1: ""}, pd.DataFrame(), out)
    ov = _rgb(out)
    assert _count_color(ov, (0, 255, 0)) == 52
    assert tuple(ov[8, 8]) == (128, 128, 128) and tuple(ov[4, 8]) == (0, 255, 0)
    assert tuple(ov[15, 15]) == (128, 128, 128) and tuple(ov[13, 15]) == (128, 128, 128)
    _draw_cellpose_overlay(src, masks, {1: "M", 2: "X"}, pd.DataFrame(), out)
    ov = _rgb(out)
    assert _count_color(ov, (255, 0, 0)) == 52
    assert _count_color(ov, (255, 20, 147)) == 8 + 12  # 3x3 border + 4-adjacent ring


# --- the full pipeline on precomputed masks -------------------------------------------


def test_quantify_cellpose_table(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """Sizes are the raw shape areas (no halo to tighten), flags M / C / N / E / E as
    derived in the module docstring, ``detector`` is 'cellpose' on occupied wells and ''
    on the empty one, measured centroids equal the shape centroids, and the empty well
    reports its (relaxed) node within 0.1 px of (230, 150).
    """
    res = quantify_plate_image_cellpose(
        backlit_plate_path, None, cfg=CFG, precomputed_masks=plate_masks.copy()
    )
    df = res.table
    assert list(df.columns) == [
        "row",
        "col",
        "size",
        "circularity",
        "flags",
        "cx",
        "cy",
        "detector",
    ]
    assert df["size"].tolist() == EXPECTED_SIZES
    assert df["flags"].tolist() == EXPECTED_FLAGS
    assert df["detector"].tolist() == ["cellpose"] * 6 + [""] + ["cellpose"] * 5
    circ = df["circularity"].to_numpy()
    assert np.isnan(circ[6])
    assert circ[3] == pytest.approx(4 * np.pi * 81 / 32**2, rel=1e-12)
    assert circ[5] == pytest.approx(4 * np.pi * 195 / 82**2, rel=1e-12)
    assert_allclose(np.delete(circ, [3, 5, 6]), 1.0)
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
    assert by[(2, 3)] == pytest.approx((230.0, 150.0), abs=0.1)
    assert (res.n_instances, res.n_offgrid) == (12, 0)
    assert res.kept_color == {
        1: "",
        2: "",
        3: "",
        4: "M",
        5: "X",
        6: "",
        7: "C",
        8: "N",
        9: "",
        10: "",
        11: "",
        12: "",
    }
    assert res.nodes is not None
    assert np.abs(res.nodes - TRUE_NODES).max() < 0.1
    assert res.masks is None


def test_quantify_cellpose_masks_and_overlay(
    backlit_plate_path: str, plate_masks: NDArray[Any], tmp_path: Path
) -> None:
    """With uniform colony intensity Otsu has nothing to split, so the returned masks
    equal the input. The overlay paints each kept instance's thick boundary in its
    category color: green just above the (1, 1) colony, red above the M primary, deep
    pink above the secondary, orange above the N well, purple above the U; interiors
    keep the colony gray 190.
    """
    out = str(tmp_path / "cp.png")
    res = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CFG,
        overlay_path=out,
        precomputed_masks=plate_masks.copy(),
        return_masks=True,
    )
    assert res.masks is not None
    assert_array_equal(res.masks, plate_masks)
    ov = _rgb(out)
    assert ov.shape == (300, 400, 3)
    assert tuple(ov[86, 110]) == (0, 255, 0)
    assert tuple(ov[85, 290]) == (255, 0, 0)
    assert tuple(ov[104, 308]) == (255, 20, 147)
    assert tuple(ov[146, 290]) == (255, 140, 0)
    assert tuple(ov[141, 161]) == (170, 0, 255)
    assert tuple(ov[90, 110]) == (190, 190, 190)
    assert _count_color(ov, (0, 128, 255)) == 0  # nothing recovered on this plate


def test_quantify_cellpose_offgrid_and_no_gel(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """An extra 7x7 instance at (250, 60) is farther than 1.0 pitch from every node: it
    counts as an off-grid contaminant, not a well. Without the gel polygon the corner
    wells lose their E flags and the plate reads 13 instances, 1 off-grid.
    """
    m = plate_masks.copy()
    m[247:254, 57:64] = 99
    cfg = CellposeSegConfig(n_rows=3, n_cols=4, gel_detect=False)
    res = quantify_plate_image_cellpose(
        backlit_plate_path, None, cfg=cfg, precomputed_masks=m
    )
    assert (res.n_instances, res.n_offgrid) == (13, 1)
    assert res.table["size"].tolist() == EXPECTED_SIZES
    assert res.table["flags"].tolist() == [
        "",
        "",
        "",
        "M",
        "",
        "C",
        "",
        "N",
        "",
        "",
        "",
        "",
    ]
    assert 99 not in res.kept_color


def test_quantify_cellpose_switches(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """``relax_grid=False`` keeps the coarse even grid (still within the 0.55-pitch node
    tolerance, same table); ``neighbor_invalidate=False`` drops only the N flag.
    """
    res = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, relax_grid=False),
        precomputed_masks=plate_masks.copy(),
    )
    assert res.table["size"].tolist() == EXPECTED_SIZES
    assert res.table["flags"].tolist() == EXPECTED_FLAGS
    assert res.nodes is not None and np.abs(res.nodes - TRUE_NODES).max() > 1.0
    res2 = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, neighbor_invalidate=False),
        precomputed_masks=plate_masks.copy(),
    )
    assert res2.table["flags"].tolist() == [
        "",
        "",
        "",
        "M",
        "",
        "C",
        "",
        "",
        "E",
        "",
        "",
        "E",
    ]
    assert res2.kept_color[8] == ""


def test_quantify_cellpose_multi_min_frac_gates_m(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """The secondary is 49 / 81 = 0.605 of the primary: ``multi_min_frac=0.61`` makes it
    a mere fragment, so the M well and its N neighbor are both clean and no X partner
    is recorded.
    """
    res = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, multi_min_frac=0.61),
        precomputed_masks=plate_masks.copy(),
    )
    assert res.table["flags"].tolist() == [
        "",
        "",
        "",
        "",
        "",
        "C",
        "",
        "",
        "E",
        "",
        "",
        "E",
    ]
    assert 5 not in res.kept_color


def test_quantify_cellpose_tightens_a_haloed_mask(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """Instance 1 enlarged to 13x13 (169 px: the 7x7 colony at 190 plus a 3 px agar
    halo at 213). Otsu splits the levels; ``tighten_grow_px=0`` stores the 49 px core and
    zeroes the halo in the returned masks; the default grow of 3 dilates the core by
    disk(3) inside the 13x13: rows of 7, 11, 11, seven rows of 13, 11, 11, 7, so
    149 = 2 * (7 + 11 + 11) + 7 * 13 px.
    """
    m = plate_masks.copy()
    m[84:97, 104:117] = 1
    res0 = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, tighten_grow_px=0),
        precomputed_masks=m.copy(),
        return_masks=True,
    )
    assert res0.table["size"].tolist()[0] == 49
    assert res0.masks is not None and int((res0.masks == 1).sum()) == 49
    res3 = quantify_plate_image_cellpose(
        backlit_plate_path, None, cfg=CFG, precomputed_masks=m.copy(), return_masks=True
    )
    assert res3.table["size"].tolist()[0] == 149
    assert res3.masks is not None and int((res3.masks == 1).sum()) == 149
    off = quantify_plate_image_cellpose(
        backlit_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, tighten_size=False),
        precomputed_masks=m.copy(),
    )
    assert off.table["size"].tolist()[0] == 169


def test_quantify_cellpose_recovers_a_missed_disk(
    backlit_disk_plate_path: str, plate_masks_missing_disk: NDArray[Any]
) -> None:
    """The (3, 3) disk is absent from the masks (11 instances). The grid says a well
    sits there; the probe finds depth 23 and threshold-recovers it: size 1001 (709 px
    disk grown 3 px), circularity 1.0, detector 'recovered', a new id 13 written into
    the masks with exactly 1001 px and colored R. With ``recover_missed_wells=False``
    the well stays empty at its node.
    """
    res = quantify_plate_image_cellpose(
        backlit_disk_plate_path,
        None,
        cfg=CFG,
        precomputed_masks=plate_masks_missing_disk.copy(),
        return_masks=True,
    )
    assert res.n_instances == 11
    row = res.table.iloc[10]
    assert (int(row["row"]), int(row["col"])) == (3, 3)
    assert (
        int(row["size"]),
        float(row["circularity"]),
        row["flags"],
        row["detector"],
    ) == (1001, 1.0, "", "recovered")
    assert (float(row["cx"]), float(row["cy"])) == pytest.approx(
        (230.0, 210.0), abs=1e-9
    )
    assert res.kept_color[13] == "R"
    assert res.masks is not None
    assert int(res.masks.max()) == 13 and int((res.masks == 13).sum()) == 1001
    off = quantify_plate_image_cellpose(
        backlit_disk_plate_path,
        None,
        cfg=CellposeSegConfig(n_rows=3, n_cols=4, recover_missed_wells=False),
        precomputed_masks=plate_masks_missing_disk.copy(),
    )
    r = off.table.iloc[10]
    assert (int(r["size"]), r["detector"]) == (0, "")
    assert 13 not in off.kept_color


def test_quantify_cellpose_uses_the_model_when_no_masks(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """Without ``precomputed_masks`` the RGB image goes to ``model.eval`` with the
    configured thresholds and element 0 of its return is the mask array; the resulting
    table equals the precomputed run.
    """
    calls: list[dict[str, Any]] = []

    class FakeModel:
        def eval(self, img: NDArray[Any], **kw: Any) -> tuple[NDArray[Any], None, None]:
            calls.append({"shape": img.shape, **kw})
            return plate_masks.copy(), None, None

    res = quantify_plate_image_cellpose(backlit_plate_path, FakeModel(), cfg=CFG)
    assert calls == [
        {
            "shape": (300, 400, 3),
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "diameter": None,
        }
    ]
    assert res.table["size"].tolist() == EXPECTED_SIZES
    assert res.table["flags"].tolist() == EXPECTED_FLAGS


def test_quantify_cellpose_default_config_rejects_the_small_plate(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """The default 14x22 array needs at least 61.6 blobs; 12 fail the lattice check."""
    with pytest.raises(
        ValueError, match="only 12 colony blobs detected for a 14x22 array"
    ):
        quantify_plate_image_cellpose(
            backlit_plate_path, None, precomputed_masks=plate_masks
        )


def test_quantify_cellpose_rejected_shape_and_off_gel_instances(
    backlit_plate_path: str, plate_masks: NDArray[Any]
) -> None:
    """A 3x21 bar (aspect 7 > 2.5) centered on the empty node (2, 3) is bucketed to that
    well but fails the shape gate: the well stays empty and reports the NODE (230, 150),
    not the bar's centroid; recovery then probes bare agar (depth 0) and leaves it. A
    7x7 instance at (20, 20) is more than 0.5 pitch outside the gel polygon and is
    skipped before bucketing, so it is neither a well nor an off-grid count. Both still
    count as raw instances (14).
    """
    m = plate_masks.copy()
    m[149:152, 220:241] = 20
    m[17:24, 17:24] = 21
    res = quantify_plate_image_cellpose(
        backlit_plate_path, None, cfg=CFG, precomputed_masks=m
    )
    assert (res.n_instances, res.n_offgrid) == (14, 0)
    assert res.table["size"].tolist() == EXPECTED_SIZES
    assert res.table["flags"].tolist() == EXPECTED_FLAGS
    rec = res.table.iloc[6].to_dict()
    assert (rec["detector"], rec["size"]) == ("", 0)
    assert (rec["cx"], rec["cy"]) == pytest.approx((230.0, 150.0), abs=0.1)
    assert 20 not in res.kept_color and 21 not in res.kept_color


# --- 2026.10.06 (Phase 21): the three remaining ``_recover_colony`` refusals ---------


def _agar_with_disk(radius: int, value: float = 190.0) -> NDArray[Any]:
    g = np.full((200, 200), 213.0)
    y, x = np.ogrid[-100:100, -100:100]
    g[(x * x + y * y) <= radius * radius] = value
    return g


def test_recover_colony_refuses_a_window_too_small_to_measure() -> None:
    """Pitch 7: the core is ``r <= 0.28 * 7 = 1.96`` px, the 3x3 block of 9 pixels
    (< 20), so the depression is never measured. The radius-3 colony (29 px) would
    otherwise be recovered: the core is all colony and the 3.85..5 px annulus all agar
    (depth 23 > 12), and it fills 29 of the 37 window pixels (r <= 3.22), above 20.
    At pitch 8 the core is already 21 pixels.
    """
    rr = np.hypot(*np.mgrid[-5:6, -5:6])
    assert int((rr <= 0.28 * 7).sum()) == 9
    assert int((rr <= 0.28 * 8).sum()) == 21
    assert int((rr <= 3).sum()) == 29
    assert _recover_colony(_agar_with_disk(3), 100, 100, 7.0, True, 12.0, 0) is None


def test_recover_colony_refuses_a_uniform_window() -> None:
    """A radius-30 colony at 190 covers the whole 0.46 * 60 = 27.6 px window while the
    0.55..0.78-pitch annulus (33..46.8 px) is bare agar: the depth 213 - 190 = 23 clears
    12, but the window is constant (no Otsu split) and the well is reported empty.
    """
    assert _recover_colony(_agar_with_disk(30), 100, 100, 60.0, True, 12.0, 0) is None


@pytest.mark.parametrize("grow_px", [0, 1])
def test_recover_colony_refuses_a_fragmented_core(grow_px: int) -> None:
    """A checkerboard of 190 pixels inside the radius-15 disk: 349 dark pixels in the
    885-pixel core (counted below) put its 25th percentile at 190 since 349 / 885 is
    0.39 > 0.25 (depth 23 > 12), but no two dark
    pixels share an edge, so the largest component is 1 px, 5 px after a disk(1)
    dilation, below ``MIN_COLONY_AREA = 20`` either way.
    """
    g = np.full((200, 200), 213.0)
    y, x = np.ogrid[-100:100, -100:100]
    checker = (np.add.outer(np.arange(200), np.arange(200)) % 2 == 0) & (
        (x * x + y * y) <= 225
    )
    g[checker] = 190.0
    rr = np.hypot(*np.mgrid[-100:100, -100:100])
    core = g[rr <= 0.28 * 60]
    assert (core.size, int((core == 190.0).sum())) == (885, 349)
    assert _recover_colony(g, 100, 100, 60.0, True, 12.0, grow_px) is None
