# tests/torchcell/sga/test_viz.py
# [[tests.torchcell.sga.test_viz]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_viz.py
"""SGAtools-style figures and the plate-address overlay labelling.

Every figure is checked structurally against its input: the imshow array equals the
(row, col) grid of values, tick labels are the plate addresses, bar heights and colors
follow the report, and the saved overlay canvas is the image plus a 2 * int(2.4 * 30) =
144 px white margin (the glyph size floors at 30 for a 200 px wide image, since
200 / 36 = 5.6 < 30).
"""

from __future__ import annotations

import os.path as osp

import matplotlib

matplotlib.use("Agg")
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402
from PIL import Image  # noqa: E402

from torchcell.sga.models import ScoreReport, StrainScore  # noqa: E402
from torchcell.sga.viz import (  # noqa: E402
    SEQUENTIAL_CMAP,
    _grid,
    _overlay_font,
    colony_shape_by_volume,
    label_plate_overlay,
    layout_heatmap,
    plate_heatmap,
    plate_labels,
    strain_fitness_plot,
    value_histogram,
)
from torchcell.utils import PLOT_PALETTE  # noqa: E402

HEAT = pd.DataFrame(
    {"row": [1, 1, 2, 2], "col": [1, 2, 1, 2], "norm": [0.5, 1.0, np.nan, 2.0]}
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _rects(ax: matplotlib.axes.Axes) -> list[Rectangle]:
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def test_grid_places_values_by_plate_address() -> None:
    """(row, col) 1-based -> 0-based array; an unlisted well is NaN."""
    df = pd.DataFrame({"row": [1, 3], "col": [2, 1], "v": [7.0, 9.0]})
    g = _grid(df, "v")
    assert g.shape == (3, 2)
    assert g[0, 1] == 7.0 and g[2, 0] == 9.0
    assert int(np.isnan(g).sum()) == 4


def test_sequential_cmap_is_magma() -> None:
    """The module-level heatmap colormap is matplotlib's magma."""
    assert SEQUENTIAL_CMAP.name == "magma"


def test_plate_heatmap_structure() -> None:
    """Image data = the value grid (NaN kept), x ticks '1'..'n', y ticks 'A'..; axis
    labels column/row; colorbar labelled with the value column (a second axes); the
    divider after column 1 is a vertical line at x = 0.5 and the half labels sit at
    x = 0.25 and (0.5 + 2) / 2 = 1.25, y = -0.9; figure size floors at 4.0 x 2.5 in.
    """
    fig = plate_heatmap(HEAT, divider_after_col=1, half_labels=("L", "R"), title="t")
    assert len(fig.axes) == 2
    ax, cax = fig.axes
    data = np.asarray(ax.images[0].get_array(), dtype=float)
    assert_array_equal(np.isnan(data), [[False, False], [True, False]])
    assert_array_equal(np.nan_to_num(data, nan=-1.0), [[0.5, 1.0], [-1.0, 2.0]])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("column", "row", "t")
    assert cax.get_ylabel() == "norm"
    assert len(ax.lines) == 1
    assert_array_equal(np.asarray(ax.lines[0].get_xdata()), [0.5, 0.5])
    assert [(t.get_text(), t.get_position()) for t in ax.texts] == [
        ("L", (0.25, -0.9)),
        ("R", (1.25, -0.9)),
    ]
    assert fig.get_size_inches().tolist() == [4.0, 2.5]
    assert ax.images[0].get_cmap().name == "magma"


def test_plate_heatmap_options() -> None:
    """No divider -> no line, no text; vmin/vmax fix the color limits; a string cmap is
    resolved by name; the figure grows 0.3 in per plate column past 13 columns.
    """
    fig = plate_heatmap(HEAT, value_col="norm", vmin=0.0, vmax=3.0, cmap="viridis")
    ax = fig.axes[0]
    assert len(ax.lines) == 0 and len(ax.texts) == 0
    assert ax.images[0].get_clim() == (0.0, 3.0)
    assert ax.images[0].get_cmap().name == "viridis"
    wide = pd.DataFrame({"row": [1], "col": [20], "norm": [1.0]})
    assert plate_heatmap(wide).get_size_inches().tolist() == [6.0, 2.5]
    # a divider without half labels draws the line only
    div = plate_heatmap(HEAT, divider_after_col=1).axes[0]
    assert len(div.lines) == 1 and len(div.texts) == 0


def test_layout_heatmap_codes_strains_alphabetically() -> None:
    """Codes are the alphabetical index (alpha 0, zeta 1), unassigned wells NaN, legend
    entries sorted with palette colors 1 and 2, color limits 0..n-1.
    """
    lay = pd.DataFrame(
        {
            "row": [1, 1, 2, 2],
            "col": [1, 2, 1, 2],
            "strain": ["zeta", "alpha", None, "alpha"],
        }
    )
    fig = layout_heatmap(lay)
    ax = fig.axes[0]
    data = np.asarray(ax.images[0].get_array(), dtype=float)
    assert_array_equal(np.nan_to_num(data, nan=-1.0), [[1.0, 0.0], [-1.0, 0.0]])
    assert ax.images[0].get_clim() == (0.0, 1.0)
    assert ax.get_title() == "strain layout"
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["alpha", "zeta"]
    handles = [h for h in legend.legend_handles if isinstance(h, Patch)]
    assert [h.get_facecolor() for h in handles] == [
        to_rgba(PLOT_PALETTE[0]),
        to_rgba(PLOT_PALETTE[1]),
    ]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]


def test_value_histogram_bins_and_color() -> None:
    """30 bars whose heights sum to the number of non-NaN values (4); bars in palette
    color 5 (blue); axis labels value column / colonies; all four spines visible.
    """
    fig = value_histogram(
        pd.DataFrame({"norm": [0.0, 0.5, 1.0, np.nan, 1.0]}), title="h"
    )
    ax = fig.axes[0]
    rects = _rects(ax)
    assert len(rects) == 30
    assert sum(r.get_height() for r in rects) == 4.0
    assert rects[0].get_facecolor() == to_rgba(PLOT_PALETTE[4])
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
        "norm",
        "colonies",
        "h",
    )
    assert all(sp.get_visible() for sp in ax.spines.values())


def test_colony_shape_by_volume_panels() -> None:
    """Left: one box per volume labelled '2.5 nL', '5.0 nL'; right: one scatter per
    volume with (size, circularity) offsets after dropping the S-flagged, missing and
    blank colonies; titles and legend as documented.
    """
    df = pd.DataFrame(
        {
            "volume_nl": [2.5, 2.5, 5.0, 5.0, 5.0, 2.5, 5.0],
            "circularity": [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.99],
            "size": [10, 20, 30, 40, 50, 60, 70],
            "is_blank": [False] * 6 + [True],
            "is_missing": [False] * 5 + [True, False],
            "flags": ["", "", "S", "", "", "", ""],
        }
    )
    fig = colony_shape_by_volume(df)
    assert len(fig.axes) == 2
    axb, axs = fig.axes
    assert [t.get_text() for t in axb.get_xticklabels()] == ["2.5 nL", "5.0 nL"]
    assert (axb.get_title(), axs.get_title()) == (
        "shape by volume",
        "circularity vs size",
    )
    assert axb.get_ylabel() == "circularity (1 = round)"
    offsets = [np.asarray(c.get_offsets()) for c in axs.collections]
    assert len(offsets) == 2
    assert_allclose(offsets[0], [[10.0, 0.9], [20.0, 0.8]])
    assert_allclose(offsets[1], [[40.0, 0.6], [50.0, 0.95]])
    legend = axs.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["2.5 nL", "5.0 nL"]


def test_strain_fitness_plot_bars() -> None:
    """Blank and unscored strains are dropped; bars sorted by relative fitness
    (geneA 0.5, BY4741 1.0, geneB 1.3); filled red for a significant knockout, gray for
    WT, white face with red edge when not significant; dashed reference at 1.0.
    """

    def _s(strain: str, rel: float | None, p: float | None = None) -> StrainScore:
        return StrainScore.model_validate(
            {
                "strain": strain,
                "n_total": 3,
                "n_used": 3,
                "relative_fitness": rel,
                "pvalue": p,
            }
        )

    rep = ScoreReport.model_validate(
        {
            "plate_id": "P1",
            "wt_name": "BY4741",
            "blank_name": "Blank_media",
            "n_colonies": 0,
            "n_missing": 0,
            "n_flagged": 0,
            "strains": [
                _s("BY4741", 1.0),
                _s("geneA", 0.5, 0.01),
                _s("geneB", 1.3, 0.5),
                _s("geneC", None),
                _s("Blank_media", 0.1),
            ],
        }
    )
    fig = strain_fitness_plot(rep)
    ax = fig.axes[0]
    rects = _rects(ax)
    assert [r.get_height() for r in rects] == [0.5, 1.0, 1.3]
    red, gray, white = (
        to_rgba(PLOT_PALETTE[1]),
        to_rgba(PLOT_PALETTE[5]),
        to_rgba("white"),
    )
    assert [r.get_facecolor() for r in rects] == [red, gray, white]
    assert [r.get_edgecolor() for r in rects] == [red, gray, red]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["geneA", "BY4741", "geneB"]
    assert ax.get_title() == "P1: single-KO fitness"
    assert ax.get_ylabel() == "relative fitness (vs BY4741)"
    assert len(ax.lines) == 1
    assert_array_equal(np.asarray(ax.lines[0].get_ydata()), [1.0, 1.0])
    # alpha decides the fill: at alpha 0.001 geneA is no longer significant
    strict = _rects(strain_fitness_plot(rep, alpha=0.001).axes[0])
    assert strict[0].get_facecolor() == white


@pytest.mark.parametrize(
    ("op", "rows", "cols"),
    [
        ("identity", ["A", "B", "C"], ["1", "2", "3", "4"]),
        ("rot180", ["C", "B", "A"], ["4", "3", "2", "1"]),
        ("flip_v", ["C", "B", "A"], ["1", "2", "3", "4"]),
        ("flip_h", ["A", "B", "C"], ["4", "3", "2", "1"]),
    ],
)
def test_plate_labels_follow_the_orientation(
    op: str, rows: list[str], cols: list[str]
) -> None:
    """Rows reverse under rot180/flip_v, columns under rot180/flip_h."""
    assert plate_labels(op, 3, 4) == (rows, cols)


def test_overlay_font_is_bundled_dejavu_bold() -> None:
    """A FreeType face at the requested size from matplotlib's mpl-data."""
    f = _overlay_font(30)
    assert f.size == 30
    path = str(f.path)
    assert path.endswith(osp.join("mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    assert osp.exists(path)


def test_label_plate_overlay_mats_the_image(tmp_path: Path) -> None:
    """A 200x150 gray overlay with a 3x3 grid: glyph size floors at 30, pad = 72, so the
    canvas is (150 + 144) x (200 + 144); the original image sits at offset (72, 72);
    cyan crosses of half-length 5 mark each node; a black column tick runs the 18 px
    above the image edge at each top-row node x; the margin corners stay white.
    """
    path = str(tmp_path / "ov.png")
    Image.fromarray(np.full((150, 200, 3), 128, np.uint8)).save(path)
    nodes = np.stack(
        np.meshgrid([40.0, 75.0, 110.0], [50.0, 100.0, 150.0], indexing="ij"), -1
    )
    label_plate_overlay(path, nodes, "rot180")
    lab = np.asarray(Image.open(path).convert("RGB"))
    assert lab.shape == (294, 344, 3)
    assert tuple(lab[0, 0]) == (255, 255, 255) and tuple(lab[293, 343]) == (
        255,
        255,
        255,
    )
    assert tuple(lab[72, 72]) == (128, 128, 128)
    for y in (40, 75, 110):
        for x in (50, 100, 150):
            assert tuple(lab[72 + y, 72 + x]) == (0, 255, 255)
            assert tuple(lab[72 + y, 72 + x + 5]) == (0, 255, 255)
            assert tuple(lab[72 + y, 72 + x - 5]) == (0, 255, 255)
            assert tuple(lab[72 + y + 5, 72 + x]) == (0, 255, 255)
            assert tuple(lab[72 + y, 72 + x + 6]) == (128, 128, 128)
    for x in (50, 100, 150):
        assert tuple(lab[71, 72 + x]) == (0, 0, 0)
        assert tuple(lab[55, 72 + x]) == (0, 0, 0)
        assert tuple(lab[72 + 150, 72 + x]) == (0, 0, 0)
    for y in (40, 75, 110):
        assert tuple(lab[72 + y, 71]) == (0, 0, 0)
        assert tuple(lab[72 + y, 72 + 200]) == (0, 0, 0)
    # glyphs are drawn in the margins: some black above the top ticks
    assert int(np.all(lab[:50, :] == 0, axis=-1).sum()) > 0
