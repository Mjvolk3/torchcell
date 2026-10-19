# tests/torchcell/utils/test_utils.py
# [[tests.torchcell.utils.test_utils]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/utils/test_utils.py
"""``torchcell.utils.utils``: figure standards, true-size SVG export, panel letters, and
scientific notation.

Palette: 18 line colors in six-slot order; entries 7-12 are the primaries shaded by
0.73 per channel and rounded (``#D79B00`` -> 215, 155, 0 -> 157, 113, 0 -> ``#9D7100``).
The first six are the locked draw.io colors.

SVG export: a 2 x 1 inch figure is written by matplotlib as ``width="144pt"
height="72pt"``; the rescale to 100 per inch multiplies by 100/72 = 1.3888888889, so
the root reads ``width="200.0000" height="100.0000" viewBox="0 0 200.0000 100.0000"``
and the content is wrapped in ``<g transform="scale(1.3888888889)">``. Saving the same
figure twice under the same basename gives identical bytes (salted ids, no date).

Panel letter: with ``x`` None the letter's anchor lands exactly on the axes' tight-bbox
left edge (the offset is minus the tick/label overhang) and 12 pt above the axes top,
that is ``12/72 * fig.dpi`` pixels; with ``x`` given it is ``pad_pt`` = 1.5 pt left of
that axes-fraction point.

Scientific notation: the docstring's four examples plus 0 -> "0", -1500 -> "-1.5e03",
5 -> "5e00", 0.5 -> "5e-1", 0.007 -> "7e-3".
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from torchcell.utils.utils import (
    MAX_HEIGHT_MM,
    PANEL_LABEL_PAD_PT,
    PANEL_LABEL_PT,
    PANEL_LABEL_RAISE_PT,
    PANEL_WIDTHS_MM,
    PAPER_RC,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    PLOT_PALETTE_NAMES,
    REPRESENTATION_DISPLAY_NAMES,
    apply_paper_style,
    display_label,
    format_scientific_notation,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

LOCKED_LINE = ["#D79B00", "#B85450", "#9673A6", "#D6B656", "#6C8EBF", "#666666"]
LOCKED_FILL = ["#FFE6CC", "#F8CECC", "#E1D5E7", "#FFF2CC", "#DAE8FC", "#F5F5F5"]


def _shade(color: str, factor: float) -> str:
    channels = [round(int(color[i : i + 2], 16) * factor) for i in (1, 3, 5)]
    return "#" + "".join(f"{c:02X}" for c in channels)


# ------------------------------------------------------------- widths and palette


def test_panel_widths_and_height_cap_are_the_nature_template() -> None:
    """Six named widths in mm, a 170 mm height cap, and 25.4 mm to the inch."""
    assert PANEL_WIDTHS_MM == {
        "full": 179.0,
        "wide": 118.9,
        "half_plus": 88.5,
        "half": 88.0,
        "third": 57.8,
        "sixth": 28.3,
    }
    assert MAX_HEIGHT_MM == 170.0
    assert mm_to_in(25.4) == 1.0
    assert mm_to_in(179.0) == 179.0 / 25.4
    assert (PANEL_LABEL_PT, PANEL_LABEL_PAD_PT, PANEL_LABEL_RAISE_PT) == (
        8.0,
        1.5,
        12.0,
    )


def test_palette_is_eighteen_colors_with_locked_primaries_and_a_0_73_dark_tier() -> (
    None
):
    """18 line colors, 18 fills, 18 names; positions 1-6 are the locked draw.io pairs;
    positions 7-12 are the primaries at 0.73 per channel (rounded); every entry is an
    uppercase ``#RRGGBB`` and no line color repeats.
    """
    assert len(PLOT_PALETTE) == len(PLOT_PALETTE_FILL) == len(PLOT_PALETTE_NAMES) == 18
    assert PLOT_PALETTE[:6] == LOCKED_LINE
    assert PLOT_PALETTE_FILL[:6] == LOCKED_FILL
    assert PLOT_PALETTE[6:12] == [_shade(color, 0.73) for color in LOCKED_LINE]
    assert _shade("#D79B00", 0.73) == "#9D7100"
    hex_color = re.compile(r"^#[0-9A-F]{6}$")
    for color in PLOT_PALETTE + PLOT_PALETTE_FILL:
        assert hex_color.match(color), color
    assert len(set(PLOT_PALETTE)) == 18
    assert PLOT_PALETTE_NAMES == [
        "amber",
        "brick",
        "lilac",
        "wheat",
        "steel blue",
        "gray",
        "bronze",
        "maroon",
        "plum",
        "old gold",
        "denim",
        "charcoal",
        "terracotta",
        "rose brown",
        "dusty mauve",
        "sand",
        "slate",
        "taupe",
    ]


def test_display_label_maps_the_two_species_lm_keys_and_nothing_else() -> None:
    """``fudt_upstream`` / ``fudt_downstream`` become the Species LM labels; any other
    key is returned unchanged.
    """
    assert REPRESENTATION_DISPLAY_NAMES == {
        "fudt_upstream": "species_lm_five_prime",
        "fudt_downstream": "species_lm_three_prime",
    }
    assert display_label("fudt_upstream") == "species_lm_five_prime"
    assert display_label("fudt_downstream") == "species_lm_three_prime"
    assert display_label("codon_frequency") == "codon_frequency"


def test_apply_paper_style_sets_paper_rc() -> None:  # test-quality: allow rcParams
    """After a deliberate 16 pt label size, the call restores each ``PAPER_RC`` key to
    its value (``font.family`` is normalized to the list ``["Arial"]``); the context
    manager undoes it afterwards.
    """
    before = plt.rcParams["axes.labelsize"]
    with matplotlib.rc_context():
        plt.rcParams["axes.labelsize"] = 16.0
        plt.rcParams["savefig.bbox"] = "tight"
        apply_paper_style()
        # Iterate rather than index by a str key: CI's matplotlib stubs type the
        # RcParams keys as a Literal set.
        applied = {key: value for key, value in plt.rcParams.items() if key in PAPER_RC}
        assert applied == {**PAPER_RC, "font.family": ["Arial"]}
        assert plt.rcParams["axes.labelsize"] == 6.0
        assert plt.rcParams["savefig.bbox"] is None
    assert plt.rcParams["axes.labelsize"] == before


# ------------------------------------------------------------------- SVG export


def _figure() -> Any:
    """A 2 x 1 inch figure; call under ``apply_paper_style`` so nothing recrops it."""
    fig = plt.figure(figsize=(2.0, 1.0))
    ax = fig.add_subplot()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    return fig


def test_savefig_true_size_svg_rescales_the_root_to_100_per_inch(
    tmp_path: Path,
) -> None:
    """144 x 72 pt becomes 200.0000 x 100.0000 with a matching viewBox and a
    ``scale(1.3888888889)`` group closing right before ``</svg>``; no ``pt`` unit
    survives on the root; the rc salt is restored afterwards. The figure is saved under
    the paper rc (``savefig.bbox`` None): importing ``torchcell.graph`` elsewhere in a
    session applies a style with tight cropping, which would shrink the root.
    """
    path = tmp_path / "panel.svg"
    with matplotlib.rc_context():
        apply_paper_style()
        fig = _figure()
        savefig_true_size_svg(fig, str(path))
        plt.close(fig)
    svg = path.read_text(encoding="utf-8")
    root = re.search(r"<svg\b[^>]*>", svg)
    assert root is not None
    tag = root.group(0)
    assert 'width="200.0000"' in tag
    assert 'height="100.0000"' in tag
    assert 'viewBox="0 0 200.0000 100.0000"' in tag
    assert "pt" not in tag
    assert svg[root.end() :].startswith('<g transform="scale(1.3888888889)">')
    assert svg.rstrip().endswith("</g></svg>")
    assert svg.count("</svg>") == 1
    assert "<dc:date>" not in svg
    assert matplotlib.rcParams["svg.hashsalt"] is None


def test_savefig_true_size_svg_is_byte_stable_and_honors_custom_dpi(
    tmp_path: Path,
) -> None:
    """Two saves of one figure under the same basename are identical bytes; with
    ``drawio_dpi`` equal to ``src_dpi`` the factor is 1 and the size stays 144 x 72.
    """
    first = tmp_path / "a" / "panel.svg"
    second = tmp_path / "b" / "panel.svg"
    first.parent.mkdir()
    second.parent.mkdir()
    unit = tmp_path / "unit.svg"
    with matplotlib.rc_context():
        apply_paper_style()
        fig = _figure()
        savefig_true_size_svg(fig, str(first))
        savefig_true_size_svg(fig, str(second))
        savefig_true_size_svg(fig, str(unit), src_dpi=72, drawio_dpi=72)
        plt.close(fig)
    assert first.read_bytes() == second.read_bytes()
    svg = unit.read_text(encoding="utf-8")
    root = re.search(r"<svg\b[^>]*>", svg)
    assert root is not None
    assert 'width="144.0000"' in root.group(0)
    assert 'height="72.0000"' in root.group(0)
    assert svg[root.end() :].startswith('<g transform="scale(1.0000000000)">')


class _TextFigure:
    """A ``savefig`` that writes fixed text and records what it was asked."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: list[dict[str, Any]] = []

    def savefig(self, path: str, **kwargs: Any) -> None:
        self.kwargs.append(kwargs)
        Path(path).write_text(self.text, encoding="utf-8")


def test_savefig_true_size_svg_refuses_non_matplotlib_output(tmp_path: Path) -> None:
    """No ``<svg>`` root, or a root without ``pt`` width and height, is an error naming
    the path; the default ``metadata={"Date": None}`` is passed unless the caller sets
    metadata.
    """
    path = tmp_path / "bad.svg"
    html: Any = _TextFigure("<html></html>")
    with pytest.raises(
        ValueError,
        match=f"^{re.escape(f'{path}: no <svg> root tag; not a matplotlib SVG')}$",
    ):
        savefig_true_size_svg(html, str(path))
    assert html.kwargs == [{"metadata": {"Date": None}}]
    unitless: Any = _TextFigure('<svg width="10" height="10"></svg>')
    with pytest.raises(
        ValueError,
        match=f"^{re.escape(f'{path}: <svg> root has no pt width/height to rescale')}$",
    ):
        savefig_true_size_svg(unitless, str(path), metadata={"Creator": "x"})
    assert unitless.kwargs == [{"metadata": {"Creator": "x"}}]


# ------------------------------------------------------------------ panel letters


def test_panel_label_sits_flush_left_of_the_tight_bbox_and_12pt_up() -> None:
    """The letter is lowercased, 8 pt bold Arial, bottom-aligned, on a white patch; with
    ``x`` None it is left-aligned at axes x 0 and its anchor maps to the tight bbox's
    left edge and ``ax.bbox.y1 + 12/72 * dpi`` pixels; with ``x`` 0.4 it is
    right-aligned at (0.4, 1.0), shifted 1.5 pt left.
    """
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_ylabel("fitness")
    tight_before = ax.get_tightbbox()
    assert tight_before is not None
    text = panel_label(ax, "A")
    assert text.get_text() == "a"
    assert text.get_fontsize() == 8.0
    assert text.get_fontweight() == "bold"
    assert text.get_fontfamily() == ["Arial"]
    assert (text.get_horizontalalignment(), text.get_verticalalignment()) == (
        "left",
        "bottom",
    )
    assert text.get_position() == (0.0, 1.0)
    patch = text.get_bbox_patch()
    assert patch is not None
    assert patch.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert patch.get_edgecolor() == (0.0, 0.0, 0.0, 0.0)
    anchor = text.get_transform().transform((0.0, 1.0))
    assert anchor[0] == pytest.approx(tight_before.x0)
    assert anchor[1] == pytest.approx(ax.bbox.y1 + 12.0 / 72.0 * fig.dpi)

    placed = panel_label(ax, "B", x=0.4)
    assert placed.get_text() == "b"
    assert placed.get_horizontalalignment() == "right"
    assert placed.get_position() == (0.4, 1.0)
    anchor_b = placed.get_transform().transform((0.4, 1.0))
    axes_point = ax.transAxes.transform((0.4, 1.0))
    assert anchor_b[0] == pytest.approx(axes_point[0] - 1.5 / 72.0 * fig.dpi)
    assert anchor_b[1] == pytest.approx(axes_point[1] + 12.0 / 72.0 * fig.dpi)
    plt.close(fig)


def test_panel_label_refuses_a_detached_or_unmeasurable_axes() -> None:
    """An axes with no figure, or one whose tight bbox is None, raises with the exact
    reason before any text is drawn.
    """
    detached: Any = SimpleNamespace(get_figure=lambda: None)
    with pytest.raises(
        ValueError, match=r"^panel_label: the axes is not attached to a figure$"
    ):
        panel_label(detached, "a")
    unmeasurable: Any = SimpleNamespace(
        get_figure=lambda: SimpleNamespace(dpi=100.0, dpi_scale_trans=None),
        get_tightbbox=lambda: None,
    )
    with pytest.raises(
        ValueError,
        match=r"^panel_label: the axes has no tight bounding box to measure$",
    ):
        panel_label(unmeasurable, "a")


# ------------------------------------------------------------ scientific notation


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, "0"),
        (10000, "1e04"),
        (10007, "1.0007e04"),
        (1234567, "1.234567e06"),
        (100, "1e02"),
        (-1500, "-1.5e03"),
        (5, "5e00"),
        (0.5, "5e-1"),
        (0.007, "7e-3"),
        (1e15, "1e15"),
    ],
)
def test_format_scientific_notation_keeps_integer_digits(
    number: float, expected: str
) -> None:
    """Two-digit zero-padded exponent, minimal mantissa, sign carried through."""
    assert format_scientific_notation(number) == expected


def test_format_scientific_notation_drops_fractional_digits() -> None:
    """Finding: the docstring (utils.py line 327) promises "preserving all significant
    digits", but the precision search (line 353) stops when the ROUNDED reconstruction
    matches the ROUNDED input, so 1234.5 -> "1.234e03" and 99.9 -> "1e01": every digit
    below the units place is lost.
    """
    assert format_scientific_notation(1234.5) == "1.234e03"
    assert format_scientific_notation(99.9) == "1e01"


def test_acs_structure_draws_at_the_acs_bond_length(tmp_path: Path) -> None:
    """A structure drawn by ``acs_structure`` is a set of vector paths in points: the
    bond is the ACS 1996 fixed length (10.8 pt, 0.381 cm), the line 0.6 pt, and the
    atom labels are glyph outlines, so the panel needs no font to render them.
    """
    pytest.importorskip("rdkit")
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch

    from torchcell.utils import ACS_BOND_PT, ACS_LINE_PT, acs_structure

    apply_paper_style()
    fig = plt.figure(figsize=(mm_to_in(57.8), mm_to_in(30)))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    box, width_pt, height_pt = acs_structure(ax, "OC=O", (50, 25))
    patches = [
        a for a in box.get_children()[0].get_children() if isinstance(a, PathPatch)
    ]
    alpha = {id(p): to_rgba(p.get_facecolor())[3] for p in patches}
    stroked = [p for p in patches if alpha[id(p)] == 0.0]
    glyphs = [p for p in patches if alpha[id(p)] == 1.0]
    # a bond is a two-point stroked path; RDKit also strokes a three-point join cap
    # at every unlabeled vertex, which is not a bond
    vertices = {id(p): np.asarray(p.get_path().vertices) for p in stroked}
    bonds = [p for p in stroked if len(vertices[id(p)]) == 2]
    assert bonds and glyphs, "a bond is a stroked path, a label a filled outline"
    assert all(p.get_linewidth() == ACS_LINE_PT for p in stroked)
    # each bond runs the ACS length minus the clearance RDKit leaves at a labeled
    # end, so it is shorter than the bond length and longer than half of it
    lengths = [
        float(np.hypot(*(vertices[id(p)][1] - vertices[id(p)][0]))) for p in bonds
    ]
    assert all(ACS_BOND_PT / 2 < length <= ACS_BOND_PT for length in lengths)
    assert 0 < width_pt < 60 and 0 < height_pt < 30
    fig.savefig(tmp_path / "formic.pdf")
    plt.close(fig)


def test_text_clashes_reports_overlap_and_the_edge_only() -> None:
    import matplotlib.pyplot as plt

    from torchcell.utils import text_clashes

    fig = plt.figure(figsize=(3, 2), dpi=100)
    fig.text(0.1, 0.5, "first label", fontsize=8)
    fig.text(0.12, 0.5, "second label", fontsize=8)  # starts inside the first
    fig.text(0.1, 0.1, "clear of both", fontsize=8)
    fig.text(0.95, 0.8, "runs off the right edge", fontsize=8)
    fig.text(0.5, 0.9, "✗", fontsize=8)  # a status mark is never a clash
    fig.text(0.5, 0.9, "", fontsize=8)  # nor is an empty string
    clashes = text_clashes(fig)
    plt.close(fig)
    assert ("first label", "second label") in clashes
    assert ("runs off the right edge", "<figure edge>") in clashes
    assert not any("clear of both" in pair for pair in clashes)
    assert len(clashes) == 2
