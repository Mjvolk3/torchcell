# tests/torchcell/viz/test_transformer_diagnostics.py
# [[tests.torchcell.viz.test_transformer_diagnostics]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_transformer_diagnostics.py
"""``TransformerDiagnostics.plot_attention_diagnostics``: six panels plus a twin axis.

Fixture: two layers given out of order (``{1: partial, 0: full}``). Layer 0 carries all
eight statistics; layer 1 carries only the five required ones, so ``max_row_weight``,
``col_entropy`` and ``max_col_sum`` fall to ``0.0`` for it, and a ``residual_ratios``
dict that names only layer 0 gives layer 1 a ratio of ``0.0`` as well.

``fig.axes`` is ``[ax1..ax6, ax5_twin]`` (twins are appended after the grid). Colors
come from ``torchcell/torchcell.mplstyle``: index 2 ``#7191A9``, 4 ``#B73C39``,
0/1/3 ``#000000``/``#D86E2F``/``#6B8D3A``, 5 ``#34699D``, 6 ``#775A9F``, 7 ``#4A9C60``,
10 ``#A05B2C``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from packaging.version import Version  # noqa: E402

from torchcell.viz.transformer_diagnostics import TransformerDiagnostics  # noqa: E402

FULL = {
    "entropy": 2.5,
    "effective_rank": 12.0,
    "top5": 0.3,
    "top10": 0.5,
    "top50": 0.9,
    "max_row_weight": 0.45,
    "col_entropy": 3.0,
    "max_col_sum": 7.0,
}
PARTIAL = {
    "entropy": 1.5,
    "effective_rank": 4.0,
    "top5": 0.6,
    "top10": 0.7,
    "top50": 0.95,
}
FIGURE_HEIGHT_IN = 15.0
# The module's hard-coded fallback; it is NOT the mplstyle palette (see the graph_recovery
# tests for the eight indices at which the two differ).
FALLBACK_COLORS = [
    "#000000",
    "#CC8250",
    "#7191A9",
    "#6B8D3A",
    "#B73C39",
    "#34699D",
    "#3D796E",
    "#4A9C60",
    "#E6A65D",
    "#A05B2C",
    "#3978B5",
    "#D86E2F",
    "#775A9F",
    "#EBB386",
    "#8D5694",
    "#52B2A8",
    "#35978A",
    "#AB4B4B",
    "#6D666F",
    "#4E7C7F",
    "#905353",
    "#C17132",
]


def _lines(fig: Figure, axis: int) -> list[tuple[list[list[float]], str, str]]:
    return [
        (
            np.asarray(line.get_xydata()).tolist(),
            str(line.get_color()),
            str(line.get_label()),
        )
        for line in fig.axes[axis].lines
    ]


def _texts(fig: Figure, axis: int) -> list[str]:
    return [t.get_text() for t in fig.axes[axis].texts]


def _legend(fig: Figure, axis: int) -> list[str]:
    legend = fig.axes[axis].get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


@pytest.fixture
def vis(tmp_path: Path) -> TransformerDiagnostics:
    return TransformerDiagnostics(str(tmp_path))


def test_default_palette_matches_graph_recovery(vis: TransformerDiagnostics) -> None:
    """Both viz classes parse the same style file; spot-check the indices the plots use."""
    assert len(vis.colors) == 22
    assert [vis.colors[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 10)] == [
        "#000000",
        "#D86E2F",
        "#7191A9",
        "#6B8D3A",
        "#B73C39",
        "#34699D",
        "#775A9F",
        "#4A9C60",
        "#A05B2C",
    ]


def test_fallback_palette_when_the_style_file_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "none.mplstyle"
    vis = TransformerDiagnostics(str(tmp_path), mplstyle_path=str(missing))
    assert vis.colors == FALLBACK_COLORS
    assert capsys.readouterr().out.startswith(
        f"Warning: Could not load colors from {missing}: "
    )


def test_fallback_palette_when_the_style_file_has_no_prop_cycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parseable style without ``axes.prop_cycle`` falls back without a warning."""
    style = tmp_path / "plain.mplstyle"
    style.write_text("lines.linewidth: 3\n")
    vis = TransformerDiagnostics(str(tmp_path), mplstyle_path=str(style))
    assert vis.colors == FALLBACK_COLORS
    assert capsys.readouterr().out == ""


def test_save_and_log_figure_logs_a_png_with_commit(
    vis: TransformerDiagnostics, wandb_recorder: Any
) -> None:
    """Unlike GraphRecoveryVisualization there is no ``wandb.run`` guard here."""
    fig = plt.figure(figsize=(1, 1))
    vis.save_and_log_figure(fig, "val_transformer_diagnostics/summary")
    payload, commit = wandb_recorder.logged[0]
    assert list(payload) == ["val_transformer_diagnostics/summary"] and commit is True
    image = payload["val_transformer_diagnostics/summary"].image
    assert image.format == "PNG"
    assert tuple(round(d) for d in image.info["dpi"]) == (300, 300)


def test_six_panels_plot_the_per_layer_series(
    vis: TransformerDiagnostics,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """Layers sort to [0, 1]; missing statistics read 0.0; each panel's data and labels."""
    figures = figure_capture(vis)
    vis.plot_attention_diagnostics(
        {1: PARTIAL, 0: FULL},
        residual_ratios={0: 0.5, 1: 0.25},
        qk_logit_stats={0: {"logit_mean": 0.0}},
        gradient_norms={0: 1.0},
        num_epochs=4,
        stage="train",
    )
    assert wandb_recorder.keys == ["train_transformer_diagnostics/summary"]
    fig = figures[0]
    assert len(fig.axes) == 7
    assert fig.get_suptitle() == "Transformer Attention Diagnostics - Epoch 4"

    assert _lines(fig, 0) == [([[0, 2.5], [1, 1.5]], "#7191A9", "Attention Entropy")]
    assert _texts(fig, 0) == ["2.50", "1.50"]
    assert fig.axes[0].get_title() == "Attention Entropy per Layer"
    assert fig.axes[0].get_ylabel() == "Entropy (nats)"

    assert _lines(fig, 1) == [([[0, 12.0], [1, 4.0]], "#B73C39", "Effective Rank")]
    assert _texts(fig, 1) == ["12.0", "4.0"]
    assert fig.axes[1].get_yscale() == "log"
    assert fig.axes[1].get_title() == "Attention Concentration (Effective Rank)"

    assert _lines(fig, 2) == [
        ([[0, 0.3], [1, 0.6]], "#000000", "Top-5 Mass"),
        ([[0, 0.5], [1, 0.7]], "#D86E2F", "Top-10 Mass"),
        ([[0, 0.9], [1, 0.95]], "#6B8D3A", "Top-50 Mass"),
    ]
    assert _legend(fig, 2) == ["Top-5 Mass", "Top-10 Mass", "Top-50 Mass"]
    assert fig.axes[2].get_ylim() == (0.0, 1.0)
    assert fig.axes[2].get_title() == "Top-K Attention Concentration"

    assert _lines(fig, 3) == [([[0, 0.45], [1, 0.0]], "#34699D", "Max Row Weight")]
    assert _texts(fig, 3) == ["0.450", "0.000"]
    assert fig.axes[3].get_ylim() == (0.0, 1.0)
    assert fig.axes[3].get_title() == "Max Row Weight (One-Hot Detection)"

    assert _lines(fig, 5) == [
        ([[0, 0.5], [1, 0.25]], "#A05B2C", "Residual Update Ratio"),
        ([[0, 0.1], [1, 0.1]], "#6B8D3A", "Healthy Lower Bound"),
        ([[0, 1.0], [1, 1.0]], "#B73C39", "Healthy Upper Bound"),
    ]
    assert _texts(fig, 5) == ["0.500", "0.250"]
    assert fig.axes[5].get_yscale() == "log"
    assert fig.axes[5].get_title() == "Residual Update Ratio (Layer Health)"
    assert fig.axes[5].get_ylabel() == "Update Ratio (||Δx|| / ||x||)"
    for axis in range(6):
        assert fig.axes[axis].get_xlabel() == "Transformer Layer"
        np.testing.assert_array_equal(fig.axes[axis].get_xticks(), [0, 1])


def test_column_panel_uses_a_twin_axis_with_a_combined_legend(
    vis: TransformerDiagnostics,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """Finding: two color comments name a different hue than the palette delivers.
    transformer_diagnostics.py:296 says "Teal" for index 6, which is ``#775A9F`` (a
    purple), and :340 says "Purple" for index 10, which is ``#A05B2C`` (a brown). The
    "Green" comment at :307 is right: index 7 is ``#4A9C60``.
    """
    figures = figure_capture(vis)
    vis.plot_attention_diagnostics(
        {0: FULL, 1: FULL}, residual_ratios={0: 0.5, 1: 0.2}, num_epochs=1
    )
    fig = figures[0]
    ax5, twin = fig.axes[4], fig.axes[6]
    assert _lines(fig, 4) == [([[0, 3.0], [1, 3.0]], "#775A9F", "Column Entropy")]
    assert _lines(fig, 6) == [([[0, 7.0], [1, 7.0]], "#4A9C60", "Max Column Sum")]
    assert _legend(fig, 4) == ["Column Entropy", "Max Column Sum"]
    assert twin.get_legend() is None
    assert ax5.get_ylabel() == "Column Entropy (nats)"
    assert ax5.yaxis.label.get_color() == "#775A9F"
    assert twin.get_ylabel() == "Max Column Sum"
    assert twin.yaxis.label.get_color() == "#4A9C60"
    assert ax5.get_title() == "Column Concentration (Sink Collapse Detection)"
    assert fig.axes[5].lines[0].get_color() == "#A05B2C"
    bbox = fig.get_tightbbox()
    assert bbox is not None
    assert FIGURE_HEIGHT_IN - 1 < bbox.height < FIGURE_HEIGHT_IN + 1


def test_zero_residual_ratio_on_the_log_axis_blows_up_the_tight_bbox(
    vis: TransformerDiagnostics, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the default ``residual_ratios=None`` (and any layer missing from the
    dict) plots ``0.0`` on a log-scaled axis and labels it with ``ax6.text(layer, 0.0,
    ...)``. Matplotlib clips ``log10(0)`` to -1000 decades, which puts that text about
    3000 figure-inches below the axes. Under matplotlib 3.10 (the torchcell env, 3.10.7)
    ``bbox_inches="tight"`` includes that text, so the 300-dpi PNG in
    ``save_and_log_figure`` is several gigapixels and ``PIL.Image.open`` raises
    ``DecompressionBombError``. Under matplotlib 3.11 (the CI runner) the tight bbox
    stays at figure size (measured 14.685 in for the 15 in figure), although the text
    still sits far below the axes, so the bomb does not fire there. The trainer passes
    ``None`` whenever its residual accumulator is empty. Saving is bypassed here; the
    geometry is asserted per matplotlib version.
    """
    figures: list[Figure] = []
    monkeypatch.setattr(
        vis, "save_and_log_figure", lambda fig, name, ts=None: figures.append(fig)
    )
    vis.plot_attention_diagnostics({0: FULL, 1: FULL}, num_epochs=1)
    fig = figures[0]
    ax6 = fig.axes[5]
    np.testing.assert_array_equal(ax6.lines[0].get_ydata(), [0.0, 0.0])
    assert [t.get_text() for t in ax6.texts] == ["0.000", "0.000"]
    label_y = ax6.texts[0].get_window_extent().y0
    assert label_y < -100 * FIGURE_HEIGHT_IN * fig.dpi
    bbox = fig.get_tightbbox()
    assert bbox is not None
    if Version(matplotlib.__version__) < Version("3.11"):
        assert bbox.height > 100 * FIGURE_HEIGHT_IN
    else:
        assert bbox.height <= FIGURE_HEIGHT_IN


def test_empty_stats_print_and_return(
    vis: TransformerDiagnostics, wandb_recorder: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    vis.plot_attention_diagnostics({})
    assert capsys.readouterr().out == "No attention stats to plot\n"
    assert wandb_recorder.logged == []
