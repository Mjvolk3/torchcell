# tests/torchcell/viz/test_graph_recovery.py
# [[tests.torchcell.viz.test_graph_recovery]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_graph_recovery.py
"""``GraphRecoveryVisualization``: palette loading, the wandb round trip, five plots.

The palette is whatever ``torchcell/torchcell.mplstyle`` declares in ``axes.prop_cycle``
(22 colors, ``#000000`` then ``#D86E2F``, ``#7191A9``, ...); the hard-coded fallback list
in the module differs from it at several indices (index 1 is ``#CC8250`` there) and is
only reached when the style file cannot be parsed. Bars carry ``alpha=0.8``, so a face
color compares against ``to_rgba(hex, 0.8)``.

Metric keys are ``"<graph>_L<layer>_H<head>"``. Recall bars sort by (graph, layer) and
take one palette color per distinct graph in first-seen order; precision lines take one
color per graph in alphabetical order and one line style per distinct layer in sorted
order (``-``, ``--``, ``-.``, ``:``); per-graph plots run over ``sorted(recall)`` and skip
graphs without precision entries.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import wandb  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from torchcell.viz.graph_recovery import GraphRecoveryVisualization  # noqa: E402

MPLSTYLE_COLORS = [
    "#000000",
    "#D86E2F",
    "#7191A9",
    "#6B8D3A",
    "#B73C39",
    "#34699D",
    "#775A9F",
    "#4A9C60",
    "#E6A65D",
    "#52B2A8",
    "#A05B2C",
    "#3978B5",
    "#EBB386",
    "#3D796E",
    "#8D5694",
    "#CC8250",
    "#35978A",
    "#AB4B4B",
    "#6D666F",
    "#4E7C7F",
    "#905353",
    "#C17132",
]
# The module's hard-coded fallback list (graph_recovery.py:55-78).
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
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _heights(fig: Figure, axis: int = 0) -> list[float]:
    return [
        float(p.get_height())
        for p in fig.axes[axis].patches
        if isinstance(p, Rectangle)
    ]


def _faces(fig: Figure) -> list[tuple[float, float, float, float]]:
    return [
        to_rgba(p.get_facecolor())
        for p in fig.axes[0].patches
        if isinstance(p, Rectangle)
    ]


def _ticklabels(fig: Figure, axis: int = 0) -> list[str]:
    return [t.get_text() for t in fig.axes[axis].get_xticklabels()]


def _texts(fig: Figure, axis: int = 0) -> list[str]:
    return [t.get_text() for t in fig.axes[axis].texts]


def _legend_labels(legend: Legend | None) -> list[str]:
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


@pytest.fixture
def wandb_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """``save_and_log_figure`` logs only when ``wandb.run`` is not None."""
    monkeypatch.setattr(wandb, "run", object())


@pytest.fixture
def vis(tmp_path: Path) -> GraphRecoveryVisualization:
    return GraphRecoveryVisualization(str(tmp_path))


def test_default_palette_is_the_mplstyle_prop_cycle(
    vis: GraphRecoveryVisualization,
) -> None:
    assert vis.colors == MPLSTYLE_COLORS


def test_unreadable_style_file_falls_back_to_the_hard_coded_palette(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: the comment "Fallback to full torchcell.mplstyle palette (22 colors)" at
    graph_recovery.py:54 (and transformer_diagnostics.py:59) is false. The fallback list
    differs from the mplstyle prop_cycle at 8 of 22 indices (1, 6, 9, 10, 11, 12, 13, 15;
    e.g. index 1 is ``#CC8250`` in the fallback and ``#D86E2F`` in the style file), so a
    plot drawn after a failed style load is colored differently from one drawn after a
    successful load. A missing file takes this path and prints a warning.
    """
    missing = tmp_path / "missing.mplstyle"
    vis = GraphRecoveryVisualization(str(tmp_path), mplstyle_path=str(missing))
    assert vis.colors == FALLBACK_COLORS
    assert [
        i for i, (a, b) in enumerate(zip(FALLBACK_COLORS, MPLSTYLE_COLORS)) if a != b
    ] == [1, 6, 9, 10, 11, 12, 13, 15]
    assert capsys.readouterr().out.startswith(
        f"Warning: Could not load colors from {missing}: "
    )


def test_style_file_without_prop_cycle_falls_back_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    style = tmp_path / "plain.mplstyle"
    style.write_text("lines.linewidth: 3\n")
    vis = GraphRecoveryVisualization(str(tmp_path), mplstyle_path=str(style))
    assert vis.colors == FALLBACK_COLORS
    assert capsys.readouterr().out == ""


def test_style_file_colors_are_hash_prefixed(tmp_path: Path) -> None:
    """Mplstyle files spell colors without '#' (it starts a comment); the loader adds it."""
    style = tmp_path / "custom.mplstyle"
    style.write_text("axes.prop_cycle: cycler('color', ['112233', '445566'])\n")
    vis = GraphRecoveryVisualization(str(tmp_path), mplstyle_path=str(style))
    assert vis.colors == ["#112233", "#445566"]


def test_save_and_log_without_a_run_closes_the_figure_and_logs_nothing(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wandb, "run", None)
    fig = plt.figure(figsize=(1, 1))
    vis.save_and_log_figure(fig, "panel/key")
    assert wandb_recorder.logged == []
    assert not plt.fignum_exists(fig.number)
    assert (
        capsys.readouterr().out
        == "Warning: wandb.run is None, skipping log for panel/key\n"
    )


def test_save_and_log_with_a_run_logs_a_300_dpi_png_under_the_key(
    vis: GraphRecoveryVisualization, wandb_recorder: Any, wandb_running: None
) -> None:
    fig = plt.figure(figsize=(1, 1))
    vis.save_and_log_figure(fig, "panel/key")
    assert len(wandb_recorder.logged) == 1
    payload, commit = wandb_recorder.logged[0]
    assert list(payload) == ["panel/key"] and commit is True
    image = payload["panel/key"].image
    assert image.format == "PNG"
    assert tuple(round(d) for d in image.info["dpi"]) == (300, 300)


def test_graph_info_summary_bars_table_and_file(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three graphs sorted alphabetically; a list reg_layer prints verbatim, a scalar as
    an int, a missing reg field as N/A; 15% headroom above the tallest bar.
    """
    figures = figure_capture(vis)
    graph_info: dict[str, dict[str, Any]] = {
        "regulatory": {
            "num_edges": 50,
            "num_nodes": 10,
            "avg_degree": 5.0,
            "reg_layer": 2.0,
        },
        "tflink": {"num_edges": 20, "num_nodes": 10, "avg_degree": 2.0},
        "physical": {
            "num_edges": 100,
            "num_nodes": 10,
            "avg_degree": 10.0,
            "reg_layer": [0, 1],
            "reg_head": 1,
        },
    }
    out = tmp_path / "summary.png"
    vis.plot_graph_info_summary(graph_info, save_path=str(out))

    assert out.read_bytes()[:8] == PNG_MAGIC
    assert capsys.readouterr().out == f"Graph info summary saved to: {out}\n"
    assert wandb_recorder.keys == ["graph_regularization_info/summary"]
    fig = figures[0]
    assert len(fig.axes) == 3
    assert fig.get_suptitle() == "Graph Statistics Summary"
    ax_edges, ax_degree, ax_table = fig.axes
    assert _heights(fig, 0) == [100.0, 50.0, 20.0]
    assert ax_edges.get_ylim() == pytest.approx((0.0, 115.0))
    assert _texts(fig, 0) == ["100", "50", "20"]
    assert ax_edges.get_title() == "Edge Count by Graph Type"
    assert _ticklabels(fig, 0) == ["physical", "regulatory", "tflink"]
    assert _faces(fig)[0] == pytest.approx(to_rgba("#7191A9", 0.8))
    assert _heights(fig, 1) == [10.0, 5.0, 2.0]
    assert ax_degree.get_ylim() == pytest.approx((0.0, 11.5))
    assert _texts(fig, 1) == ["10.0", "5.0", "2.0"]
    assert ax_degree.get_title() == "Graph Density (Avg Degree)"
    assert ax_degree.patches[0].get_facecolor() == pytest.approx(
        to_rgba("#D86E2F", 0.8)
    )
    cells = ax_table.tables[0].get_celld()
    rows = [[cells[(r, c)].get_text().get_text() for c in range(6)] for r in range(4)]
    assert rows == [
        ["Graph", "Nodes", "Edges", "Avg Degree", "Reg Layer", "Reg Head"],
        ["physical", "10", "100", "10.00", "[0, 1]", "1"],
        ["regulatory", "10", "50", "5.00", "2", "N/A"],
        ["tflink", "10", "20", "2.00", "N/A", "N/A"],
    ]
    assert cells[(0, 0)].get_facecolor() == pytest.approx(to_rgba("#4A4A4A"))
    assert cells[(1, 0)].get_facecolor() == pytest.approx(to_rgba("#FFFFFF"))
    assert cells[(2, 0)].get_facecolor() == pytest.approx(to_rgba("#F0F0F0"))
    assert cells[(3, 0)].get_facecolor() == pytest.approx(to_rgba("#FFFFFF"))

    # Without save_path nothing new is written; the wandb log still happens.
    vis.plot_graph_info_summary(graph_info)
    assert capsys.readouterr().out == ""
    assert wandb_recorder.keys == ["graph_regularization_info/summary"] * 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["summary.png"]


def test_graph_info_summary_without_graphs_prints_and_returns(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vis.plot_graph_info_summary({})
    assert capsys.readouterr().out == "No graph info to plot\n"
    assert wandb_recorder.logged == []


def test_recall_bars_sorted_by_graph_then_layer_and_colored_per_graph(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """Keys without ``_L`` keep their name as label and layer 0; ``Lx`` parses as layer 0."""
    figures = figure_capture(vis)
    vis.plot_edge_recovery_recall(
        {
            "physical_L1_H0": 0.5,
            "physical_L0_H1": 0.25,
            "regulatory_L0_H0": 0.75,
            "plain": 0.1,
            "weird_Lx_H0": 0.9,
        },
        num_epochs=7,
        stage="test",
    )
    assert wandb_recorder.keys == ["test_edge_recovery_summary/recall"]
    fig = figures[0]
    ax = fig.axes[0]
    assert _ticklabels(fig) == [
        "physical (L0)",
        "physical (L1)",
        "plain",
        "regulatory (L0)",
        "weird (L0)",
    ]
    assert _heights(fig) == [0.25, 0.5, 0.1, 0.75, 0.9]
    assert _texts(fig) == ["0.250", "0.500", "0.100", "0.750", "0.900"]
    assert _faces(fig) == pytest.approx(
        [
            to_rgba("#000000", 0.8),
            to_rgba("#000000", 0.8),
            to_rgba("#D86E2F", 0.8),
            to_rgba("#7191A9", 0.8),
            to_rgba("#6B8D3A", 0.8),
        ]
    )
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_title() == "Edge Recovery: Recall@Degree\nEpoch 7"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Graph + Layer", "Recall@Degree")


def test_recall_without_metrics_prints_and_returns(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vis.plot_edge_recovery_recall({}, num_epochs=1)
    assert capsys.readouterr().out == "No recall metrics to plot\n"
    assert wandb_recorder.logged == []


def test_precision_lines_styles_and_two_legends(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """Missing k falls to 0.0; layers {0, 2} map to '-' and '--'; legends split by role."""
    figures = figure_capture(vis)
    vis.plot_edge_recovery_precision(
        {
            "physical_L0_H0": {8: 0.9, 32: 0.7},
            "physical_L2_H0": {8: 0.5},
            "regulatory_L0_H0": {8: 0.3, 32: 0.2},
        },
        k_values=[8, 32],
        num_epochs=3,
    )
    assert wandb_recorder.keys == ["val_edge_recovery_summary/precision"]
    ax = figures[0].axes[0]
    assert [(line.get_color(), line.get_linestyle()) for line in ax.lines] == [
        ("#000000", "-"),
        ("#000000", "--"),
        ("#D86E2F", "-"),
    ]
    np.testing.assert_array_equal(ax.lines[0].get_xydata(), [[8, 0.9], [32, 0.7]])
    np.testing.assert_array_equal(ax.lines[1].get_xydata(), [[8, 0.5], [32, 0.0]])
    np.testing.assert_array_equal(ax.lines[2].get_xydata(), [[8, 0.3], [32, 0.2]])
    layer_legend = ax.get_legend()
    assert layer_legend is not None
    assert layer_legend.get_title().get_text() == "Reg Layer"
    assert _legend_labels(layer_legend) == ["Layer 0", "Layer 2"]
    graph_legends = [a for a in ax.artists if isinstance(a, Legend)]
    assert len(graph_legends) == 1
    assert graph_legends[0].get_title().get_text() == "Graph Type"
    assert _legend_labels(graph_legends[0]) == ["physical", "regulatory"]
    assert ax.get_xscale() == "log"
    assert _ticklabels(figures[0]) == ["8", "32"]
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_title() == "Edge Recovery: Precision@k\nEpoch 3"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("k (Top-k Attention)", "Precision@k")


def test_precision_keys_without_layer_info_fall_back_to_layer_zero(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """A key with no ``_L`` and a key whose layer is not an int both count as layer 0."""
    figures = figure_capture(vis)
    vis.plot_edge_recovery_precision(
        {"weird_Lx_H0": {8: 0.6}, "plain": {8: 0.4}}, k_values=[8], num_epochs=1
    )
    assert wandb_recorder.keys == ["val_edge_recovery_summary/precision"]
    ax = figures[0].axes[0]
    assert [(line.get_color(), line.get_linestyle()) for line in ax.lines] == [
        ("#000000", "-"),
        ("#D86E2F", "-"),
    ]
    np.testing.assert_array_equal(ax.lines[0].get_xydata(), [[8, 0.4]])
    np.testing.assert_array_equal(ax.lines[1].get_xydata(), [[8, 0.6]])
    assert _legend_labels(ax.get_legend()) == ["Layer 0"]
    graph_legends = [a for a in ax.artists if isinstance(a, Legend)]
    assert _legend_labels(graph_legends[0]) == ["plain", "weird"]


def test_precision_without_metrics_prints_and_returns(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vis.plot_edge_recovery_precision({}, k_values=[8], num_epochs=1)
    assert capsys.readouterr().out == "No precision metrics to plot\n"
    assert wandb_recorder.logged == []


def test_per_graph_plots_one_figure_per_graph_with_precision_entries(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """``c`` has recall but no precision and is skipped; ``x/y`` becomes ``x_y`` in the key.

    ylim is ``min(max(max precision, recall) * 1.15, 1.0)``: graph ``a`` caps at 1.0
    (``0.9 * 1.15 > 1``), graph ``b`` gets ``0.4 * 1.15 = 0.46``.
    """
    figures = figure_capture(vis)
    vis.plot_edge_recovery_per_graph(
        recall_metrics={"b": 0.4, "a": 0.9, "c": 0.5, "x/y": 0.2},
        precision_metrics={"a": {8: 0.2, 32: 0.6}, "b": {8: 0.1}, "x/y": {8: 0.1}},
        k_values=[8, 32],
        num_epochs=2,
    )
    assert wandb_recorder.keys == [
        "val_edge_recovery/per_graph/a",
        "val_edge_recovery/per_graph/b",
        "val_edge_recovery/per_graph/x_y",
    ]
    assert len(figures) == 3
    fig_a, fig_b, _ = figures
    assert _heights(fig_a) == [0.2, 0.6]
    assert fig_a.axes[0].get_ylim() == pytest.approx((0.0, 1.0))
    assert _heights(fig_b) == [0.1, 0.0]
    assert fig_b.axes[0].get_ylim() == pytest.approx((0.0, 0.46))
    ax_b = fig_b.axes[0]
    assert len(ax_b.lines) == 1
    np.testing.assert_array_equal(ax_b.lines[0].get_ydata(), [0.4, 0.4])
    assert ax_b.lines[0].get_color() == "#B73C39"
    assert _faces(fig_b)[0] == pytest.approx(to_rgba("#7191A9", 0.8))
    assert _legend_labels(ax_b.get_legend()) == ["Recall@degree = 0.400", "Precision@k"]
    assert _texts(fig_b) == ["0.100", "0.000"]
    assert _ticklabels(fig_b) == ["8", "32"]
    assert ax_b.get_title() == "Edge Recovery: b\nEpoch 2"
    assert (ax_b.get_xlabel(), ax_b.get_ylabel()) == ("k (Top-k Attention)", "Score")


def test_per_graph_without_either_metric_prints_and_returns(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vis.plot_edge_recovery_per_graph({"a": 0.1}, {}, k_values=[8], num_epochs=1)
    vis.plot_edge_recovery_per_graph({}, {"a": {8: 0.1}}, k_values=[8], num_epochs=1)
    assert capsys.readouterr().out == "No edge recovery metrics to plot\n" * 2
    assert wandb_recorder.logged == []


def test_edge_mass_bars_labels_and_baseline(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """Labels are ``graph (L<layer> H<head>)`` sorted by (graph, label); baseline at 0.5."""
    figures = figure_capture(vis)
    vis.plot_edge_mass_alignment(
        {
            "physical_L0_H1": 0.6,
            "physical_L0_H0": 0.7,
            "regulatory_L1_H0": 0.2,
            "plain": 0.3,
        },
        num_epochs=9,
    )
    assert wandb_recorder.keys == ["val_edge_recovery_summary/edge_mass"]
    fig = figures[0]
    ax = fig.axes[0]
    assert _ticklabels(fig) == [
        "physical (L0 H0)",
        "physical (L0 H1)",
        "plain",
        "regulatory (L1 H0)",
    ]
    assert _heights(fig) == [0.7, 0.6, 0.3, 0.2]
    assert _texts(fig) == ["0.700", "0.600", "0.300", "0.200"]
    assert _faces(fig) == pytest.approx(
        [
            to_rgba("#000000", 0.8),
            to_rgba("#000000", 0.8),
            to_rgba("#D86E2F", 0.8),
            to_rgba("#7191A9", 0.8),
        ]
    )
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.5, 0.5])
    assert _legend_labels(ax.get_legend()) == ["Random baseline (approx.)"]
    assert ax.get_ylim() == (0.0, 1.0)
    assert (
        ax.get_title()
        == "Edge-Mass Alignment: Fraction of Attention on Known Edges\nEpoch 9"
    )
    assert (ax.get_xlabel(), ax.get_ylabel()) == (
        "Graph + Layer + Head",
        "Edge-Mass Fraction",
    )


def test_edge_mass_without_metrics_prints_and_returns(
    vis: GraphRecoveryVisualization,
    wandb_recorder: Any,
    wandb_running: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vis.plot_edge_mass_alignment({}, num_epochs=1)
    assert capsys.readouterr().out == "No edge-mass metrics to plot\n"
    assert wandb_recorder.logged == []
