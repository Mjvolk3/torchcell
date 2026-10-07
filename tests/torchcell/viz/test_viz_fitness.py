# tests/torchcell/viz/test_viz_fitness.py
# [[tests.torchcell.viz.test_viz_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_viz_fitness.py
"""``torchcell.viz.fitness.box_plot`` and its simulated-data helpers.

Fixture: predictions ``[0.45, 0.55, 0.65, 0.95, 1.05, nan, 0.75]`` against measured
``[0.5, 0.6, 0.7, 0.9, 1.0, 0.3, nan]``. Rows with a NaN on either side drop, leaving five
pairs. Spearman is 1.000 (monotone); Pearson on those five pairs is 0.997 and its square,
R-squared, 0.993. The ten bins are ``[0, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, inf)``;
prediction 0.45 lands in bin 1 (median 0.5), 0.55 in bin 2 (0.6), 0.65 in bin 3 (0.7),
0.95 in bin 6 (0.9) and 1.05 in bin 7 (1.0); the other five bins are empty (NaN median).
Box i sits at ``i + 0.5`` with width 0.98, and each median line is shortened by 0.026 on
both ends: x from ``i + 0.036`` to ``i + 0.964``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from torchcell.viz import fitness  # noqa: E402

PREDICTIONS = [0.45, 0.55, 0.65, 0.95, 1.05, float("nan"), 0.75]
MEASURED = [0.5, 0.6, 0.7, 0.9, 1.0, 0.3, float("nan")]
EXPECTED_MEDIANS = [
    float("nan"),
    0.5,
    0.6,
    0.7,
    float("nan"),
    float("nan"),
    0.9,
    1.0,
    float("nan"),
    float("nan"),
]


def _medians(fig: Figure) -> list[Line2D]:
    """Median lines are the only Line2D drawn at linewidth 4.0."""
    return [line for line in fig.axes[0].lines if line.get_linewidth() == 4.0]


def test_box_plot_bins_predictions_and_reports_correlations() -> None:
    fig = fitness.box_plot(torch.tensor(MEASURED), torch.tensor(PREDICTIONS))
    ax = fig.axes[0]
    assert ax.get_title() == "Pearson: 0.997, Spearman: 1.000, R²: 0.993"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "0.0",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
        "1.1",
        "1.2",
        "Inf",
    ]
    medians = _medians(fig)
    assert len(medians) == 10
    assert len(ax.patches) == 10
    np.testing.assert_allclose(
        [float(np.asarray(m.get_ydata())[0]) for m in medians],
        EXPECTED_MEDIANS,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [np.asarray(m.get_xdata()).tolist() for m in medians],
        [[i + 0.036, i + 0.964] for i in range(10)],
    )
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Predicted growth", "Measured growth")
    assert ax.get_ylim() == pytest.approx((0.05, 1.25))
    np.testing.assert_allclose(ax.get_yticks(), np.arange(0.1, 1.3, 0.1))
    assert [t.get_text() for t in ax.texts] == ["(WT)", "(WT)"]
    assert tuple(fig.get_size_inches()) == pytest.approx((7.08, 6.0))
    assert fig.dpi == 140


def test_box_plot_wild_type_reference_lines() -> None:
    """22 black lines: the 20 zero-width whisker caps (``capwidths=0``; the module recolors
    boxes, whiskers and medians but never the caps), two per box at ``x = i + 0.5`` and,
    with one value per occupied bin, at that value's y (NaN for empty bins), then the
    horizontal line at y=1 and the vertical one at x=7 (the 1.0 tick). One gray tick line
    stands at each of the 11 bin edges.
    """
    fig = fitness.box_plot(np.array(MEASURED), np.array(PREDICTIONS))
    black = [
        (np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist())
        for line in fig.axes[0].lines
        if line.get_color() == "black"
    ]
    assert len(black) == 22
    caps, references = black[:20], black[20:]
    assert [x for x, _ in caps] == [
        [i + 0.5, i + 0.5] for i in range(10) for _ in (0, 1)
    ]
    np.testing.assert_allclose(
        [y for _, y in caps],
        [[m, m] for m in EXPECTED_MEDIANS for _ in (0, 1)],
        rtol=1e-6,
    )
    assert references == [([0, 1], [1, 1]), ([7.0, 7.0], [0, 1])]
    gray = [line for line in fig.axes[0].lines if line.get_color() == "#838383"]
    assert [np.asarray(line.get_xdata())[0] for line in gray] == list(range(11))


def test_box_plot_with_one_valid_pair_reports_not_available() -> None:
    """A single pair leaves the correlations undefined; the title shows N/A for each."""
    fig = fitness.box_plot(np.array([0.5, float("nan")]), np.array([0.5, 0.1]))
    assert fig.axes[0].get_title() == "Pearson: N/A, Spearman: N/A, R²: N/A"


def test_box_plot_with_constant_measured_values_reports_not_available() -> None:
    """Constant measured values: scipy returns nan (with ConstantInputWarning) for all
    three statistics rather than raising, so none of the ``except ValueError`` branches
    run; what is pinned is the nan -> "N/A" title formatting.
    """
    fig = fitness.box_plot(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.6, 0.7]))
    assert fig.axes[0].get_title() == "Pearson: N/A, Spearman: N/A, R²: N/A"


def test_generate_simulated_data_shapes_and_ranges() -> None:
    """N + n // 20 samples, both clipped to [0, 1.2], as float64 tensors."""
    np.random.seed(0)
    true_values, predictions = fitness.generate_simulated_data(1000)
    assert true_values.shape == predictions.shape == (1050,)
    assert true_values.dtype == predictions.dtype == torch.float64
    assert float(true_values.min()) >= 0.0 and float(true_values.max()) <= 1.2
    assert float(predictions.min()) >= 0.0 and float(predictions.max()) <= 1.2
    assert not torch.isnan(true_values).any() and not torch.isnan(predictions).any()


def test_generate_simulated_data_with_nan_special_cases() -> None:
    """N + n//20 + n//50 samples; NaN and inf land at the same rows in both tensors,
    and the n // 50 identical pairs at 0.5 survive (seed 0 leaves all 20).
    """
    np.random.seed(0)
    true_values, predictions = fitness.generate_simulated_data_with_nan(1000)
    assert true_values.shape == predictions.shape == (1070,)
    assert torch.equal(torch.isnan(true_values), torch.isnan(predictions))
    assert torch.equal(torch.isinf(true_values), torch.isinf(predictions))
    assert int(torch.isnan(true_values).sum()) == 12
    assert int(torch.isinf(true_values).sum()) == 6
    assert int(((true_values == 0.5) & (predictions == 0.5)).sum()) == 20


def test_main_builds_one_figure_and_shows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # test-quality: allow main() returns None; asserted through the recorded plt.show call and the live axes
    """``main`` draws the NaN-seeded box plot and hands exactly one figure to ``plt.show``."""
    shown: list[int] = []
    monkeypatch.setattr(
        plt, "show", lambda *a, **k: shown.append(len(plt.get_fignums()))
    )
    np.random.seed(0)
    fitness.main()
    assert shown == [1]
    assert plt.gca().get_xlabel() == "Predicted growth"


# Phase 24: linregress ValueError on constant predictions


def test_box_plot_constant_predictions_report_r_squared_not_available() -> None:
    """Three identical predictions make ``linregress`` raise ``ValueError`` (all x values
    identical, scipy 1.16); the helper turns that into NaN, shown as ``N/A``. Pearson and
    Spearman of a constant input are NaN without raising. All three predictions (0.5) land
    in bin 2, ``[0.5, 0.6)``, whose median is the middle measured value 0.6.

    The ``spearmanr`` ``except ValueError`` branch is unreachable here: scipy raises only on
    unequal lengths, which the shared NaN mask rules out.
    """
    fig = fitness.box_plot(np.array([0.4, 0.6, 0.8]), np.array([0.5, 0.5, 0.5]))
    assert fig.axes[0].get_title() == "Pearson: N/A, Spearman: N/A, R²: N/A"
    medians = [float(np.asarray(m.get_ydata())[0]) for m in _medians(fig)]
    assert medians[2] == pytest.approx(0.6)
    assert np.isnan(medians[:2] + medians[3:]).all()
    plt.close(fig)
