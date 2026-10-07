# tests/torchcell/viz/test_genetic_interaction_score.py
# [[tests.torchcell.viz.test_genetic_interaction_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_genetic_interaction_score.py
"""``torchcell.viz.genetic_interaction_score.box_plot`` and its simulated-data helpers.

Fixture: predictions ``[-0.45, -0.3, -0.2, 0.05, 0.3, nan]`` against measured
``[-0.5, -0.25, -0.1, 0.1, 0.2, 0.0]``; the NaN pair drops. Spearman 1.000 (monotone),
Pearson 0.961, R-squared 0.924 (``0.9612**2``). Bins are
``(-inf, -.40, -.32, -.24, -.16, -.08, 0, .08, .16, .24, inf)``: -0.45 falls in bin 0
(median -0.5), -0.3 in bin 2 (-0.25), -0.2 in bin 3 (-0.1), 0.05 in bin 6 (0.1), 0.3 in
bin 9 (0.2); the rest are empty. Median lines run from ``i + 0.036`` to ``i + 0.964``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from torchcell.viz import genetic_interaction_score as gis  # noqa: E402

PREDICTIONS = [-0.45, -0.3, -0.2, 0.05, 0.3, float("nan")]
MEASURED = [-0.5, -0.25, -0.1, 0.1, 0.2, 0.0]
NAN = float("nan")
EXPECTED_MEDIANS = [-0.5, NAN, -0.25, -0.1, NAN, NAN, 0.1, NAN, NAN, 0.2]


def _medians(fig: Figure) -> list[Line2D]:
    return [line for line in fig.axes[0].lines if line.get_linewidth() == 4.0]


def test_box_plot_bins_predictions_and_reports_correlations() -> None:
    fig = gis.box_plot(torch.tensor(MEASURED), torch.tensor(PREDICTIONS))
    ax = fig.axes[0]
    assert ax.get_title() == "Pearson: 0.961, Spearman: 1.000, R²: 0.924"
    labels = ax.get_xticklabels()
    assert [t.get_text() for t in labels] == [
        "-Inf",
        "-0.40",
        "-0.32",
        "-0.24",
        "-0.16",
        "-0.08",
        "0.00",
        "0.08",
        "0.16",
        "0.24",
        "Inf",
    ]
    assert {t.get_rotation() for t in labels} == {45.0}
    medians = _medians(fig)
    assert len(medians) == 10 and len(ax.patches) == 10
    np.testing.assert_allclose(
        [float(np.asarray(m.get_ydata())[0]) for m in medians],
        EXPECTED_MEDIANS,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [np.asarray(m.get_xdata()).tolist() for m in medians],
        [[i + 0.036, i + 0.964] for i in range(10)],
    )
    assert (ax.get_xlabel(), ax.get_ylabel()) == (
        "Predicted genetic interaction",
        "Measured genetic interaction",
    )
    assert ax.get_ylim() == pytest.approx((-0.85, 0.45))
    np.testing.assert_allclose(ax.get_yticks(), [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    assert tuple(fig.get_size_inches()) == pytest.approx((7.08, 6.0))


def test_box_plot_neutral_reference_lines() -> None:
    """22 black lines: the 20 zero-width whisker caps (``capwidths=0``; never recolored),
    two per box at ``x = i + 0.5`` and at the bin's single value (NaN when empty), then
    the horizontal line at y=0 and the vertical one at x=6 (the 0.00 tick).
    """
    fig = gis.box_plot(np.array(MEASURED), np.array(PREDICTIONS))
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
    assert references == [([0, 1], [0, 0]), ([6, 6], [0, 1])]


def test_box_plot_with_one_valid_pair_reports_not_available() -> None:
    fig = gis.box_plot(np.array([0.1, NAN]), np.array([0.1, 0.2]))
    assert fig.axes[0].get_title() == "Pearson: N/A, Spearman: N/A, R²: N/A"


def test_generate_simulated_data_layout() -> None:
    """N + n // 50 rows; NaN rows coincide; measured values stay inside [-0.8, 0.8]
    while the unclipped extreme predictions may not (seed 0 gives 11 NaN rows).
    """
    np.random.seed(0)
    true_values, predictions = gis.generate_simulated_data(1000)
    assert true_values.shape == predictions.shape == (1020,)
    assert torch.equal(torch.isnan(true_values), torch.isnan(predictions))
    assert int(torch.isnan(true_values).sum()) == 11
    finite = true_values[~torch.isnan(true_values)]
    assert float(finite.min()) >= -0.8 and float(finite.max()) <= 0.8


def test_generate_simulated_data_with_nan_special_cases() -> None:
    """N + n//50 + 2 * (n//100) rows; NaN and inf coincide; n // 100 identical 0.1 pairs
    and n // 100 pairs drawn from {0.2, 0.3} survive the shuffle (seed 0 draws 6 at 0.2
    and 4 at 0.3, and no other row lands exactly on those values).
    """
    np.random.seed(0)
    true_values, predictions = gis.generate_simulated_data_with_nan(1000)
    assert true_values.shape == predictions.shape == (1040,)
    assert torch.equal(torch.isnan(true_values), torch.isnan(predictions))
    assert torch.equal(torch.isinf(true_values), torch.isinf(predictions))
    assert int(torch.isnan(true_values).sum()) == 49
    assert int(torch.isinf(true_values).sum()) == 10
    assert int(((true_values == 0.1) & (predictions == 0.1)).sum()) == 10
    two_value = (true_values == predictions) & torch.isin(
        true_values, torch.tensor([0.2, 0.3], dtype=torch.float64)
    )
    assert int(two_value.sum()) == 10
    assert int((true_values[two_value] == 0.2).sum()) == 6
    assert int((true_values[two_value] == 0.3).sum()) == 4


def test_main_builds_one_figure_and_shows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # test-quality: allow main() returns None; asserted through the recorded plt.show call and the live axes
    """``main`` draws the simulated box plot and hands exactly one figure to ``plt.show``."""
    shown: list[int] = []
    monkeypatch.setattr(
        plt, "show", lambda *a, **k: shown.append(len(plt.get_fignums()))
    )
    np.random.seed(0)
    gis.main()
    assert shown == [1]
    assert plt.gca().get_xlabel() == "Predicted genetic interaction"


# Phase 24: linregress ValueError on constant predictions


def test_box_plot_constant_predictions_report_r_squared_not_available() -> None:
    """Three identical predictions make ``linregress`` raise ``ValueError`` (all x values
    identical, scipy 1.16); the helper turns that into NaN, shown as ``N/A``. The
    ``spearmanr`` ``except ValueError`` branch is unreachable: scipy raises only on unequal
    lengths, which the shared NaN mask rules out. All three predictions (0.1) land in
    bin 7, ``[0.08, 0.16)``, whose median is the middle measured value 0.0.
    """
    fig = gis.box_plot(np.array([-0.1, 0.0, 0.1]), np.array([0.1, 0.1, 0.1]))
    assert fig.axes[0].get_title() == "Pearson: N/A, Spearman: N/A, R²: N/A"
    medians = [float(np.asarray(m.get_ydata())[0]) for m in _medians(fig)]
    assert medians[7] == 0.0
    assert np.isnan(medians[:7] + medians[8:]).all()
    plt.close(fig)
