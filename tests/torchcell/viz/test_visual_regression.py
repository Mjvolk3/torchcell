# tests/torchcell/viz/test_visual_regression.py
# [[tests.torchcell.viz.test_visual_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_visual_regression.py
"""``torchcell.viz.visual_regression.Visualization`` on hand-worked inputs.

Correlation fixture (target 0): predictions ``[0.1, 0.2, nan-row, 0.3]`` against true
``[0.1, 0.2, nan, 0.4]``. The NaN row is dropped, leaving ``x = [0.1, 0.2, 0.3]``,
``y = [0.1, 0.2, 0.4]``: MSE ``(0 + 0 + 0.01)/3 = 3.333e-03``, Spearman 1.000, Pearson
``r = 3/sqrt(28/3) = 0.98198 -> 0.982`` (centered x ``[-1, 0, 1]/10``, centered y
``[-4, -1, 5]/30``: ``sum xy = 0.03``, ``sum x^2 = 0.02``, ``sum y^2 = 0.04667``).

Distribution fixture: ``y_true = [0,0,0,1,1,1,2,2,2]`` and ``y_pred = y_true + 1``. n=9
gives ``bins = int(sqrt(9)) = 3`` with edges ``[0, 2/3, 4/3, 2]``. Wasserstein distance
is the shift, ``1.0000``. True density is ``0.5`` per bin; pred falls ``[0, 3, 3]`` into
those edges (3 lands outside), so after epsilon-normalization ``p = [1/3, 1/3, 1/3]``,
``q = [~0, 1/2, 1/2]``, ``m = [1/6, 5/12, 5/12]`` and
``JS = 0.5 * (KL(p||m) + KL(q||m)) = 0.5 * (0.08229 + 0.18232) = 0.1323``.

Sample-metrics fixture (target 0): pred ``[1, 2, 3, 4]``, true ``[1, 2, 3, 5]``: MSE and
MAE ``0.25``, Spearman 1, Wasserstein ``0.25``, Pearson ``6.5/sqrt(5 * 8.75) = 0.98271``;
``bins = 2`` with edges ``[1, 3, 5]`` puts two points in each bin for both series, so
JS is exactly 0.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import umap  # noqa: E402
import wandb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from torchcell.viz.visual_regression import Visualization  # noqa: E402

NAN = float("nan")


def _legend_labels(fig: Figure) -> list[str]:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def _scatter_points(fig: Figure) -> NDArray[np.float64]:
    return np.asarray(fig.axes[0].collections[0].get_offsets(), dtype=np.float64)


@pytest.fixture
def vis(tmp_path: Path) -> Visualization:
    return Visualization(str(tmp_path / "run"))


def test_init_creates_the_figures_directory(tmp_path: Path) -> None:
    vis = Visualization(str(tmp_path / "run"), max_points=7)
    assert Path(vis.artifact_dir) == tmp_path / "run" / "figures"
    assert (tmp_path / "run" / "figures").is_dir()
    assert vis.artifact is None
    assert vis.max_points == 7


def test_init_wandb_artifact_records_name_and_type(
    vis: Visualization, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[tuple[str, str]] = []

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            made.append((name, type))

    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    vis.init_wandb_artifact("run-figs")
    assert made == [("run-figs", "figures")]
    assert isinstance(vis.artifact, FakeArtifact)


def test_get_base_title() -> None:
    vis_title = Visualization.get_base_title
    assert vis_title(Visualization.__new__(Visualization), "mse", 3) == (
        "Training Results\nLoss: mse\nEpochs: 3"
    )
    assert vis_title(Visualization.__new__(Visualization), "z", 3, "latent") == (
        "Training Results\nLatent: z\nEpochs: 3"
    )


def test_plot_correlations_scatter_title_and_identity_line(
    vis: Visualization,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    figures = figure_capture(vis)
    predictions = torch.tensor([[0.1, 1.0], [0.2, 2.0], [0.4, 3.0], [0.3, 4.0]])
    true_values = torch.tensor([[0.1, 1.0], [0.2, 2.0], [NAN, 3.0], [0.4, 4.0]])
    vis.plot_correlations(predictions, true_values, 0, "mse", 5, None, stage="val")
    assert wandb_recorder.keys == ["val/correlations_target_0"]
    assert wandb_recorder.logged[0][1] is True
    fig = figures[0]
    ax = fig.axes[0]
    np.testing.assert_allclose(
        _scatter_points(fig), [[0.1, 0.1], [0.2, 0.2], [0.3, 0.4]], rtol=1e-6
    )
    assert ax.get_title() == (
        "Training Results\nLoss: mse\nEpochs: 5\nTarget 0\n"
        "MSE=3.333e-03, n=3\nPearson=0.982, Spearman=1.000"
    )
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Predicted Target 0", "True Target 0")
    assert len(ax.lines) == 1
    diagonal = np.asarray(ax.lines[0].get_xydata())
    np.testing.assert_array_equal(diagonal[:, 0], diagonal[:, 1])
    assert (ax.lines[0].get_color(), ax.lines[0].get_linestyle()) == ("k", "--")


def test_plot_correlations_key_without_stage_and_bfloat16_input(
    vis: Visualization,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    """bfloat16 tensors are upcast before numpy; an empty stage drops the slash."""
    figures = figure_capture(vis)
    predictions = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16)
    true_values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16)
    vis.plot_correlations(predictions, true_values, 0, "mse", 1, None)
    assert wandb_recorder.keys == ["correlations_target_0"]
    np.testing.assert_array_equal(
        _scatter_points(figures[0]), [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )


def test_plot_correlations_subsamples_to_max_points(
    tmp_path: Path, wandb_recorder: Any, figure_capture: Callable[[Any], list[Figure]]
) -> None:
    """max_points=2 with four valid points keeps rows [2, 3] under seed 0
    (``np.random.choice(4, 2, replace=False)``); the title reports n=2.
    """
    vis = Visualization(str(tmp_path), max_points=2)
    figures = figure_capture(vis)
    predictions = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    true_values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    np.random.seed(0)
    vis.plot_correlations(predictions, true_values, 0, "mse", 1, None)
    np.testing.assert_allclose(
        _scatter_points(figures[0]), [[0.3, 3.0], [0.4, 4.0]], rtol=1e-6
    )
    assert "n=2\n" in figures[0].axes[0].get_title()


def test_plot_correlations_skips_with_fewer_than_two_points(
    vis: Visualization, wandb_recorder: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = torch.tensor([[0.1], [0.2]])
    true_values = torch.tensor([[1.0], [NAN]])
    vis.plot_correlations(predictions, true_values, 0, "mse", 1, None)
    assert (
        capsys.readouterr().out
        == "Not enough valid points for correlation plot for target 0. Skipping.\n"
    )
    assert wandb_recorder.logged == []


def test_plot_correlations_reports_a_failed_correlation_and_skips(
    vis: Visualization,
    wandb_recorder: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception from the correlation call is printed with the target; no figure."""

    def failing(*args: Any, **kwargs: Any) -> tuple[float, float]:
        raise ValueError("boom")

    monkeypatch.setattr("scipy.stats.pearsonr", failing)
    predictions = torch.tensor([[0.1], [0.2], [0.3]])
    true_values = torch.tensor([[1.0], [2.0], [3.0]])
    vis.plot_correlations(predictions, true_values, 0, "mse", 1, None)
    assert (
        capsys.readouterr().out == "Correlation calculation failed for target 0: boom\n"
    )
    assert wandb_recorder.logged == []


def test_plot_distribution_metrics_text_histograms_and_labels(
    vis: Visualization,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
) -> None:
    figures = figure_capture(vis)
    column = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    true_values = torch.stack([column, torch.full_like(column, 9.0)], dim=1)
    predictions = true_values + 1
    vis.plot_distribution(true_values, predictions, "mse", 0, 5, None, stage="val")
    assert wandb_recorder.keys == ["val/distribution_target_0"]
    fig = figures[0]
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["Wasserstein: 1.0000\nJS Div: 0.1323"]
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert [round(float(b.get_height()), 6) for b in bars] == [0.5] * 6
    assert [round(float(b.get_x()), 4) for b in bars] == [
        0.0,
        0.6667,
        1.3333,
        1.0,
        1.6667,
        2.3333,
    ]
    assert _legend_labels(fig) == ["True", "Predicted"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Target 0", "Density")
    assert ax.get_title() == (
        "Training Results\nLoss: mse\nEpochs: 5\nDistribution Matching Target 0"
    )


def test_plot_distribution_skips_with_fewer_than_two_points(
    vis: Visualization, wandb_recorder: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    true_values = torch.tensor([[1.0], [NAN]])
    vis.plot_distribution(true_values, true_values, "mse", 0, 1, None)
    assert (
        capsys.readouterr().out
        == "Not enough valid points for distribution plot for target 0. Skipping.\n"
    )
    assert wandb_recorder.logged == []


class FakeUMAP:
    """Records constructor kwargs and the array fitted; embeds row i at (2i, 2i + 1)."""

    constructed: list[dict[str, Any]] = []
    fitted: list[NDArray[Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Record the reducer's construction kwargs."""
        FakeUMAP.constructed.append(kwargs)

    def fit_transform(self, features: NDArray[Any]) -> NDArray[np.float64]:
        """Record the fitted array and return a deterministic 2-D embedding."""
        FakeUMAP.fitted.append(features.copy())
        return np.arange(features.shape[0] * 2, dtype=float).reshape(-1, 2)


def test_plot_umap_drops_nan_rows_and_colors_by_label(
    vis: Visualization,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeUMAP.constructed.clear()
    FakeUMAP.fitted.clear()
    monkeypatch.setattr(umap, "UMAP", FakeUMAP)
    figures = figure_capture(vis)
    features = torch.tensor([[1.0, 2.0], [NAN, 0.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.tensor([10.0, 20.0, 30.0, 40.0])
    vis.plot_umap(
        features, labels, "z", 0, 5, "ts-ignored", stage="val", title_type="latent"
    )
    assert wandb_recorder.keys == ["val/umap_z_target_0"]
    assert FakeUMAP.constructed == [
        {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
    ]
    np.testing.assert_array_equal(FakeUMAP.fitted[0], [[1, 2], [3, 4], [5, 6]])
    fig = figures[0]
    assert len(fig.axes) == 2  # scatter axes + colorbar
    np.testing.assert_array_equal(_scatter_points(fig), [[0, 1], [2, 3], [4, 5]])
    colors = fig.axes[0].collections[0].get_array()
    assert colors is not None
    np.testing.assert_array_equal(np.asarray(colors), [10.0, 30.0, 40.0])
    assert fig.axes[1].get_ylabel() == "Target 0"
    assert fig.axes[0].get_title() == (
        "Training Results\nLatent: z\nEpochs: 5\nUMAP Projection (z) for Target 0"
    )


def test_plot_umap_subsamples_to_max_points(
    tmp_path: Path,
    wandb_recorder: Any,
    figure_capture: Callable[[Any], list[Figure]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_points=2 with three valid rows fits rows [0, 2], labels kept in step."""
    FakeUMAP.constructed.clear()
    FakeUMAP.fitted.clear()
    monkeypatch.setattr(umap, "UMAP", FakeUMAP)
    vis = Visualization(str(tmp_path), max_points=2)
    figures = figure_capture(vis)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.tensor([10.0, 30.0, 50.0])
    np.random.seed(1)
    vis.plot_umap(features, labels, "z", 0, 1, None)
    # seed 1: np.random.choice(3, 2, replace=False) is [0, 2]
    np.testing.assert_array_equal(FakeUMAP.fitted[0], [[1.0, 2.0], [5.0, 6.0]])
    colors = figures[0].axes[0].collections[0].get_array()
    assert colors is not None
    np.testing.assert_array_equal(np.asarray(colors), [10.0, 50.0])
    assert wandb_recorder.keys == ["umap_z_target_0"]


def test_plot_umap_skips_with_fewer_than_two_valid_rows(
    vis: Visualization,
    wandb_recorder: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeUMAP.constructed.clear()
    monkeypatch.setattr(umap, "UMAP", FakeUMAP)
    features = torch.tensor([[1.0, 2.0], [NAN, 0.0]])
    vis.plot_umap(features, torch.tensor([1.0, 2.0]), "z", 2, 1, None)
    assert capsys.readouterr().out == (
        "Not enough valid features for UMAP plot for target 2. Skipping.\n"
    )
    assert FakeUMAP.constructed == []
    assert wandb_recorder.logged == []


def test_log_sample_metrics_values_and_the_empty_stage_key_prefix(
    vis: Visualization, wandb_recorder: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: an empty ``stage`` yields keys with a leading slash (``/MSE_target_0``).

    The plot methods drop the separator when ``stage == ""``; this method builds
    ``f"{prefix}/..."`` unconditionally. Target 1 has one valid label and is skipped with
    a printed notice rather than logged.
    """
    predictions = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, NAN], [4.0, NAN]])
    true_values = torch.tensor([[1.0, NAN], [2.0, NAN], [3.0, NAN], [5.0, 1.0]])
    vis.log_sample_metrics(predictions, true_values)
    assert capsys.readouterr().out == (
        "Not enough valid samples for sample metrics for target 1.\n"
    )
    assert len(wandb_recorder.logged) == 1
    payload, commit = wandb_recorder.logged[0]
    assert commit is None
    assert payload == {
        "/MSE_target_0": pytest.approx(0.25),
        "/MAE_target_0": pytest.approx(0.25),
        "/Pearson_target_0": pytest.approx(6.5 / np.sqrt(5 * 8.75), rel=1e-6),
        "/Spearman_target_0": pytest.approx(1.0),
        "/Wasserstein_target_0": pytest.approx(0.25),
        "/JS_div_target_0": pytest.approx(0.0, abs=1e-12),
    }


def test_log_sample_metrics_with_a_stage_prefix(
    vis: Visualization, wandb_recorder: Any
) -> None:
    predictions = torch.tensor([[1.0], [2.0], [3.0]])
    true_values = torch.tensor([[1.0], [2.0], [3.0]])
    vis.log_sample_metrics(predictions, true_values, stage="val")
    assert sorted(wandb_recorder.logged[0][0]) == [
        "val/JS_div_target_0",
        "val/MAE_target_0",
        "val/MSE_target_0",
        "val/Pearson_target_0",
        "val/Spearman_target_0",
        "val/Wasserstein_target_0",
    ]
    assert wandb_recorder.logged[0][0]["val/MSE_target_0"] == 0.0
    assert wandb_recorder.logged[0][0]["val/Pearson_target_0"] == pytest.approx(1.0)


def _record_calls(
    vis: Visualization, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    for name in (
        "plot_correlations",
        "plot_distribution",
        "plot_umap",
        "log_sample_metrics",
    ):

        def recorder(*args: Any, _name: str = name, **kwargs: Any) -> None:
            calls.append((_name, args, kwargs))

        monkeypatch.setattr(vis, name, recorder)
    return calls


def test_visualize_model_outputs_plots_only_informative_targets(
    vis: Visualization, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Column 1 is all zeros (padding) so target dims are [0, 2]; UMAP per latent x dim."""
    calls = _record_calls(vis, monkeypatch)
    true_values = torch.tensor([[1.0, 0.0, NAN], [2.0, 0.0, 0.5]])
    predictions = torch.zeros(2, 3)
    latents = {"z": torch.ones(2, 4)}
    vis.visualize_model_outputs(
        predictions, true_values, latents, "mse", 7, "ts", stage="val"
    )
    names = [c[0] for c in calls]
    assert names == [
        "plot_correlations",
        "plot_distribution",
        "plot_correlations",
        "plot_distribution",
        "plot_umap",
        "plot_umap",
        "log_sample_metrics",
    ]
    assert [c[1][2] for c in calls if c[0] == "plot_correlations"] == [0, 2]
    assert [c[1][3] for c in calls if c[0] == "plot_distribution"] == [0, 2]
    umap_calls = [c for c in calls if c[0] == "plot_umap"]
    assert [c[1][3] for c in umap_calls] == [0, 2]
    assert all(c[1][2] == "z" and c[2]["title_type"] == "latent" for c in umap_calls)
    torch.testing.assert_close(umap_calls[1][1][1], true_values[:, 2], equal_nan=True)
    assert calls[-1][2] == {"stage": "val"}
    assert calls[0][1][0] is predictions and calls[0][1][1] is true_values


def test_visualize_model_outputs_single_and_all_zero_targets_use_dim_zero(
    vis: Visualization, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_calls(vis, monkeypatch)
    vis.visualize_model_outputs(
        torch.zeros(3, 1), torch.zeros(3, 1), {}, "mse", 1, None
    )
    assert [c[0] for c in calls] == [
        "plot_correlations",
        "plot_distribution",
        "log_sample_metrics",
    ]
    assert calls[0][1][2] == 0
    calls.clear()
    vis.visualize_model_outputs(
        torch.zeros(3, 2), torch.zeros(3, 2), {}, "mse", 1, None
    )
    assert [c[1][2] for c in calls if c[0] == "plot_correlations"] == [0]
