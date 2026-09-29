# tests/torchcell/viz/conftest.py
# [[tests.torchcell.viz.conftest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/conftest.py
"""Fixtures shared by the ``torchcell.viz`` tests.

Every plotting class here ends by handing a figure to ``wandb.log`` through
``wandb.Image``; the tests never reach the real wandb. ``wandb_recorder`` swaps both
for recorders and hands back what was logged, ``figure_capture`` keeps the figure a
``save_and_log_figure`` call received so its axes can be inspected after the plot
method has closed it (``plt.close`` only detaches the figure from pyplot; the artists
stay intact), and the autouse fixture closes every figure between tests.
"""

from collections.abc import Callable, Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import wandb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402


class LoggedImage:
    """Stand-in for ``wandb.Image`` that keeps the PIL image it was built from."""

    def __init__(self, image: Image.Image) -> None:
        """Keep the decoded PNG."""
        self.image = image


class WandbRecorder:
    """Records every ``wandb.log`` payload and every ``wandb.Image`` construction."""

    def __init__(self) -> None:
        """Start with nothing logged."""
        self.logged: list[tuple[dict[str, Any], Any]] = []
        self.images: list[Image.Image] = []

    def log(self, payload: dict[str, Any], commit: Any = None) -> None:
        """Record one ``wandb.log`` call as ``(payload, commit)``."""
        self.logged.append((payload, commit))

    def make_image(self, image: Image.Image) -> LoggedImage:
        """Record the PIL image ``wandb.Image`` would wrap and return a stand-in."""
        self.images.append(image)
        return LoggedImage(image)

    @property
    def keys(self) -> list[str]:
        """Panel keys in the order they were logged, one key per ``wandb.log`` call."""
        return [key for payload, _ in self.logged for key in payload]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def wandb_recorder(monkeypatch: pytest.MonkeyPatch) -> WandbRecorder:
    """``wandb.log`` and ``wandb.Image`` replaced by recorders for the test's duration."""
    recorder = WandbRecorder()
    monkeypatch.setattr(wandb, "log", recorder.log)
    monkeypatch.setattr(wandb, "Image", recorder.make_image)
    return recorder


@pytest.fixture
def figure_capture(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], list[Figure]]:
    """Return ``install(vis) -> figures``: every figure ``vis.save_and_log_figure`` gets.

    The original method still runs, so the wandb key and the PNG round trip are
    exercised; the list just keeps a reference to each figure for inspection.
    """

    def install(vis: Any) -> list[Figure]:
        figures: list[Figure] = []
        original = vis.save_and_log_figure

        def recording(fig: Figure, name: str, timestamp_str: str | None = None) -> None:
            figures.append(fig)
            original(fig, name, timestamp_str)

        monkeypatch.setattr(vis, "save_and_log_figure", recording)
        return figures

    return install
