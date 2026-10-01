# tests/torchcell/benchmark/conftest.py
# [[tests.torchcell.benchmark.conftest]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/conftest.py
"""A hand-built benchmark bundle small enough to grade by hand.

``toy-fitness`` has two train records, four validation records and four test records
with one target, ``fitness``. Validation labels are 1, 2, 3, 4 and test labels are
0.5, 1.0, 1.5, 2.5, so every metric a test asserts can be checked on paper.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from torchcell.benchmark.bundle import BenchmarkBundle, write_bundle
from torchcell.benchmark.submission import Split

SLUG = "toy-fitness"
TARGET = "fitness"
BUILT_AT = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
SPLITS: dict[Split, list[str]] = {
    Split.TRAIN: ["t1", "t2"],
    Split.VAL: ["v1", "v2", "v3", "v4"],
    Split.TEST: ["s1", "s2", "s3", "s4"],
}
LABELS: dict[tuple[str, str], float] = {
    ("v1", TARGET): 1.0,
    ("v2", TARGET): 2.0,
    ("v3", TARGET): 3.0,
    ("v4", TARGET): 4.0,
    ("s1", TARGET): 0.5,
    ("s2", TARGET): 1.0,
    ("s3", TARGET): 1.5,
    ("s4", TARGET): 2.5,
}


def write_toy_bundle(datasets_root: Path) -> None:
    """Write the ``toy-fitness`` bundle under ``datasets_root``."""
    write_bundle(
        datasets_root,
        slug=SLUG,
        title="Toy fitness",
        description="Eight records with one target, for tests.",
        loader_class="ToyFitnessDataset",
        citation_key="toy2026",
        version="1",
        primary_metric="pearson",
        splits=SPLITS,
        values=LABELS,
        built_at=BUILT_AT,
    )


def predictions_csv(predictions: Mapping[tuple[str, str], float]) -> bytes:
    """A predictions CSV for ``toy-fitness`` holding ``predictions``, in key order."""
    lines = ["record_id,split,target,prediction"]
    for (record_id, target), value in predictions.items():
        split = "val" if record_id.startswith("v") else "test"
        lines.append(f"{record_id},{split},{target},{value!r}")
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def labels() -> dict[tuple[str, str], float]:
    """The labels of ``toy-fitness`` (a fresh copy, so a test may edit it)."""
    return dict(LABELS)


@pytest.fixture
def to_csv() -> Callable[[Mapping[tuple[str, str], float]], bytes]:
    """The function that renders a predictions CSV for ``toy-fitness``."""
    return predictions_csv


@pytest.fixture
def datasets_root(tmp_path: Path) -> Path:
    """A datasets root holding the ``toy-fitness`` bundle."""
    root = tmp_path / "datasets"
    root.mkdir()
    write_toy_bundle(root)
    return root


@pytest.fixture
def bundle(datasets_root: Path) -> BenchmarkBundle:
    """The loaded ``toy-fitness`` bundle."""
    return BenchmarkBundle.load(datasets_root / SLUG)
