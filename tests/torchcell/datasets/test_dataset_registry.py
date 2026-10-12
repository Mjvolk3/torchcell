# tests/torchcell/datasets/test_dataset_registry.py
# [[tests.torchcell.datasets.test_dataset_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_dataset_registry.py
"""Registry-wide contract for every registered dataset class, without building one.

Instances come from ``cls.__new__(cls)``: ``__init__`` runs ``process()``, which downloads
or reads the raw mirror, and the class-level contract (``experiment_class``,
``reference_class``, ``raw_file_names``) must hold before any data exists. A property
that reads ``self.root`` or another instance attribute fails the sweep, and that failure
is a finding about the loader, not about the test (Phase 1 of
[[plan.test-suite-buildout.2026.09.25]]).
"""

import importlib

import pytest

from torchcell.data.experiment_dataset import ExperimentDataset
from torchcell.datamodels.schema import Experiment, ExperimentReference
from torchcell.datasets import scerevisiae
from torchcell.datasets.dataset_registry import dataset_registry, register_dataset

REGISTERED = dict(sorted(dataset_registry.items()))

# Loaders with no raw files at all (built from the SGD genome, nothing downloaded).
# Strict: a loader listed here that declares raw files fails the sweep.
EMPTY_RAW = {"GeneEssentialitySgdDataset"}

# Loaders whose raw_file_names is a bare str, against the base class's list[str] (each
# carries a `type: ignore[override]`). PyG's raw_paths accepts a string, so the read path
# works; the fix touches six preprocess_raw call sites that only the data-gated tests
# exercise, so it waits for a PR that can cover them. Strict: a loader that starts
# returning a list must leave this set.
STR_RAW = {
    "DmfKuzmin2018Dataset",
    "DmiKuzmin2018Dataset",
    "SmfCostanzo2016Dataset",
    "SmfKuzmin2018Dataset",
    "SmfKuzmin2020Dataset",
    "TmfKuzmin2018Dataset",
    "TmiKuzmin2018Dataset",
}


def _shell(cls: type[ExperimentDataset]) -> ExperimentDataset:
    """The loader without its PyG init: no disk, no download."""
    instance = cls.__new__(cls)
    assert isinstance(instance, ExperimentDataset)
    return instance


def test_registry_holds_the_datasets_present_on_2026_09_26() -> None:
    """52 classes were registered when the sweep was written; a drop is a regression."""
    assert len(REGISTERED) >= 52
    assert set(scerevisiae.__all__) <= set(REGISTERED), sorted(
        set(scerevisiae.__all__) - set(REGISTERED)
    )
    # The class the package exports IS the registered one: a second class registered
    # under the same name would win the registry and lose here.
    for exported in scerevisiae.__all__:
        assert getattr(scerevisiae, exported) is REGISTERED[exported]


@pytest.mark.parametrize("name", list(REGISTERED))
def test_registered_class_contract(name: str) -> None:
    """Key equals the class name, the class is an ExperimentDataset under torchcell.datasets."""
    cls = REGISTERED[name]
    assert cls.__name__ == name
    assert issubclass(cls, ExperimentDataset)
    assert cls.__module__.startswith("torchcell.datasets.")
    module = importlib.import_module(cls.__module__)
    assert getattr(module, name) is cls


@pytest.mark.parametrize("name", list(REGISTERED))
def test_schema_classes_and_raw_files_are_declared_at_class_level(name: str) -> None:
    """experiment_class / reference_class are schema types; raw_file_names is a list of names.

    The seven STR_RAW loaders are held to the string they return today instead.
    """
    shell = _shell(REGISTERED[name])
    assert issubclass(shell.experiment_class, Experiment)
    assert issubclass(shell.reference_class, ExperimentReference)
    assert shell.experiment_class is not Experiment
    assert shell.reference_class is not ExperimentReference
    raw = shell.raw_file_names
    if name in STR_RAW:
        assert isinstance(raw, str) and raw
        return
    assert isinstance(raw, list)
    assert (len(raw) == 0) == (name in EMPTY_RAW)
    assert all(isinstance(f, str) and f for f in raw)
    assert len(set(raw)) == len(raw)


def test_register_dataset_returns_the_class_and_keys_by_name() -> None:
    """The decorator is an identity that adds one registry entry keyed by __name__."""

    class _ProbeDataset(ExperimentDataset):
        pass

    try:
        returned = register_dataset(_ProbeDataset)
        assert returned is _ProbeDataset
        assert dataset_registry["_ProbeDataset"] is _ProbeDataset
    finally:
        dataset_registry.pop("_ProbeDataset", None)
    assert "_ProbeDataset" not in dataset_registry
