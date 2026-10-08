# tests/torchcell/knowledge_graphs/test_dataset_visibility.py
# [[tests.torchcell.knowledge_graphs.test_dataset_visibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_dataset_visibility.py
"""The build-time visibility gate: private datasets never reach the public graph.

``ExperimentDataset.visibility`` is a ClassVar on the LOADER, so the gate reads it off
the class and a yaml listing a private loader is refused by name rather than silently
dropped. The private adapter map is separate from the public one, so a private dataset
has no adapter in a public build even if its name is configured.
"""

from __future__ import annotations

import pytest

from torchcell.data.experiment_dataset import Visibility
from torchcell.knowledge_graphs.dataset_adapter_map import (
    PRIVATE_DATASET_ADAPTER_MAP,
    PrivateDatasetRefused,
    build_adapter_map,
    dataset_adapter_map,
    refuse_private_datasets,
)


class _PublicLoader:
    visibility = Visibility.public


class _PrivateLoader:
    visibility = Visibility.private


class _UndeclaredLoader:
    """A class with no ``visibility`` at all (not an ``ExperimentDataset``)."""


class _PrivateAdapter:
    pass


def test_experiment_dataset_defaults_to_public() -> None:
    """Public is the default, so no existing loader changes behavior."""
    from torchcell.data.experiment_dataset import ExperimentDataset

    assert ExperimentDataset.visibility is Visibility.public
    assert {v.value for v in Visibility} == {"public", "private"}


def test_every_mapped_public_dataset_is_actually_public() -> None:
    """``dataset_adapter_map`` is the PUBLIC map; a private class in it is a mistake."""
    private = [
        cls.__name__
        for cls in dataset_adapter_map
        if getattr(cls, "visibility", Visibility.public) is Visibility.private
    ]
    assert private == []


def test_the_private_map_starts_empty_and_is_disjoint_from_the_public_one() -> None:
    """Private loaders register in their own map, never in the public one."""
    assert PRIVATE_DATASET_ADAPTER_MAP == {}
    assert set(PRIVATE_DATASET_ADAPTER_MAP) & set(dataset_adapter_map) == set()


def test_build_adapter_map_unions_the_private_map_only_when_asked() -> None:
    """``--include-private`` is the ONLY route from a private loader to an adapter."""
    PRIVATE_DATASET_ADAPTER_MAP[_PrivateLoader] = _PrivateAdapter
    try:
        assert _PrivateLoader not in build_adapter_map()
        assert (
            build_adapter_map(include_private=True)[_PrivateLoader] is _PrivateAdapter
        )
        assert set(build_adapter_map()) == set(dataset_adapter_map)
    finally:
        del PRIVATE_DATASET_ADAPTER_MAP[_PrivateLoader]


def test_refuse_private_datasets_names_every_private_class_and_the_flag() -> None:
    """A private loader in a public build raises, naming it and the flag that allows it."""
    refuse_private_datasets([_PublicLoader, _UndeclaredLoader])
    with pytest.raises(PrivateDatasetRefused) as excinfo:
        refuse_private_datasets([_PublicLoader, _PrivateLoader])
    message = str(excinfo.value)
    assert "_PrivateLoader" in message
    assert "--include-private" in message
    assert "_PublicLoader" not in message


def test_include_private_both_passes_the_gate_and_reaches_the_adapter() -> None:
    """The flag has to do both, or an in-house build passes the gate and then crashes."""
    PRIVATE_DATASET_ADAPTER_MAP[_PrivateLoader] = _PrivateAdapter
    try:
        refuse_private_datasets([_PublicLoader, _PrivateLoader], include_private=True)
        assert (
            build_adapter_map(include_private=True)[_PrivateLoader] is _PrivateAdapter
        )
    finally:
        del PRIVATE_DATASET_ADAPTER_MAP[_PrivateLoader]


def test_the_private_dataset_package_documents_the_contract() -> None:
    """The namespace's docstring is where the rule lives for a loader author."""
    import torchcell.datasets.private_torchcell as private

    doc = private.__doc__ or ""
    assert "Visibility.private" in doc
    assert "tc-data" in doc
    assert "dataset_name" in doc
