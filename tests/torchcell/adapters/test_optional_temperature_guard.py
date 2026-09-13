# tests/torchcell/adapters/test_optional_temperature_guard.py
"""``Environment.temperature`` is optional, so the adapter must survive a gapped one.

The schema allows a curation layer that never carried a temperature to record a typed
gap instead of guessing one, and the adapter used to walk straight through the field.
This is the behavioral half of the guard; the static half (no adapter chain dereferences
past an Optional schema field) lives in
``tests/torchcell/datamodels/test_ontology_coherence.py``.

The second thing it pins is that a PRESENT temperature still emits the same node
PROPERTIES, and that its id is the temperature's identity projection (value + typed
unit) rather than a hash of the whole dump.
"""

from __future__ import annotations

import json
from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.identity import (
    environment_identity,
    identity_sha256,
    temperature_identity,
)
from torchcell.datamodels.schema import Environment, Media, Temperature, TemperatureUnit

MEDIA = Media(name="YPD", state="liquid", is_synthetic=False)


def _environment(temperature: Temperature | None) -> Environment:
    return Environment(media=MEDIA, temperature=temperature)


def _record(temperature: Temperature | None) -> dict[str, Any]:
    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.environment = _environment(temperature)  # type: ignore[attr-defined]
    return {"experiment": experiment}


def _undecorated(method: Any) -> Any:
    return cast(Any, method).__wrapped__


def test_a_gapped_temperature_emits_no_temperature_node() -> None:
    adapter = CellAdapter.__new__(CellAdapter)
    nodes = _undecorated(CellAdapter._temperature_node)(
        adapter, _record(None), "temperature (chunked)"
    )
    assert nodes == []


def test_a_gapped_temperature_emits_no_temperature_edge() -> None:
    adapter = CellAdapter.__new__(CellAdapter)
    edges = _undecorated(CellAdapter._temperature_to_environment_edge)(
        adapter, _record(None), "temperature to environment (chunked)"
    )
    assert edges == []


def test_a_gapped_temperature_still_yields_an_environment_node_with_a_null() -> None:
    adapter = CellAdapter.__new__(CellAdapter)
    node = _undecorated(CellAdapter._environment_node)(
        adapter, _record(None), "environment (chunked)"
    )
    assert node.get_label() == "environment"
    assert node.get_properties()["temperature"] is None


def test_a_present_temperature_is_emitted_with_its_identity_id() -> None:
    """The properties do not move; the id is the composition, not the dump."""
    temperature = Temperature(value=30.0, unit=TemperatureUnit.celsius)
    adapter = CellAdapter.__new__(CellAdapter)
    nodes = _undecorated(CellAdapter._temperature_node)(
        adapter, _record(temperature), "temperature (chunked)"
    )
    assert len(nodes) == 1
    node = nodes[0]
    assert node.get_id() == identity_sha256(temperature_identity(temperature))
    assert node.get_label() == "temperature"
    props = node.get_properties()
    assert props["value"] == 30.0
    assert props["unit"] == TemperatureUnit.celsius
    assert json.loads(props["serialized_data"]) == temperature.model_dump()

    edges = _undecorated(CellAdapter._temperature_to_environment_edge)(
        adapter, _record(temperature), "temperature to environment (chunked)"
    )
    assert len(edges) == 1
    assert edges[0].get_source_id() == node.get_id()
    assert edges[0].get_label() == "temperature member of"
    assert edges[0].get_target_id() == identity_sha256(
        environment_identity(_environment(temperature))
    )

    environment_node = _undecorated(CellAdapter._environment_node)(
        adapter, _record(temperature), "environment (chunked)"
    )
    assert environment_node.get_properties()["temperature"] == 30.0
