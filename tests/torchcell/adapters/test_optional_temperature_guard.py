# tests/torchcell/adapters/test_optional_temperature_guard.py
"""``Environment.temperature`` is optional, so the adapter must survive a gapped one.

The schema allows a curation layer that never carried a temperature to record a typed
gap instead of guessing one, and the adapter used to walk straight through the field.
This is the behavioral half of the guard; the static half (no adapter chain dereferences
past an Optional schema field) lives in
``tests/torchcell/datamodels/test_ontology_coherence.py``.

The second thing it pins is that a PRESENT temperature is emitted exactly as before,
because every served dataset's temperature nodes are content-addressed on this output.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
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


def test_a_present_temperature_is_emitted_byte_identically() -> None:
    """The node id and properties a served dataset was built on must not move."""
    temperature = Temperature(value=30.0, unit=TemperatureUnit.celsius)
    adapter = CellAdapter.__new__(CellAdapter)
    nodes = _undecorated(CellAdapter._temperature_node)(
        adapter, _record(temperature), "temperature (chunked)"
    )
    assert len(nodes) == 1
    node = nodes[0]
    assert (
        node.get_id()
        == hashlib.sha256(
            json.dumps(temperature.model_dump()).encode("utf-8")
        ).hexdigest()
    )
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
    assert (
        edges[0].get_target_id()
        == hashlib.sha256(
            json.dumps(_environment(temperature).model_dump()).encode("utf-8")
        ).hexdigest()
    )

    environment_node = _undecorated(CellAdapter._environment_node)(
        adapter, _record(temperature), "environment (chunked)"
    )
    assert environment_node.get_properties()["temperature"] == 30.0
