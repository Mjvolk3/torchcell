# tests/torchcell/adapters/test_environment_node_identity.py
"""Every environment-side edge addresses a node the adapter actually emits.

The media / temperature / environment-perturbation / environment ids are computed in
a node method AND again in each edge method. When those computations were separate
copies of ``sha256(json.dumps(model.model_dump()))`` they could only agree by
accident, and they agreed on the WRONG thing: the dump carries the stating dataset's
quotes, so two datasets on one medium got two media nodes. Now each class has one id
function and both callers use it, which is what these tests pin: the edges agree with
the nodes, and two datasets stating the same medium reach ONE media node.

The adapter methods are driven on a synthetic record (no dataset, no LMDB), the same
way ``test_optional_temperature_guard`` and ``test_crispr_construct_nodes`` do.
"""

from __future__ import annotations

from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels import schema as s
from torchcell.datamodels.identity import (
    environment_identity,
    environment_perturbation_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

_SHA = "0" * 64
_NACL_KEY = "FAPWRFPIFSIZLT-UHFFFAOYSA-M"


def _sourced(quote: str, citation_key: str) -> SourcedValue:
    return SourcedValue(
        value="20 g/L",
        quote=quote,
        provenance=Provenance(
            source_uri="paper.md", citation_key=citation_key, sha256=_SHA
        ),
    )


def _media(name: str, citation_key: str, quote: str) -> s.Media:
    """The same plate as two different papers would state it."""
    return s.Media(
        name=name,
        state="solid",
        is_synthetic=False,
        base_medium="YPD",
        components=[
            s.MediaComponent(
                compound=s.Compound(
                    name="D-glucose", inchikey="WQZGKKKJIJFFOK-GASJEMHNSA-N"
                ),
                role=s.MediaComponentRole.carbon_source,
                concentration=s.Concentration(
                    value=2.0, unit=s.ConcentrationUnit.percent_w_v
                ),
                provenance=[_sourced(quote, citation_key)],
                note=f"stated by {citation_key}",
            )
        ],
        provenance=[_sourced(quote, citation_key)],
    )


def _environment(temperature: s.Temperature | None) -> s.Environment:
    return s.Environment(
        media=_media("YPD plates", "mota2024", "20 g/L glucose"),
        temperature=temperature,
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(name="NaCl", inchikey=_NACL_KEY),
                concentration=s.Concentration(
                    value=0.4, unit=s.ConcentrationUnit.molar
                ),
            )
        ],
    )


class _FakeExperiment:
    """Minimal stand-in: the edge methods dump the experiment for its own id."""

    def __init__(self, environment: s.Environment) -> None:
        self.environment = environment

    def model_dump(self) -> dict[str, Any]:
        return {"experiment": "synthetic"}


def _record(environment: s.Environment) -> dict[str, Any]:
    return {"experiment": _FakeExperiment(environment)}


def _undecorated(method: Any) -> Any:
    return cast(Any, method).__wrapped__


def _adapter() -> CellAdapter:
    return CellAdapter.__new__(CellAdapter)


def test_the_node_ids_are_the_identity_projections_not_the_dumps() -> None:
    environment = _environment(s.Temperature(value=30.0))
    record = _record(environment)
    adapter = _adapter()

    media_node = _undecorated(CellAdapter._media_node)(adapter, record, "media")
    assert media_node.get_id() == identity_sha256(media_identity(environment.media))

    (temperature_node,) = _undecorated(CellAdapter._temperature_node)(
        adapter, record, "temperature"
    )
    assert temperature_node.get_id() == identity_sha256(
        temperature_identity(cast(s.Temperature, environment.temperature))
    )

    environment_node = _undecorated(CellAdapter._environment_node)(
        adapter, record, "environment"
    )
    assert environment_node.get_id() == identity_sha256(
        environment_identity(environment)
    )

    (perturbation_node,) = _undecorated(CellAdapter._environment_perturbation_node)(
        adapter, record, "environment perturbation"
    )
    assert perturbation_node.get_id() == identity_sha256(
        environment_perturbation_identity(environment.perturbations[0])
    )


def test_every_environment_edge_agrees_with_the_node_it_points_at() -> None:
    environment = _environment(s.Temperature(value=30.0))
    record = _record(environment)
    adapter = _adapter()

    environment_id = _undecorated(CellAdapter._environment_node)(
        adapter, record, "environment"
    ).get_id()
    media_id = _undecorated(CellAdapter._media_node)(adapter, record, "media").get_id()
    (temperature_node,) = _undecorated(CellAdapter._temperature_node)(
        adapter, record, "temperature"
    )
    (perturbation_node,) = _undecorated(CellAdapter._environment_perturbation_node)(
        adapter, record, "environment perturbation"
    )

    media_edge = _undecorated(CellAdapter._media_to_environment_edge)(
        adapter, record, "media to environment"
    )
    assert media_edge.get_source_id() == media_id
    assert media_edge.get_target_id() == environment_id

    (temperature_edge,) = _undecorated(CellAdapter._temperature_to_environment_edge)(
        adapter, record, "temperature to environment"
    )
    assert temperature_edge.get_source_id() == temperature_node.get_id()
    assert temperature_edge.get_target_id() == environment_id

    (perturbation_edge,) = _undecorated(
        CellAdapter._environment_perturbation_to_environment_edges
    )(adapter, record, "environment perturbation to environment")
    assert perturbation_edge.get_source_id() == perturbation_node.get_id()
    assert perturbation_edge.get_target_id() == environment_id

    experiment_edge = _undecorated(CellAdapter._environment_to_experiment_edge)(
        adapter, record, "environment to experiment"
    )
    assert experiment_edge.get_source_id() == environment_id


def test_two_datasets_stating_one_medium_reach_one_media_node() -> None:
    """The join the campaign is for: same plate, different papers, one node."""
    adapter = _adapter()
    mota = _record(
        s.Environment(media=_media("YPD (solid, 2% agar)", "mota2024", "20 g/L"))
    )
    bloom = _record(
        s.Environment(media=_media("YPD plates", "bloom2019", "2% glucose plates"))
    )
    mota_node = _undecorated(CellAdapter._media_node)(adapter, mota, "media")
    bloom_node = _undecorated(CellAdapter._media_node)(adapter, bloom, "media")
    assert mota_node.get_id() == bloom_node.get_id()
    # the node still carries the stating dataset's own full dump and label
    assert mota_node.get_properties()["name"] != bloom_node.get_properties()["name"]
    assert (
        mota_node.get_properties()["serialized_data"]
        != bloom_node.get_properties()["serialized_data"]
    )


def test_a_gapped_temperature_is_a_different_environment_node() -> None:
    adapter = _adapter()
    measured = _undecorated(CellAdapter._environment_node)(
        adapter, _record(_environment(s.Temperature(value=30.0))), "environment"
    )
    gapped = _undecorated(CellAdapter._environment_node)(
        adapter, _record(_environment(None)), "environment"
    )
    assert measured.get_id() != gapped.get_id()
    assert gapped.get_properties()["temperature"] is None


def test_a_dose_change_is_a_different_environment_and_perturbation_node() -> None:
    adapter = _adapter()
    weak = _environment(s.Temperature(value=30.0))
    strong = weak.model_copy(
        update={
            "perturbations": [
                s.SmallMoleculePerturbation(
                    compound=s.Compound(name="NaCl", inchikey=_NACL_KEY),
                    concentration=s.Concentration(
                        value=1.0, unit=s.ConcentrationUnit.molar
                    ),
                )
            ]
        }
    )
    weak_node = _undecorated(CellAdapter._environment_node)(
        adapter, _record(weak), "environment"
    )
    strong_node = _undecorated(CellAdapter._environment_node)(
        adapter, _record(strong), "environment"
    )
    assert weak_node.get_id() != strong_node.get_id()
    (weak_perturbation,) = _undecorated(CellAdapter._environment_perturbation_node)(
        adapter, _record(weak), "environment perturbation"
    )
    (strong_perturbation,) = _undecorated(CellAdapter._environment_perturbation_node)(
        adapter, _record(strong), "environment perturbation"
    )
    assert weak_perturbation.get_id() != strong_perturbation.get_id()
