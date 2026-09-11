"""Unit tests for the pseudobulk-expression and environment-perturbation node methods."""

import hashlib
import json
from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    Media,
    PseudobulkExpressionPhenotype,
    SmallMoleculePerturbation,
    Temperature,
)


def _nacl_environment() -> Environment:
    return Environment(
        media=Media(name="YPD", state="liquid", is_synthetic=False),
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("sodium chloride"),
                concentration=Concentration(value=0.4, unit=ConcentrationUnit.molar),
            )
        ],
        aerobicity="aerobic",
        duration_hours=0.25,
    )


def test_pseudobulk_properties_keep_scalars_typed_and_dict_as_json() -> None:
    phenotype = PseudobulkExpressionPhenotype(
        expression_log2_ratio={"YAL001C": 0.5, "YAL002W": -1.0},
        dispersion=1.2,
        n_cells=40,
    )
    props = CellAdapter._pseudobulk_expression_properties(phenotype)
    assert json.loads(props["expression_log2_ratio"]) == {
        "YAL001C": 0.5,
        "YAL002W": -1.0,
    }
    assert props["dispersion"] == 1.2
    assert props["n_cells"] == 40
    assert props["measurement_type"] == "pseudobulk_scrnaseq_log2fc"
    assert props["label_name"] == "expression_log2_ratio"
    assert props["label_statistic_name"] == "dispersion"
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()


def test_environment_perturbation_node_is_content_addressed() -> None:
    perturbation = _nacl_environment().perturbations[0]
    node = CellAdapter._environment_perturbation_node_from(perturbation)
    expected_id = hashlib.sha256(
        json.dumps(perturbation.model_dump()).encode("utf-8")
    ).hexdigest()
    assert node.get_id() == expected_id
    assert node.get_label() == "environment perturbation"
    props = node.get_properties()
    assert props["perturbation_type"] == "small_molecule"
    assert props["compound_name"] == "sodium chloride"
    assert props["inchikey"] == "FAPWRFPIFSIZLT-UHFFFAOYSA-M"
    assert props["concentration_value"] == 0.4
    assert props["concentration_unit"] == "M"
    assert json.loads(props["serialized_data"]) == perturbation.model_dump()
    # the same perturbation in another environment yields the same node id
    assert (
        CellAdapter._environment_perturbation_node_from(
            _nacl_environment().perturbations[0]
        ).get_id()
        == expected_id
    )


def test_environment_perturbation_edge_targets_the_environment_hash() -> None:
    environment = _nacl_environment()
    environment_id = hashlib.sha256(
        json.dumps(environment.model_dump()).encode("utf-8")
    ).hexdigest()

    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.environment = environment  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    # call the undecorated method (data_chunker wraps it for the pool path)
    undecorated = cast(
        Any, CellAdapter._environment_perturbation_to_environment_edges
    ).__wrapped__
    edges = undecorated(
        adapter,
        {"experiment": experiment},
        "environment perturbation to environment (chunked)",
    )
    assert len(edges) == 1
    assert edges[0].get_target_id() == environment_id
    assert edges[0].get_label() == "environment perturbation member of"
    assert (
        edges[0].get_source_id()
        == CellAdapter._environment_perturbation_node_from(
            environment.perturbations[0]
        ).get_id()
    )
