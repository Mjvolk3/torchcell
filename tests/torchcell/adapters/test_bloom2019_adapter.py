# tests/torchcell/adapters/test_bloom2019_adapter.py
"""Unit tests for the segregant-genotype and environment-response node methods."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    HaplotypeBlock,
    MeasurementType,
    SegregantGenotype,
    SegregantParent,
)


def _genotype() -> SegregantGenotype:
    parent = SegregantParent(
        name="RMx",
        peter_strain_id="AAA",
        assembly_member="1011Assemblies.tar.gz::GENOMES_ASSEMBLED/AAA_6.re.fa",
        assembly_sha256="53540d09",
        engineered_background="RM MatAlpha AMN1-BY ho::HphMX flo8::NatMX",
    )
    by = parent.model_copy(
        update={"name": "BYa", "peter_strain_id": None, "assembly_member": "S288C"}
    )
    return SegregantGenotype(
        cross="A",
        segregant_id="A01_01",
        parent_1=by,
        parent_2=parent,
        blocks=[
            HaplotypeBlock(
                chromosome="chrI", start=693, end=9000, parent=1, n_markers=5
            ),
            HaplotypeBlock(
                chromosome="chrI", start=9500, end=20000, parent=2, n_markers=7
            ),
        ],
        call_method="R/qtl argmax.geno hard call",
        marker_matrix_sha256="21c4bed2",
    )


def test_segregant_genotype_node_is_content_addressed() -> None:
    genotype = _genotype()
    node = CellAdapter._segregant_genotype_node_from(genotype)
    expected = hashlib.sha256(
        json.dumps(genotype.model_dump()).encode("utf-8")
    ).hexdigest()
    assert node.get_id() == expected
    assert node.get_label() == "segregant genotype"
    props = node.get_properties()
    assert props["cross"] == "A" and props["segregant_id"] == "A01_01"
    assert props["parent_1"] == "BYa" and props["parent_2"] == "RMx"
    assert props["n_blocks"] == 2
    assert json.loads(props["serialized_data"]) == genotype.model_dump()
    assert CellAdapter._segregant_genotype_node_from(_genotype()).get_id() == expected


def test_genotype_to_experiment_edge_hashes_the_same_way() -> None:
    genotype = _genotype()

    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.genotype = genotype  # type: ignore[attr-defined]
    experiment.model_dump = lambda: {"genotype": genotype.model_dump()}  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    undecorated = cast(Any, CellAdapter._genotype_to_experiment_edge).__wrapped__
    edge = undecorated(
        adapter, {"experiment": experiment}, "genotype to experiment (chunked)"
    )
    assert (
        edge.get_source_id()
        == CellAdapter._segregant_genotype_node_from(genotype).get_id()
    )


def test_environment_response_properties_project_the_typed_axes() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.control_regression_residual,
        assay_type=AssayType.colony_size_array,
        environment_response=-0.75,
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["environment_response"] == -0.75
    assert props["environment_response_se"] is None
    assert props["measurement_type"] == "control_regression_residual"
    assert props["assay_type"] == "colony_size_array"
    assert props["label_name"] == "environment_response"
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()
