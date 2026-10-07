# tests/torchcell/adapters/test_bloom2019_adapter.py
"""Unit tests for the segregant-genotype and environment-response node methods."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import pytest

import torchcell.adapters.bloom2019_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.bloom2019_adapter import Bloom2019Adapter
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import (
    ArtifactRef,
    AssayType,
    EnvironmentResponsePhenotype,
    HaplotypeBlock,
    MeasurementType,
    SegregantGenotype,
    SegregantParent,
)
from torchcell.datasets.scerevisiae.bloom2019 import Bloom2019Dataset


def _genotype() -> SegregantGenotype:
    parent = SegregantParent(
        name="RMx",
        peter_strain_id="AAA",
        assembly_ref=ArtifactRef(
            tier="genomes",
            key="peter2018_1011_assemblies",
            path="1011Assemblies.tar.gz",
            member="GENOMES_ASSEMBLED/AAA_6.re.fa",
            sha256="5" * 64,
        ),
        engineered_background="RM MatAlpha AMN1-BY ho::HphMX flo8::NatMX",
    )
    by = parent.model_copy(
        update={
            "name": "BYa",
            "peter_strain_id": None,
            "assembly_ref": ArtifactRef(
                tier="genomes",
                key="sgd_S288C_R64-4-1_20230830",
                path="S288C_reference_sequence_R64-4-1_20230830.fsa",
                sha256="d" * 64,
            ),
        }
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
    assert "serialized_data" not in props
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
    assert "serialized_data" not in props


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        Bloom2019Adapter,
        "bloom2019_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=False, env_perturbation=True
        ),
        Bloom2019Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``Bloom2019Adapter`` loads ``conf/bloom2019_adapter.yaml``: 16 node and 14 edge methods (``environment response phenotype``; segregant genotype; NO perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

    The adapter is checked against the dataset ``dataset_adapter_map`` pairs it with
    (asserted to be the class above). The conf lists its methods in the order the
    adapter runs them, no edge dangles, every chunked entity node is linked, and the
    phenotype method matches that dataset's ``experiment_class``. The adapter keeps the dataset and the worker / chunk sizes
    it was given (3, 2, 500, 50), calls ``wandb.init`` once and logs the method table
    (event number, name, node/edge, factor or NaN for a non-chunked method) then the
    dataset name and the pinned start time; nothing is printed.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
