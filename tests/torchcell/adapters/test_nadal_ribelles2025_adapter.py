"""Unit tests for the pseudobulk-expression and environment-perturbation node methods."""

import json
from typing import Any, cast

import pytest

import torchcell.adapters.nadal_ribelles2025_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.nadal_ribelles2025_adapter import (
    NadalRibellesPerturbSeq2025Adapter,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.identity import (
    environment_identity,
    environment_perturbation_identity,
    identity_sha256,
)
from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    Media,
    PseudobulkExpressionPhenotype,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae.nadal_ribelles2025 import (
    NadalRibellesPerturbSeq2025Dataset,
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
    assert "serialized_data" not in props


def test_environment_perturbation_node_is_content_addressed() -> None:
    perturbation = _nacl_environment().perturbations[0]
    node = CellAdapter._environment_perturbation_node_from(perturbation)
    expected_id = identity_sha256(environment_perturbation_identity(perturbation))
    assert node.get_id() == expected_id
    assert node.get_label() == "environment perturbation"
    props = node.get_properties()
    assert props["perturbation_type"] == "small_molecule"
    assert props["compound_name"] == "sodium chloride"
    assert props["inchikey"] == "FAPWRFPIFSIZLT-UHFFFAOYSA-M"
    assert props["concentration_value"] == 0.4
    assert props["concentration_unit"] == "M"
    assert "serialized_data" not in props
    # the same perturbation in another environment yields the same node id
    assert (
        CellAdapter._environment_perturbation_node_from(
            _nacl_environment().perturbations[0]
        ).get_id()
        == expected_id
    )


def test_environment_perturbation_edge_targets_the_environment_hash() -> None:
    environment = _nacl_environment()
    environment_id = identity_sha256(environment_identity(environment))

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


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        NadalRibellesPerturbSeq2025Adapter,
        "nadal_ribelles_perturbseq2025_adapter.yaml",
        Shape("pseudobulk expression phenotype", env_perturbation=True),
        NadalRibellesPerturbSeq2025Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``NadalRibellesPerturbSeq2025Adapter`` loads ``conf/nadal_ribelles_perturbseq2025_adapter.yaml``: 17 node and 15 edge methods (``pseudobulk expression phenotype``; gene-keyed genotype; perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
