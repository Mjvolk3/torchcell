# tests/torchcell/adapters/test_hale2024_adapter.py
# [[tests.torchcell.adapters.test_hale2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_hale2024_adapter.py
"""The perturbed-segregant node methods and the Hale 2024 adapter constructor.

A ``PerturbedSegregantGenotype`` node is content-addressed by the sha256 of the whole
genotype dump and projects BOTH halves (the background's cross, segregant, role, parents
and block count; the perturbations' gene lists). Its background is emitted as the plain
``segregant genotype`` node it is, with the id that node method already uses, and the
``segregant background of`` edge joins the two ids; the generic genotype-to-experiment,
perturbation and CRISPR-construct methods address the composite unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import pytest

import torchcell.adapters.hale2024_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.hale2024_adapter import Hale2024Adapter
from torchcell.datamodels.schema import (
    ArtifactRef,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    HaplotypeBlock,
    PerturbedSegregantGenotype,
    SegregantGenotype,
    SegregantParent,
)
from torchcell.datasets.scerevisiae.hale2024 import CrispriSegregantHale2024Dataset


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj.model_dump()).encode("utf-8")).hexdigest()


def _genotype(role: str = "segregant") -> PerturbedSegregantGenotype:
    parent = SegregantParent(
        name="BY",
        assembly_ref=ArtifactRef(
            tier="genomes",
            key="sgd_S288C_R64-4-1_20230830",
            path="x.fsa",
            sha256="d" * 64,
        ),
        engineered_background="BY fcy1Δ",
    )
    three_s = parent.model_copy(update={"name": "3S", "engineered_background": "3S"})
    parents = [1] if role == "parent_1" else [1, 2]
    background = SegregantGenotype(
        cross="BYx3S",
        segregant_id="x_BY_BYx" if role == "parent_1" else "BYa_1C_A10",
        parent_1=parent,
        parent_2=three_s,
        blocks=[
            HaplotypeBlock(
                chromosome="chrI",
                start=10 * i + 1,
                end=10 * i + 5,
                parent=p,  # type: ignore[arg-type]
                n_markers=3,
            )
            for i, p in enumerate(parents)
        ],
        call_method="hard calls",
        marker_matrix_sha256="0" * 64,
    )
    return PerturbedSegregantGenotype(
        background=background,
        background_role=role,  # type: ignore[arg-type]
        perturbations=[
            CrisprInterferencePerturbation(
                systematic_gene_name="YDR116C",
                perturbed_gene_name="MRPL1",
                crispr=CrisprConstruct(
                    effector="dCas9-Mxi1",
                    guide_sequence="TTTTCTCTTTTCGCTTAAAA",
                    n_guides=1,
                ),
            )
        ],
    )


def test_perturbed_segregant_node_projects_both_halves() -> None:
    genotype = _genotype()
    node = CellAdapter._perturbed_segregant_genotype_node_from(genotype)
    assert node.get_id() == _sha(genotype)
    assert node.get_label() == "perturbed segregant genotype"
    assert node.get_properties() == {
        "id": _sha(genotype),
        "preferred_id": "perturbed segregant genotype",
        "cross": "BYx3S",
        "segregant_id": "BYa_1C_A10",
        "background_role": "segregant",
        "parent_1": "BY",
        "parent_2": "3S",
        "n_blocks": 2,
        "systematic_gene_names": ["YDR116C"],
        "perturbed_gene_names": ["MRPL1"],
        "perturbation_types": ["crispr_interference"],
    }


def _call(method: Any, genotype: PerturbedSegregantGenotype, name: str) -> Any:
    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.genotype = genotype  # type: ignore[attr-defined]
    experiment.model_dump = lambda: {"genotype": genotype.model_dump()}  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    return cast(Any, method).__wrapped__(adapter, {"experiment": experiment}, name)


def test_background_node_and_edge_join_the_mosaic_to_the_composite() -> None:
    genotype = _genotype("parent_1")
    background_node = _call(
        CellAdapter._segregant_background_node,
        genotype,
        "segregant background (chunked)",
    )
    assert background_node.get_label() == "segregant genotype"
    assert background_node.get_id() == _sha(genotype.background)
    assert background_node.get_properties()["segregant_id"] == "x_BY_BYx"
    edge = _call(
        CellAdapter._segregant_background_to_genotype_edge,
        genotype,
        "segregant background to genotype (chunked)",
    )
    assert edge.get_source_id() == _sha(genotype.background)
    assert edge.get_target_id() == _sha(genotype)
    assert edge.get_label() == "segregant background of"
    composite = _call(
        CellAdapter._perturbed_segregant_genotype_node,
        genotype,
        "perturbed segregant genotype (chunked)",
    )
    assert composite.get_id() == edge.get_target_id()
    assert composite.get_properties()["background_role"] == "parent_1"


def test_generic_methods_address_the_composite_unchanged() -> None:
    genotype = _genotype()
    to_experiment = _call(
        CellAdapter._genotype_to_experiment_edge,
        genotype,
        "genotype to experiment (chunked)",
    )
    assert to_experiment.get_source_id() == _sha(genotype)
    (member,) = _call(
        CellAdapter._perturbation_to_genotype_edges,
        genotype,
        "perturbation to genotype (chunked)",
    )
    assert member.get_source_id() == _sha(genotype.perturbations[0])
    assert member.get_target_id() == _sha(genotype)
    (construct,) = _call(
        CellAdapter._crispr_construct_node, genotype, "crispr construct (chunked)"
    )
    assert construct.get_properties()["guide_sequence"] == "TTTTCTCTTTTCGCTTAAAA"
    assert construct.get_properties()["effector"] == "dCas9-Mxi1"


_INIT_CASES = [
    AdapterCase(
        Hale2024Adapter,
        "crispri_segregant_hale2024_adapter.yaml",
        Shape("environment response phenotype", crispr=True, env_perturbation=True),
        CrispriSegregantHale2024Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``conf/crispri_segregant_hale2024_adapter.yaml``: 19 node and 17 edge methods.

    The perturbed segregant genotype and its background node, ``perturbation`` and
    ``crispr construct``, environment perturbations, the environment response phenotype;
    edges add ``segregant background to genotype`` beside the CRISPRi pair. Everything
    else is the harness's contract (see ``_adapter_init_harness``).
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
