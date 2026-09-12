# tests/torchcell/adapters/test_crispr_construct_nodes.py
"""Unit tests for the ``crispr construct`` node and its edge to a perturbation.

A CRISPR screen's effector, spacer and library sub-pool are the reagent identity of the
strain, and before this class they existed in the graph only inside a perturbation's
``serialized_data``. The construct is its own node class rather than extra properties on
``perturbation``, because ``perturbation`` is a served graph class whose property set
cannot change without a full rebuild.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

import torchcell
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import (
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Genotype,
    KanMxDeletionPerturbation,
)
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    cell_adapter_surface,
)

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent


def _construct() -> CrisprConstruct:
    return CrisprConstruct(
        effector="dCas9-Mxi1",
        guide_sequence="GTCAGGTACTCCGAATTCGA",
        n_guides=4,
        library_pool="gene_tiling_20bp",
    )


def _perturbation() -> CrisprInterferencePerturbation:
    return CrisprInterferencePerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", crispr=_construct()
    )


def _record(perturbations: list[Any]) -> dict[str, Any]:
    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.genotype = Genotype(perturbations=perturbations)  # type: ignore[attr-defined]
    return {"experiment": experiment}


def _undecorated(method: Any) -> Any:
    """The chunk handler behind ``@data_chunker`` (the decorator needs an LMDB loader)."""
    return cast(Any, method).__wrapped__


def test_crispr_construct_node_is_content_addressed_and_projects_the_reagent() -> None:
    construct = _construct()
    node = CellAdapter._crispr_construct_node_from(construct)
    expected = hashlib.sha256(
        json.dumps(construct.model_dump()).encode("utf-8")
    ).hexdigest()
    assert node.get_id() == expected
    assert node.get_label() == "crispr construct"
    props = node.get_properties()
    assert props["effector"] == "dCas9-Mxi1"
    assert props["guide_sequence"] == "GTCAGGTACTCCGAATTCGA"
    assert props["n_guides"] == 4
    assert props["library_pool"] == "gene_tiling_20bp"
    assert props["effector_plasmid_uri"] is None
    assert props["effector_plasmid_sha256"] is None
    assert json.loads(props["serialized_data"]) == construct.model_dump()
    # Two equal constructs are one node, which is what lets the reagent join records.
    assert CellAdapter._crispr_construct_node_from(_construct()).get_id() == expected


def test_crispr_construct_node_method_emits_one_node_per_construct() -> None:
    adapter = CellAdapter.__new__(CellAdapter)
    nodes = _undecorated(CellAdapter._crispr_construct_node)(
        adapter, _record([_perturbation()]), "crispr construct (chunked)"
    )
    assert [n.get_label() for n in nodes] == ["crispr construct"]
    assert (
        nodes[0].get_id()
        == CellAdapter._crispr_construct_node_from(_construct()).get_id()
    )


def test_a_non_crispr_perturbation_emits_no_construct_node_or_edge() -> None:
    data = _record(
        [
            KanMxDeletionPerturbation(
                systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
            )
        ]
    )
    adapter = CellAdapter.__new__(CellAdapter)
    assert (
        _undecorated(CellAdapter._crispr_construct_node)(
            adapter, data, "crispr construct (chunked)"
        )
        == []
    )
    assert (
        _undecorated(CellAdapter._crispr_construct_to_perturbation_edges)(
            adapter, data, "crispr construct to perturbation (chunked)"
        )
        == []
    )


def test_crispr_construct_edge_points_from_construct_to_its_perturbation() -> None:
    adapter = CellAdapter.__new__(CellAdapter)
    edges = _undecorated(CellAdapter._crispr_construct_to_perturbation_edges)(
        adapter,
        _record([_perturbation()]),
        "crispr construct to perturbation (chunked)",
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.get_label() == "crispr construct member of"
    assert (
        edge.get_source_id()
        == CellAdapter._crispr_construct_node_from(_construct()).get_id()
    )
    # The target is the perturbation node id _perturbation_node computes, unchanged.
    assert (
        edge.get_target_id()
        == hashlib.sha256(
            json.dumps(_perturbation().model_dump()).encode("utf-8")
        ).hexdigest()
    )


def test_the_new_methods_are_registered_in_the_adapter_method_table() -> None:
    """A method absent from the table can never be enabled by an adapter conf."""
    _, table = cell_adapter_surface(
        (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    assert table["crispr construct (chunked)"] == "_crispr_construct_node"
    assert (
        table["crispr construct to perturbation (chunked)"]
        == "_crispr_construct_to_perturbation_edges"
    )


def test_graph_schema_declares_the_construct_class_and_its_edge() -> None:
    schema = yaml.safe_load(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )
    node = schema["crispr construct"]
    assert node["represented_as"] == "node"
    assert set(node["properties"]) == {
        "effector",
        "guide_sequence",
        "n_guides",
        "library_pool",
        "effector_plasmid_uri",
        "effector_plasmid_sha256",
        "serialized_data",
    }
    edge = schema["crispr construct member of"]
    assert edge["represented_as"] == "edge"
    assert edge["source"] == "crispr construct"
    assert edge["target"] == "perturbation"
