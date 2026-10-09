# tests/torchcell/knowledge_graphs/test_graph_schema_ontology.py
# [[tests.torchcell.knowledge_graphs.test_graph_schema_ontology]]
"""The committed graph schema builds into BioCypher's ontology on the pinned Biolink head.

BioCypher resolves every schema class's ``is_a`` against its head ontology before the
first node is written, and a class it cannot place fails the build (the ``crispr
construct`` comment in ``torchcell_schema_config.yaml`` records rehearsal job 1956).
That failure used to surface only in a slurm generation job. Here the ontology is built
in-process from the sha256-pinned local mirror (``biocypher/ontology/``, issue #619) and
the real ``OntologyMapping`` of the committed schema, which takes under a second.

Pinned: every node class reaches the Biolink root ``entity``; the four bacterial-program
classes sit where their ``is_a`` puts them; and the served ``perturbation`` class keeps
its ancestry, with ``bacterial perturbation`` beside it under ``genotype`` rather than
beneath it (a child would carry the ``Perturbation`` label, and every served
``MATCH (:Perturbation)`` would start returning bacterial nodes).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from biocypher._mapping import OntologyMapping
from biocypher._ontology import Ontology

import torchcell
from torchcell.knowledge_graphs.head_ontology import REPO_ONTOLOGY_PATH
from torchcell.knowledge_graphs.kg_manifest import (
    SCHEMA_CONFIG_RELPATH,
    graph_schema_from_yaml,
)

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built() -> tuple[OntologyMapping, Ontology]:
    mapping = OntologyMapping(config_file=str(REPO_ROOT / SCHEMA_CONFIG_RELPATH))
    ontology = Ontology(
        head_ontology={
            "url": str(REPO_ROOT / REPO_ONTOLOGY_PATH),
            "root_node": "entity",
        },
        ontology_mapping=mapping,
    )
    return mapping, ontology


def _ancestry(ontology: Ontology, label: str) -> list[str]:
    return list(ontology.get_ancestors(label))


def test_every_node_class_reaches_the_biolink_root(
    built: tuple[OntologyMapping, Ontology],
) -> None:
    mapping, ontology = built
    graph = ontology._nx_graph
    nodes = [
        key
        for key, value in mapping.extended_schema.items()
        if value.get("represented_as") == "node"
    ]
    declared = graph_schema_from_yaml(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )
    assert set(nodes) == {k for k, v in declared.items() if v.kind == "node"}
    assert [key for key in nodes if not nx.has_path(graph, key, "entity")] == []


@pytest.mark.parametrize(
    "label,ancestry",
    [
        (
            "bacterial perturbation",
            [
                "bacterial perturbation",
                "genotype",
                "biological entity",
                "named thing",
                "entity",
            ],
        ),
        (
            "product titer phenotype",
            [
                "product titer phenotype",
                "phenotypic feature",
                "disease or phenotypic feature",
                "biological entity",
                "named thing",
                "entity",
            ],
        ),
        (
            "protein turnover phenotype",
            [
                "protein turnover phenotype",
                "phenotypic feature",
                "disease or phenotypic feature",
                "biological entity",
                "named thing",
                "entity",
            ],
        ),
        (
            "flux phenotype",
            [
                "flux phenotype",
                "phenotypic feature",
                "disease or phenotypic feature",
                "biological entity",
                "named thing",
                "entity",
            ],
        ),
    ],
)
def test_the_bacterial_program_classes_resolve_through_biolink(
    built: tuple[OntologyMapping, Ontology], label: str, ancestry: list[str]
) -> None:
    _, ontology = built
    assert _ancestry(ontology, label) == ancestry
    # the parent chain is Biolink's own, not a node BioCypher invented for a typo
    assert not any(
        ontology._nx_graph.nodes[parent].get("user_extension")
        for parent in ancestry[1:]
    )


def test_served_perturbation_ancestry_is_unchanged_and_has_no_bacterial_child(
    built: tuple[OntologyMapping, Ontology],
) -> None:
    _, ontology = built
    graph = ontology._nx_graph
    assert _ancestry(ontology, "perturbation") == [
        "perturbation",
        "genotype",
        "biological entity",
        "named thing",
        "entity",
    ]
    assert not nx.has_path(graph, "bacterial perturbation", "perturbation")
    assert not nx.has_path(graph, "perturbation", "bacterial perturbation")


def test_widened_served_edges_keep_their_own_ancestry(
    built: tuple[OntologyMapping, Ontology],
) -> None:
    """A served edge gaining a source gains virtual leaves BELOW it, never a new parent.

    The written relationship type comes from the edge class's own ancestry, so it stays
    ``PerturbationMemberOf`` / ``PhenotypeMemberOf`` / ``CrisprConstructMemberOf``.
    """
    mapping, ontology = built
    assert _ancestry(ontology, "perturbation member of") == [
        "perturbation member of",
        "genetically associated with",
    ]
    assert _ancestry(ontology, "phenotype member of") == [
        "phenotype member of",
        "participates in",
    ]
    assert _ancestry(ontology, "crispr construct member of") == [
        "crispr construct member of",
        "part of",
    ]
    assert {
        key
        for key in mapping.extended_schema
        if key.endswith(".perturbation member of")
    } == {
        "perturbation.perturbation member of",
        "bacterial perturbation.perturbation member of",
        "bacterial sequence variant perturbation.perturbation member of",
    }
