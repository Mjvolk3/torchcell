# tests/torchcell/knowledge_graphs/supported_queries/test_cypher_deps.py
# [[tests.torchcell.knowledge_graphs.supported_queries.test_cypher_deps]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/supported_queries/test_cypher_deps.py
"""The Cypher dependency extractor on the real 025 solid-growth query and on small
queries that exercise each rule and each documented limit.

The expected sets for ``experiments/025-solid-growth/queries/001_all_solid_growth.cql``
were read off the file: 15 ``UNION ALL`` blocks, each matching ``Dataset``,
``Experiment``, ``Genotype``, ``ExperimentReference``, ``PhenotypicFeature``,
``Environment`` and ``Media`` through seven relationship types; ``PerturbationMemberOf``
appears only inside the pattern comprehensions over the unlabeled ``pert``. The graph
levels are 8 x ``global``, 4 x ``edge``, 2 x ``hyperedge`` and 1 x ``node``; the medium is
filtered by ``m.state``, never ``m.name``; ``p.systematic_gene_name`` reads through the
iteration variable ``p`` (bound by ``p IN``, no label).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from torchcell.knowledge_graphs.supported_queries.cypher_deps import (
    blank_literals,
    extract_dependencies,
    split_blocks,
    strip_comments,
)
from torchcell.knowledge_graphs.supported_queries.registry import QueryDependencies

REPO = Path(__file__).resolve().parents[4]
QUERY_025 = REPO / "experiments/025-solid-growth/queries/001_all_solid_growth.cql"

DATASETS_025 = [
    "DmfCostanzo2016Dataset",
    "DmfKuzmin2018Dataset",
    "DmfKuzmin2020Dataset",
    "DmiCostanzo2016Dataset",
    "DmiKuzmin2018Dataset",
    "DmiKuzmin2020Dataset",
    "GeneEssentialitySgdDataset",
    "SmfCostanzo2016Dataset",
    "SmfKuzmin2018Dataset",
    "SmfKuzmin2020Dataset",
    "SynthLethalityYeastSynthLethDbDataset",
    "TmfKuzmin2018Dataset",
    "TmfKuzmin2020Dataset",
    "TmiKuzmin2018Dataset",
    "TmiKuzmin2020Dataset",
]


def test_real_025_query_dependencies_exact() -> None:
    deps = extract_dependencies(QUERY_025.read_text(encoding="utf-8"))
    assert deps == QueryDependencies(
        node_labels=[
            "Dataset",
            "Environment",
            "Experiment",
            "ExperimentReference",
            "Genotype",
            "Media",
            "PhenotypicFeature",
        ],
        relationship_types=[
            "EnvironmentMemberOf",
            "ExperimentMemberOf",
            "ExperimentReferenceOf",
            "GenotypeMemberOf",
            "MediaMemberOf",
            "PerturbationMemberOf",
            "PhenotypeMemberOf",
        ],
        properties=[
            "graph_level",
            "id",
            "serialized_data",
            "state",
            "systematic_gene_name",
        ],
        label_properties=[
            "Dataset.id",
            "Experiment.id",
            "Experiment.serialized_data",
            "ExperimentReference.serialized_data",
            "Media.state",
            "PhenotypicFeature.graph_level",
        ],
        dataset_ids=DATASETS_025,
        graph_levels=["edge", "global", "hyperedge", "node"],
        media_names=[],
        parameters=["gene_set"],
    )


def test_real_025_query_has_fifteen_blocks() -> None:
    text = blank_literals(QUERY_025.read_text(encoding="utf-8"))
    assert len(split_blocks(text)) == 15


def test_shipped_copy_of_025_reads_the_same_dependencies() -> None:
    shipped = REPO / "torchcell/knowledge_graphs/queries/solid_growth_025.cql"
    assert extract_dependencies(
        shipped.read_text(encoding="utf-8")
    ) == extract_dependencies(QUERY_025.read_text(encoding="utf-8"))


def test_comments_are_ignored_but_slashes_inside_literals_are_not_comments() -> None:
    cypher = (
        "// MATCH (ghost:Ghost) WHERE ghost.id = 'Nope'\n"
        "MATCH (p:Publication) /* (x:Hidden)\n  [:HiddenOf] */\n"
        "WHERE p.doi_url = 'https://doi.org/x' RETURN p.serialized_data\n"
    )
    deps = extract_dependencies(cypher)
    assert deps.node_labels == ["Publication"]
    assert deps.relationship_types == []
    assert deps.label_properties == [
        "Publication.doi_url",
        "Publication.serialized_data",
    ]


def test_patterns_inside_string_literals_are_not_read() -> None:
    cypher = "MATCH (m:Media) WHERE m.name = '(x:Fake)-[:FakeOf]-($p)' RETURN m.state"
    deps = extract_dependencies(cypher)
    assert deps.node_labels == ["Media"]
    assert deps.relationship_types == []
    assert deps.parameters == []
    assert deps.media_names == ["(x:Fake)-[:FakeOf]-($p)"]


def test_in_lists_double_quotes_multi_labels_and_type_alternation() -> None:
    cypher = (
        "MATCH (d:Dataset)<-[r:ExperimentMemberOf|:ExperimentReferenceMemberOf]-(e:A:B)\n"
        "WHERE d.id IN ['DsA', \"DsB\"] AND e.graph_level IN ['node', 'edge']\n"
        "MATCH (e)<-[:MediaMemberOf]-(m:Media) WHERE m.name IN ['YPD', 'SC']\n"
        "RETURN e.x, r.weight\n"
    )
    deps = extract_dependencies(cypher)
    assert deps.node_labels == ["A", "B", "Dataset", "Media"]
    assert deps.relationship_types == [
        "ExperimentMemberOf",
        "ExperimentReferenceMemberOf",
        "MediaMemberOf",
    ]
    assert deps.dataset_ids == ["DsA", "DsB"]
    assert deps.graph_levels == ["edge", "node"]
    assert deps.media_names == ["SC", "YPD"]
    # r.weight reads through a relationship variable: a property, never a label property
    assert deps.properties == ["graph_level", "id", "name", "weight", "x"]
    assert deps.label_properties == [
        "A.graph_level",
        "A.x",
        "B.graph_level",
        "B.x",
        "Dataset.id",
        "Media.name",
    ]


def test_unbound_variables_and_function_calls_are_not_property_reads() -> None:
    cypher = (
        "MATCH (g:Genotype) WITH apoc.coll.toSet(g.systematic_gene_names) AS genes\n"
        "RETURN genes, stranger.prop, size(genes)\n"
    )
    deps = extract_dependencies(cypher)
    assert deps.properties == ["systematic_gene_names"]
    assert deps.label_properties == ["Genotype.systematic_gene_names"]


def test_variables_and_labels_are_scoped_to_their_union_block() -> None:
    cypher = (
        "MATCH (x:Dataset) WHERE x.id = 'DsA' RETURN x.id AS id\n"
        "UNION ALL\n"
        "MATCH (x:Experiment) WHERE x.id = 'NotADataset' RETURN x.id AS id\n"
        "union\n"
        "MATCH (m:Environment) WHERE m.name = 'NotAMedium' RETURN m.id AS id\n"
    )
    deps = extract_dependencies(cypher)
    assert deps.dataset_ids == ["DsA"]
    assert deps.media_names == []
    assert deps.label_properties == [
        "Dataset.id",
        "Environment.id",
        "Environment.name",
        "Experiment.id",
    ]


def test_documented_limits_where_label_predicate_and_left_literal() -> None:
    cypher = "MATCH (d) WHERE d:Dataset AND 'DsA' = d.id RETURN d.id"
    deps = extract_dependencies(cypher)
    assert deps.node_labels == []
    assert deps.dataset_ids == []
    assert deps.properties == ["id"]
    assert deps.label_properties == []


def test_strip_and_blank_keep_offsets() -> None:
    cypher = "MATCH (a:A) // note\nWHERE a.s = 'x\\'y' /* c */ RETURN a"
    stripped = strip_comments(cypher)
    blanked = blank_literals(cypher)
    assert stripped == "MATCH (a:A)        \nWHERE a.s = 'x\\'y'         RETURN a"
    assert blanked == "MATCH (a:A)        \nWHERE a.s = '    '         RETURN a"
    assert len(stripped) == len(blanked) == len(cypher)


def test_unterminated_literal_runs_to_the_end() -> None:
    cypher = "MATCH (a:A) WHERE a.s = 'open"
    assert blank_literals(cypher) == "MATCH (a:A) WHERE a.s = '    "
    assert strip_comments(cypher) == cypher


def test_split_blocks_offsets() -> None:
    assert split_blocks("A UNION B union all C") == [(0, 2), (7, 10), (19, 21)]


def test_dependencies_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValidationError, match="node_labels must be sorted and unique"):
        QueryDependencies(
            node_labels=["B", "A"],
            relationship_types=[],
            properties=[],
            label_properties=[],
            dataset_ids=[],
            graph_levels=[],
            media_names=[],
            parameters=[],
        )
