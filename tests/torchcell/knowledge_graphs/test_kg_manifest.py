"""Tests for the served knowledge-graph manifest and the admission rule."""

from torchcell.knowledge_graphs.kg_manifest import (
    GraphSchemaEntry,
    cell_adapter_surface,
    graph_schema_from_yaml,
)

SCHEMA_YAML = """
dataset:
    represented_as: node

experiment:
    represented_as: node
    is_a: information content entity
    properties:
        serialized_data: str

fitness phenotype:
    is_a: phenotypic feature
    represented_as: node
    properties:
        fitness: float
        serialized_data: str

phenotype member of:
    is_a: participates in
    represented_as: edge
    input_label: phenotype
    source: [fitness phenotype]
    target: [experiment, experiment reference]

media member of:
    is_a: part of
    represented_as: edge
    source: media
    target: environment
"""

ADAPTER_SRC = '''
class CellAdapter:
    def __init__(self, config):
        """doc"""
        self.config = config
        self.node_methods = [
            ("experiment (chunked)", self._experiment_node),
            ("fitness phenotype (chunked)", self._fitness_phenotype_node),
        ]
        self.edge_methods = [
            ("genotype to experiment (chunked)", self._genotype_to_experiment_edge),
        ]

    def get_nodes(self):
        """plumbing"""
        return 1

    def _experiment_node(self, data, method_name):
        """Experiment node."""
        return data

    def _fitness_phenotype_node(self, data, method_name):
        return data["fitness"]

    def _genotype_to_experiment_edge(self, data, method_name):
        return (data, data)
'''


def test_graph_schema_from_yaml_splits_nodes_and_edges() -> None:
    schema = graph_schema_from_yaml(SCHEMA_YAML)
    assert schema["dataset"] == GraphSchemaEntry(kind="node", properties=[])
    assert schema["fitness phenotype"].properties == ["fitness", "serialized_data"]
    edge = schema["phenotype member of"]
    assert edge.kind == "edge"
    assert edge.source == ["fitness phenotype"]
    assert edge.target == ["experiment", "experiment reference"]
    assert schema["media member of"].source == ["media"]


def test_served_nodes_unchanged_rules() -> None:
    node = GraphSchemaEntry(kind="node", properties=["a", "serialized_data"])
    assert node.served_nodes_unchanged_by(node)
    assert not node.served_nodes_unchanged_by(None)
    assert not node.served_nodes_unchanged_by(
        GraphSchemaEntry(kind="node", properties=["a", "b", "serialized_data"])
    )
    edge = GraphSchemaEntry(kind="edge", source=["x"], target=["experiment"])
    # gaining a source label is additive (a new phenotype family joining the edge)
    assert edge.served_nodes_unchanged_by(
        GraphSchemaEntry(kind="edge", source=["x", "y"], target=["experiment"])
    )
    # losing one, or changing kind, is not
    assert not edge.served_nodes_unchanged_by(
        GraphSchemaEntry(kind="edge", source=[], target=["experiment"])
    )
    assert not edge.served_nodes_unchanged_by(GraphSchemaEntry(kind="node"))


def test_cell_adapter_surface_reads_table_and_fingerprints_methods() -> None:
    methods, table = cell_adapter_surface(ADAPTER_SRC)
    assert table == {
        "experiment (chunked)": "_experiment_node",
        "fitness phenotype (chunked)": "_fitness_phenotype_node",
        "genotype to experiment (chunked)": "_genotype_to_experiment_edge",
    }
    assert set(methods) == {
        "__init__",
        "get_nodes",
        "_experiment_node",
        "_fitness_phenotype_node",
        "_genotype_to_experiment_edge",
    }


def test_cell_adapter_fingerprints_ignore_docstrings_and_new_table_entries() -> None:
    methods, _ = cell_adapter_surface(ADAPTER_SRC)
    # docstring edit: no drift
    edited = ADAPTER_SRC.replace(
        '"""Experiment node."""', '"""Experiment node (edited)."""'
    )
    assert (
        cell_adapter_surface(edited)[0]["_experiment_node"]
        == methods["_experiment_node"]
    )
    # registering an ADDITIONAL method in the table: __init__ fingerprint unchanged,
    # new method present, table extended
    extended = (
        ADAPTER_SRC.replace(
            '            ("fitness phenotype (chunked)", self._fitness_phenotype_node),\n',
            '            ("fitness phenotype (chunked)", self._fitness_phenotype_node),\n'
            '            ("new phenotype (chunked)", self._new_phenotype_node),\n',
        )
        + "\n    def _new_phenotype_node(self, data, method_name):\n        return 0\n"
    )
    methods2, table2 = cell_adapter_surface(extended)
    assert methods2["__init__"] == methods["__init__"]
    assert "_new_phenotype_node" in methods2
    assert table2["new phenotype (chunked)"] == "_new_phenotype_node"
    # a body change IS drift
    changed = ADAPTER_SRC.replace(
        'return data["fitness"]', 'return data["fitness"] * 2'
    )
    assert (
        cell_adapter_surface(changed)[0]["_fitness_phenotype_node"]
        != methods["_fitness_phenotype_node"]
    )
    # a plumbing change IS drift
    plumbing = ADAPTER_SRC.replace("return 1", "return 2")
    assert cell_adapter_surface(plumbing)[0]["get_nodes"] != methods["get_nodes"]
