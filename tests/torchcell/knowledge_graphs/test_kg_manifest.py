"""Tests for the served knowledge-graph manifest and the admission rule."""

from collections.abc import Sequence

import pytest

from torchcell.knowledge_graphs.kg_manifest import (
    AdapterDrift,
    AdmissionReport,
    GraphSchemaEntry,
    ServedDrift,
    batch_report_from_members,
    cell_adapter_surface,
    format_batch_report,
    graph_schema_from_yaml,
    parse_n_experiments,
    split_dataset_args,
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


# ------------------------------------------------------------------ batch admission


def _member(
    name: str,
    *,
    reasons: Sequence[str] = (),
    closure: dict[str, str] | None = None,
    novel: Sequence[str] = (),
    shared: dict[str, list[str]] | None = None,
    stale: Sequence[ServedDrift] = (),
) -> AdmissionReport:
    """One dataset's admission report, with only the batch-relevant fields varied."""
    return AdmissionReport(
        dataset_class=name,
        checked_at="2026-09-12T00:00:00+00:00",
        torchcell_commit="abc1234",
        torchcell_dirty=False,
        served_commit="513cbfa1",
        verdict="blocked" if reasons else "admissible",
        reasons=list(reasons),
        new_dataset_closure=closure if closure is not None else {},
        novel_symbols=list(novel),
        shared_symbols=shared if shared is not None else {},
        stale_served=list(stale),
        graph_schema_changed=[],
        graph_schema_added=[],
        adapter_drift=AdapterDrift(),
        adapter_methods_added=[],
        adapter_drift_acknowledged=None,
        dev_lmdb_status="fresh",
        dev_lmdb_root="/scratch/data/torchcell/toy",
        in_adapter_map=True,
        undeclared_phenotype_methods=[],
    )


def test_batch_is_admissible_only_when_every_member_is() -> None:
    ok = batch_report_from_members([_member("ADataset"), _member("BDataset")])
    assert ok.verdict == "admissible"
    assert ok.dataset_classes == ["ADataset", "BDataset"]
    assert ok.reasons == []

    mixed = batch_report_from_members(
        [_member("ADataset"), _member("BDataset", reasons=["dev-tree LMDB is stale"])]
    )
    assert mixed.verdict == "blocked"
    assert mixed.reasons == ["BDataset: dev-tree LMDB is stale"]


def test_batch_reports_symbols_two_members_both_introduce() -> None:
    # a symbol novel to the SERVED store that two members share is additive: it blocks
    # nothing, and the batch report names it and the members that bring it
    report = batch_report_from_members(
        [
            _member("ADataset", novel=["Solvent", "EnvironmentPerturbation"]),
            _member("BDataset", novel=["Solvent", "SegregantGenotype"]),
        ]
    )
    assert report.verdict == "admissible"
    assert report.co_introduced_symbols == {"Solvent": ["ADataset", "BDataset"]}
    assert "Solvent (ADataset, BDataset)" in format_batch_report(report)


def test_batch_names_the_served_datasets_importing_a_changed_symbol() -> None:
    drift = [
        ServedDrift(dataset_class="Costanzo2016Dataset", changed_symbols=["Media"]),
        ServedDrift(
            dataset_class="Kuzmin2018Dataset", changed_symbols=["Media", "Temperature"]
        ),
    ]
    report = batch_report_from_members(
        [
            _member("ADataset", reasons=["served datasets changed"], stale=drift),
            _member(
                "BDataset",
                reasons=["served datasets changed"],
                stale=drift,
                closure={"Media": "f1"},
            ),
        ]
    )
    assert report.verdict == "blocked"
    assert report.changed_symbol_importers == {
        "Media": ["Costanzo2016Dataset", "Kuzmin2018Dataset"],
        "Temperature": ["Kuzmin2018Dataset"],
    }
    text = format_batch_report(report)
    assert "Media: served Costanzo2016Dataset, Kuzmin2018Dataset" in text
    # the member whose own closure carries the changed symbol is named too
    assert "batch members importing it: BDataset" in text


def test_batch_refuses_an_empty_or_repeated_membership() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        batch_report_from_members([])
    with pytest.raises(ValueError, match="more than once"):
        batch_report_from_members([_member("ADataset"), _member("ADataset")])


def test_split_dataset_args_takes_repeats_and_comma_lists() -> None:
    assert split_dataset_args(["ADataset"]) == ["ADataset"]
    assert split_dataset_args(["ADataset", "BDataset"]) == ["ADataset", "BDataset"]
    assert split_dataset_args(["ADataset,BDataset", " CDataset "]) == [
        "ADataset",
        "BDataset",
        "CDataset",
    ]
    with pytest.raises(ValueError, match="more than once"):
        split_dataset_args(["ADataset,ADataset"])
    with pytest.raises(ValueError, match="empty dataset name"):
        split_dataset_args(["ADataset,"])


def test_parse_n_experiments_single_bare_count_and_batch_named_counts() -> None:
    # the single-dataset call is unchanged: a bare count
    assert parse_n_experiments(["6188"], ["ADataset"]) == {"ADataset": 6188}
    assert parse_n_experiments(
        ["ADataset=6188", "BDataset=42"], ["ADataset", "BDataset"]
    ) == {"ADataset": 6188, "BDataset": 42}
    with pytest.raises(ValueError, match="NAME=COUNT"):
        parse_n_experiments(["6188"], ["ADataset", "BDataset"])
    with pytest.raises(ValueError, match="missing for"):
        parse_n_experiments(["ADataset=6188"], ["ADataset", "BDataset"])
    with pytest.raises(ValueError, match="not in the report"):
        parse_n_experiments(["ADataset=1", "CDataset=1"], ["ADataset"])
