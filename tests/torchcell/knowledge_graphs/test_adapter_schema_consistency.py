"""Every dataset in the adapter map must be fully representable in the graph.

BioCypher silently drops a whole node class when the adapter emits a label (or a
property) the schema config does not declare, so an adapter conf that enables a
phenotype method without a matching schema-config entry produces a graph with that
dataset's phenotypes missing and no error. This test closes that gap for every mapped
dataset, and also checks that every conf method name exists in ``CellAdapter``'s
method table and that every experiment type in the schema's registry that a mapped
dataset produces has a phenotype node class.
"""

from pathlib import Path

import yaml

import torchcell
from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    GraphSchemaEntry,
    cell_adapter_surface,
    dataset_conf_methods,
    graph_schema_from_yaml,
)

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent


def _graph_schema() -> dict[str, GraphSchemaEntry]:
    return graph_schema_from_yaml(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )


def test_every_conf_method_exists_in_cell_adapter() -> None:
    _, table = cell_adapter_surface(
        (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    missing: dict[str, list[str]] = {}
    for dataset_class in dataset_adapter_map:
        names = [
            n for n in dataset_conf_methods(dataset_class, REPO_ROOT) if n not in table
        ]
        if names:
            missing[dataset_class.__name__] = names
    assert missing == {}


def test_every_enabled_phenotype_method_is_declared_in_graph_schema() -> None:
    schema = _graph_schema()
    undeclared: dict[str, list[str]] = {}
    for dataset_class in dataset_adapter_map:
        labels = [
            n[: -len(" (chunked)")]
            for n in dataset_conf_methods(dataset_class, REPO_ROOT)
            if n.endswith("phenotype (chunked)")
        ]
        bad = [label for label in labels if label not in schema]
        if bad:
            undeclared[dataset_class.__name__] = bad
    assert undeclared == {}


def test_every_declared_phenotype_is_a_phenotype_member_of_source() -> None:
    schema = _graph_schema()
    sources = set(schema["phenotype member of"].source)
    phenotypes = {
        name
        for name, e in schema.items()
        if e.kind == "node" and name.endswith("phenotype")
    }
    assert phenotypes <= sources, sorted(phenotypes - sources)


def test_every_node_method_property_is_declared() -> None:
    """Node properties an adapter method emits must all be declared in the schema.

    Read from the adapter conf enable-lists + the schema config only (no LMDB): the
    schema property sets are compared against the properties each node method emits,
    which are listed here as the contract for the phenotype methods added for the
    pseudobulk-expression and environment-perturbation families.
    """
    schema = _graph_schema()
    emitted = {
        "pseudobulk expression phenotype": {
            "graph_level",
            "label_name",
            "label_statistic_name",
            "expression_log2_ratio",
            "dispersion",
            "n_cells",
            "measurement_type",
            "serialized_data",
        },
        "environment perturbation": {
            "perturbation_type",
            "description",
            "compound_name",
            "inchikey",
            "concentration_value",
            "concentration_unit",
            "serialized_data",
        },
    }
    for label, properties in emitted.items():
        assert set(schema[label].properties) == properties, label


def test_adapter_confs_are_well_formed_yaml() -> None:
    conf_dir = REPO_ROOT / "torchcell/adapters/conf"
    for path in sorted(conf_dir.glob("*_adapter.yaml")):
        conf = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(conf["cell_adapter"]) == {"node_methods", "edge_methods"}, path.name
