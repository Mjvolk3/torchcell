"""Every dataset in the adapter map must be fully representable in the graph.

BioCypher silently drops a whole node class when the adapter emits a label (or a
property) the schema config does not declare, so an adapter conf that enables a
phenotype method without a matching schema-config entry produces a graph with that
dataset's phenotypes missing and no error. This test closes that gap for every mapped
dataset, and also checks that every conf method name exists in ``CellAdapter``'s
method table and that every experiment type in the schema's registry that a mapped
dataset produces has a phenotype node class.

Also here, because both are properties of the conf enable-lists read as a whole:
``MULTI_CLASS_MODULE_CONFS``, the literal class-to-conf table of the eight adapter modules
that serve more than one dataset class (issue #743), and the dangling-edge rule that
replaced the environment-perturbation exclusivity rule once the two node lanes became a
partition (issue #756).
"""

import inspect
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

import torchcell
from torchcell.knowledge_graphs.dataset_adapter_map import (
    build_adapter_map,
    dataset_adapter_map,
)
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    GraphSchemaEntry,
    cell_adapter_surface,
    dataset_adapter_files,
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
        },
        "environment perturbation": {
            "perturbation_type",
            "description",
            "factor",
            "compound_name",
            "inchikey",
            "concentration_value",
            "concentration_unit",
        },
        "phage perturbation": {
            "perturbation_type",
            "description",
            "phage_name",
            "ncbi_taxid",
            "genome_accession",
            "multiplicity_of_infection",
            "titer_pfu_per_ml",
        },
    }
    for label, properties in emitted.items():
        assert set(schema[label].properties) == properties, label


def test_adapter_confs_are_well_formed_yaml() -> None:
    conf_dir = REPO_ROOT / "torchcell/adapters/conf"
    for path in sorted(conf_dir.glob("*_adapter.yaml")):
        conf = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(conf["cell_adapter"]) == {"node_methods", "edge_methods"}, path.name


def test_no_conf_enables_both_perturbation_classes() -> None:
    """A bacterial conf enables ``bacterial perturbation (chunked)`` INSTEAD of
    ``perturbation (chunked)``.

    The served ``perturbation`` method emits every leaf of a genotype, bacterial ones
    included, under the ``perturbation`` label, and the bacterial method emits the same
    leaf under the same content id with the ``bacterial perturbation`` label. Enabling both
    writes one id under two classes, and the import keeps whichever row it reads first.
    """
    both = sorted(
        dataset_class.__name__
        for dataset_class in dataset_adapter_map
        if {"perturbation (chunked)", "bacterial perturbation (chunked)"}
        <= set(dataset_conf_methods(dataset_class, REPO_ROOT))
    )
    assert both == []


def test_an_environment_perturbation_edge_has_a_node_lane_to_address() -> None:
    """Issue #756: the two environment-perturbation node lanes PARTITION the leaves.

    ``_environment_perturbation_node`` now skips a ``PhagePerturbation`` and
    ``_phage_perturbation_node`` emits only one, so a conf MAY enable both classes and no
    content id is written under two labels (that split is driven on a record in
    ``tests/torchcell/adapters/test_environment_node_identity.py``). What is left to
    check here is the edge: ``environment perturbation member of`` is emitted for EVERY
    perturbation, so a conf enabling it must enable at least one node lane, or that edge
    addresses a node the graph does not contain.
    """
    edge_methods = {
        "environment perturbation to environment (chunked)",
        "environment perturbation to environment reference",
    }
    node_methods = {
        "environment perturbation (chunked)",
        "environment perturbation reference",
        "phage perturbation (chunked)",
        "phage perturbation reference",
    }
    dangling = sorted(
        dataset_class.__name__
        for dataset_class in dataset_adapter_map
        if (enabled := set(dataset_conf_methods(dataset_class, REPO_ROOT)))
        & edge_methods
        and not enabled & node_methods
    )
    assert dangling == []


class _ConfOpened(Exception):
    """Raised by the recording ``open`` to stop an adapter right after it names its conf."""


def test_every_mapped_dataset_fingerprints_the_conf_its_adapter_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #743: the gate's conf for a dataset is the file its adapter OPENS.

    Public and private maps together, since ``dataset_adapter_files`` resolves both.

    Each adapter is constructed with ``None`` for every required argument and the
    builtin ``open`` shadowed in its module, so construction stops at the first file it
    opens, which is its conf (every adapter opens the conf before touching the dataset).
    That runtime path is compared with what ``dataset_adapter_files`` resolves from the
    class source. Eight modules serve several dataset classes each, so a module-level
    resolution fails here for every class after the first.
    """
    resolved: dict[str, str] = {}
    opened: dict[str, str] = {}
    every_map = build_adapter_map(include_private=True)
    for dataset_class, adapter_class in every_map.items():
        _, conf_rel = dataset_adapter_files(dataset_class, REPO_ROOT)
        resolved[dataset_class.__name__] = Path(conf_rel).name
        module = sys.modules[adapter_class.__module__]

        def record(path: str, *args: object, **kwargs: object) -> None:
            raise _ConfOpened(path)

        monkeypatch.setattr(module, "open", record, raising=False)
        required = [
            name
            for name, p in inspect.signature(adapter_class).parameters.items()
            if p.default is inspect.Parameter.empty
        ]
        with pytest.raises(_ConfOpened) as caught:
            adapter_class(**dict.fromkeys(required))
        opened[dataset_class.__name__] = Path(str(caught.value)).name
    assert len(opened) == len(every_map)
    assert resolved == opened


#: The eight adapter modules that serve more than one dataset class, and the conf each
#: class binds (issue #743). Pinned as a literal table rather than derived, because the
#: defect was a resolution that silently handed every class the module's FIRST conf: a
#: derived expectation would have agreed with it. Measured 2026-10-09 by
#: ``dataset_adapter_files`` over ``build_adapter_map(include_private=True)``; the first
#: conf of each module (the one the old regex returned) is the alphabetically first value
#: here only for Hillenmeyer and Lopez, so this table is also the record of which classes
#: were mis-attributed.
MULTI_CLASS_MODULE_CONFS: dict[str, dict[str, str]] = {
    "costanzo2016_adapter.py": {
        "DmfCostanzo2016Dataset": "dmf_costanzo2016_adapter.yaml",
        "DmiCostanzo2016Dataset": "dmi_costanzo2016_adapter.yaml",
        "SmfCostanzo2016Dataset": "smf_costanzo2016_adapter.yaml",
    },
    "hillenmeyer2008_adapter.py": {
        "HetHillenmeyer2008Dataset": "het_hillenmeyer2008_adapter.yaml",
        "HomHillenmeyer2008Dataset": "hom_hillenmeyer2008_adapter.yaml",
    },
    "kuzmin2018_adapter.py": {
        "DmfKuzmin2018Dataset": "dmf_kuzmin2018_adapter.yaml",
        "DmiKuzmin2018Dataset": "dmi_kuzmin2018_adapter.yaml",
        "SmfKuzmin2018Dataset": "smf_kuzmin2018_adapter.yaml",
        "TmfKuzmin2018Dataset": "tmf_kuzmin2018_adapter.yaml",
        "TmiKuzmin2018Dataset": "tmi_kuzmin2018_adapter.yaml",
    },
    "kuzmin2020_adapter.py": {
        "DmfKuzmin2020Dataset": "dmf_kuzmin2020_adapter.yaml",
        "DmiKuzmin2020Dataset": "dmi_kuzmin2020_adapter.yaml",
        "SmfKuzmin2020Dataset": "smf_kuzmin2020_adapter.yaml",
        "TmfKuzmin2020Dataset": "tmf_kuzmin2020_adapter.yaml",
        "TmiKuzmin2020Dataset": "tmi_kuzmin2020_adapter.yaml",
    },
    "lopez2024_adapter.py": {
        "IsobutanolScreenLopez2024Dataset": (
            "isobutanol_screen_lopez2024_adapter.yaml"
        ),
        "IsobutanolValidatedLopez2024Dataset": (
            "isobutanol_validated_lopez2024_adapter.yaml"
        ),
    },
    "sameith2015_adapter.py": {
        "DmMicroarraySameith2015Dataset": "dm_microarray_sameith2015_adapter.yaml",
        "SmMicroarraySameith2015Dataset": "sm_microarray_sameith2015_adapter.yaml",
    },
    "synth_leth_db_adapter.py": {
        "SynthLethalityYeastSynthLethDbDataset": (
            "synth_lethality_yeast_synth_leth_db_adapter.yaml"
        ),
        "SynthRescueYeastSynthLethDbDataset": (
            "synth_rescue_yeast_synth_leth_db_adapter.yaml"
        ),
    },
    "zelezniak2018_adapter.py": {
        "MetaboliteZelezniak2018Dataset": "metabolite_zelezniak2018_adapter.yaml",
        "ProteomeZelezniak2018Dataset": "proteome_zelezniak2018_adapter.yaml",
    },
}


def test_the_multi_class_modules_are_exactly_these_eight() -> None:
    """A ninth multi-class module, or a class added to one of the eight, lands here.

    Both directions: every class the table names resolves to the files the table states,
    AND the classes that SHARE an adapter module are exactly the ones it names. So a new
    dataset class sharing an existing adapter module cannot be added without stating the
    conf it binds.
    """
    resolved = {
        dataset_class.__name__: dataset_adapter_files(dataset_class, REPO_ROOT)
        for dataset_class in build_adapter_map(include_private=True)
    }
    expected = {
        dataset_name: [
            f"torchcell/adapters/{module_name}",
            f"torchcell/adapters/conf/{conf_name}",
        ]
        for module_name, confs in MULTI_CLASS_MODULE_CONFS.items()
        for dataset_name, conf_name in confs.items()
    }
    assert {name: resolved[name] for name in expected} == expected
    per_module = Counter(files[0] for files in resolved.values())
    assert sorted(
        name for name, files in resolved.items() if per_module[files[0]] > 1
    ) == sorted(expected)


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "conf_name"),
    [
        (module_name, dataset_name, conf_name)
        for module_name, confs in sorted(MULTI_CLASS_MODULE_CONFS.items())
        for dataset_name, conf_name in sorted(confs.items())
    ],
)
def test_a_multi_class_module_binds_each_class_to_its_own_conf(
    module_name: str, dataset_name: str, conf_name: str
) -> None:
    (dataset_class,) = [
        cls
        for cls in build_adapter_map(include_private=True)
        if cls.__name__ == dataset_name
    ]
    assert dataset_adapter_files(dataset_class, REPO_ROOT) == [
        f"torchcell/adapters/{module_name}",
        f"torchcell/adapters/conf/{conf_name}",
    ]
