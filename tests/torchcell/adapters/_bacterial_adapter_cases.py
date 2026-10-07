# tests/torchcell/adapters/_bacterial_adapter_cases.py
# [[tests.torchcell.adapters._bacterial_adapter_cases]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/_bacterial_adapter_cases.py
"""Shared cases and checks for the bacterial adapters (not a test module).

Every E. coli and P. putida dataset class (plan.bacteria-ontology-genome step 9) has
its own adapter module, its own conf and its own paired test file,
``test_<module>.py``; each of those runs the checks here for its one dataset, and
``test_bacterial_adapters.py`` holds the checks across the whole set. The checks:

* ``assert_construction`` / ``assert_missing_conf`` (``_adapter_init_harness``): the
  constructor loads exactly the conf its graph shape implies and refuses a missing conf
  by its exact path, with the perturbation nodes the ``bacterial perturbation`` class
  rather than the served yeast ``perturbation`` class;
* ``assert_conf_registered_and_declared``: every conf method name is registered on
  ``CellAdapter`` and every node class the conf can emit, the phenotype included, is
  declared in ``torchcell_schema_config.yaml`` (BioCypher drops an undeclared class
  silently), read through the admission gate's own ``dataset_conf_methods``;
* ``assert_gate_resolves_own_files``: ``kg_manifest`` resolves the dataset to ITS
  adapter module and ITS conf. A module holding two adapter classes resolves both to the
  first conf it names, which is why every bacterial dataset class has a module of its
  own;
* ``assert_dev_store_graph`` (data-gated, ``--data`` with a real ``DATA_ROOT``): the
  adapter runs every enabled method over the first records of its dev-tree LMDB the way
  the build runs them (one single-pass traversal for the chunked nodes, one for the
  chunked edges); the emitted graph is closed (every edge endpoint is an emitted node),
  every label and property is declared, and every sub-object family the conf leaves off
  is absent from the records. A store that is absent or mid-rebuild (no
  ``build_manifest.json``) is skipped, and so is one without a cached
  ``experiment_reference_index.json``, because computing that index would write it into
  the store.
"""

from __future__ import annotations

import importlib
import inspect
import os
import os.path as osp
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import pytest
import yaml

import torchcell
import torchcell.datasets.ecoli  # noqa: F401  # populates the registry
import torchcell.datasets.pputida  # noqa: F401  # populates the registry
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    install_recorder,
)
from torchcell.adapters import (
    CarbonSourceTong2020Adapter,
    CrispriArrayYunus2026Adapter,
    CrispriKnockdownYunus2026Adapter,
    EnvChemgenWang2015Adapter,
    GeneEssentialityGoodall2018Adapter,
    IsoprenolSelectionMenasalvas2025Adapter,
    IsoprenolTiterCarruthers2025Adapter,
    IsoprenolTiterDeSiqueira2025Adapter,
    IsoprenolToleranceLim2025Adapter,
    IsoprenylAcetateTiterKang2026Adapter,
    MetabolomeFuhrer2017Adapter,
    MetabolomeRapp2026Adapter,
    ProteinTurnoverGupta2024Adapter,
    ProteomeCaglar2017Adapter,
    ProteomeCarruthers2025Adapter,
    ProteomeDeSiqueira2025Adapter,
    ProteomeLim2025Adapter,
    PutidaPrecise321Lim2022Adapter,
    RbTnseqBorchert2024Adapter,
    RbTnseqPrice2018EcoliAdapter,
    RnaseqCaglar2017Adapter,
    RnaseqLamoureux2023Adapter,
)
from torchcell.adapters.cell_adapter import SINGLE_PASS_EDGES, SINGLE_PASS_NODES
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.datasets.ecoli.caglar2017 import (
    ProteomeCaglar2017Dataset,
    RnaseqCaglar2017Dataset,
)
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.datasets.ecoli.goodall2018 import GeneEssentialityGoodall2018Dataset
from torchcell.datasets.ecoli.gupta2024 import ProteinTurnoverGupta2024Dataset
from torchcell.datasets.ecoli.lamoureux2023 import RnaseqLamoureux2023Dataset
from torchcell.datasets.ecoli.price2018 import RbTnseqPrice2018EcoliDataset
from torchcell.datasets.ecoli.rapp2026 import MetabolomeRapp2026Dataset
from torchcell.datasets.ecoli.tong2020 import CarbonSourceTong2020Dataset
from torchcell.datasets.ecoli.wang2015 import EnvChemgenWang2015Dataset
from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset
from torchcell.datasets.pputida.carruthers2025 import (
    IsoprenolTiterCarruthers2025Dataset,
    ProteomeCarruthers2025Dataset,
)
from torchcell.datasets.pputida.desiqueira2025 import (
    IsoprenolTiterDeSiqueira2025Dataset,
    ProteomeDeSiqueira2025Dataset,
)
from torchcell.datasets.pputida.kang2026 import IsoprenylAcetateTiterKang2026Dataset
from torchcell.datasets.pputida.lim2022 import PutidaPrecise321Lim2022Dataset
from torchcell.datasets.pputida.lim2025 import (
    IsoprenolToleranceLim2025Dataset,
    ProteomeLim2025Dataset,
)
from torchcell.datasets.pputida.menasalvas2025 import (
    IsoprenolSelectionMenasalvas2025Dataset,
)
from torchcell.datasets.pputida.yunus2026 import (
    CrispriArrayYunus2026Dataset,
    CrispriKnockdownYunus2026Dataset,
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
KG_BACTERIA = REPO_ROOT / "torchcell/knowledge_graphs/conf/kg_bacteria.yaml"
BACTERIAL_PACKAGES = ("torchcell.datasets.ecoli.", "torchcell.datasets.pputida.")


class Bacterial(NamedTuple):
    """One adapter: its harness case and the module stem that holds it."""

    case: AdapterCase
    module: str


def _case(
    adapter_cls: type[Any],
    module: str,
    slug: str,
    dataset_cls: type[Any],
    phenotype: str,
    *,
    perturbation: bool = True,
    crispr: bool = False,
    env_perturbation: bool = True,
) -> Bacterial:
    shape = Shape(
        phenotype,
        perturbation=perturbation,
        crispr=crispr,
        env_perturbation=env_perturbation,
        bacterial=True,
    )
    return Bacterial(
        AdapterCase(adapter_cls, f"{slug}_adapter.yaml", shape, dataset_cls),
        f"{module}_adapter",
    )


RNASEQ = "rnaseq expression phenotype"
PROTEOME = "protein abundance phenotype"
TITER = "product titer phenotype"
RESPONSE = "environment response phenotype"
TURNOVER = "protein turnover phenotype"

# The shape of each dataset's records, measured on its dev-tree LMDB on 2026-10-07 (and,
# for the two RB-TnSeq stores rebuilding at the time, read off `build_genotype` /
# `build_environment` in the loader). Caglar 2017 is a wild-type panel with no
# perturbation in any record; Fuhrer 2017 and Goodall 2018 carry no environment
# perturbation; the CRISPRi leaves of Carruthers, Menasalvas and Yunus carry a
# CrisprConstruct.
BACTERIAL: list[Bacterial] = [
    _case(
        RnaseqCaglar2017Adapter,
        "caglar2017_rnaseq",
        "rnaseq_caglar2017",
        RnaseqCaglar2017Dataset,
        RNASEQ,
        perturbation=False,
    ),
    _case(
        ProteomeCaglar2017Adapter,
        "caglar2017_proteome",
        "proteome_caglar2017",
        ProteomeCaglar2017Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        MetabolomeFuhrer2017Adapter,
        "fuhrer2017",
        "metabolome_fuhrer2017",
        MetabolomeFuhrer2017Dataset,
        "metabolite phenotype",
        env_perturbation=False,
    ),
    _case(
        GeneEssentialityGoodall2018Adapter,
        "goodall2018",
        "gene_essentiality_goodall2018",
        GeneEssentialityGoodall2018Dataset,
        "gene essentiality phenotype",
        env_perturbation=False,
    ),
    _case(
        ProteinTurnoverGupta2024Adapter,
        "gupta2024",
        "protein_turnover_gupta2024",
        ProteinTurnoverGupta2024Dataset,
        TURNOVER,
    ),
    _case(
        RnaseqLamoureux2023Adapter,
        "lamoureux2023",
        "rnaseq_lamoureux2023",
        RnaseqLamoureux2023Dataset,
        RNASEQ,
    ),
    _case(
        RbTnseqPrice2018EcoliAdapter,
        "price2018_ecoli",
        "rbtnseq_price2018_ecoli",
        RbTnseqPrice2018EcoliDataset,
        RESPONSE,
    ),
    _case(
        MetabolomeRapp2026Adapter,
        "rapp2026",
        "metabolome_rapp2026",
        MetabolomeRapp2026Dataset,
        "metabolite phenotype",
        crispr=True,
    ),
    _case(
        CarbonSourceTong2020Adapter,
        "tong2020",
        "ecoli_carbon_source_tong2020",
        CarbonSourceTong2020Dataset,
        "fitness phenotype",
    ),
    _case(
        EnvChemgenWang2015Adapter,
        "wang2015",
        "env_chemgen_wang2015",
        EnvChemgenWang2015Dataset,
        RESPONSE,
    ),
    _case(
        RbTnseqBorchert2024Adapter,
        "borchert2024",
        "rbtnseq_borchert2024",
        RbTnseqBorchert2024Dataset,
        RESPONSE,
    ),
    _case(
        IsoprenolTiterCarruthers2025Adapter,
        "carruthers2025_titer",
        "isoprenol_titer_carruthers2025",
        IsoprenolTiterCarruthers2025Dataset,
        TITER,
        crispr=True,
    ),
    _case(
        ProteomeCarruthers2025Adapter,
        "carruthers2025_proteome",
        "proteome_carruthers2025",
        ProteomeCarruthers2025Dataset,
        PROTEOME,
        crispr=True,
    ),
    _case(
        ProteomeDeSiqueira2025Adapter,
        "desiqueira2025_proteome",
        "proteome_desiqueira2025",
        ProteomeDeSiqueira2025Dataset,
        PROTEOME,
    ),
    _case(
        IsoprenolTiterDeSiqueira2025Adapter,
        "desiqueira2025_titer",
        "isoprenol_titer_desiqueira2025",
        IsoprenolTiterDeSiqueira2025Dataset,
        TITER,
    ),
    _case(
        IsoprenylAcetateTiterKang2026Adapter,
        "kang2026",
        "isoprenyl_acetate_titer_kang2026",
        IsoprenylAcetateTiterKang2026Dataset,
        TITER,
    ),
    _case(
        PutidaPrecise321Lim2022Adapter,
        "lim2022",
        "putida_precise321_lim2022",
        PutidaPrecise321Lim2022Dataset,
        RNASEQ,
    ),
    _case(
        IsoprenolToleranceLim2025Adapter,
        "lim2025_tolerance",
        "isoprenol_tolerance_lim2025",
        IsoprenolToleranceLim2025Dataset,
        RESPONSE,
    ),
    _case(
        ProteomeLim2025Adapter,
        "lim2025_proteome",
        "proteome_lim2025",
        ProteomeLim2025Dataset,
        PROTEOME,
    ),
    _case(
        IsoprenolSelectionMenasalvas2025Adapter,
        "menasalvas2025",
        "isoprenol_selection_menasalvas2025",
        IsoprenolSelectionMenasalvas2025Dataset,
        RESPONSE,
        crispr=True,
    ),
    _case(
        CrispriArrayYunus2026Adapter,
        "yunus2026_array",
        "crispri_array_yunus2026",
        CrispriArrayYunus2026Dataset,
        PROTEOME,
        crispr=True,
    ),
    _case(
        CrispriKnockdownYunus2026Adapter,
        "yunus2026_knockdown",
        "crispri_knockdown_yunus2026",
        CrispriKnockdownYunus2026Dataset,
        PROTEOME,
        crispr=True,
    ),
]
IDS = [b.case.adapter_cls.__name__ for b in BACTERIAL]


def graph_schema() -> dict[str, GraphSchemaEntry]:
    return graph_schema_from_yaml(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )


def node_class(method_name: str) -> str:
    """The graph node class a conf node method writes (``... reference`` included)."""
    label = method_name.removesuffix(" (chunked)")
    if label == "experiment reference":
        return label
    return label.removesuffix(" reference")


def registered_bacterial_classes() -> set[type[Any]]:
    return {
        cls
        for cls in dataset_registry.values()
        if cls.__module__.startswith(BACTERIAL_PACKAGES)
    }


def case_for(dataset_cls: type[Any]) -> Bacterial:
    """The one case whose dataset class is ``dataset_cls``."""
    (match,) = [b for b in BACTERIAL if b.case.dataset_cls is dataset_cls]
    return match


def adapter_module(bacterial: Bacterial) -> ModuleType:
    """The adapter's own module, checked to be the one the case names."""
    module = importlib.import_module(f"torchcell.adapters.{bacterial.module}")
    assert bacterial.case.adapter_cls.__module__ == module.__name__
    return module


def assert_conf_registered_and_declared(bacterial: Bacterial) -> None:
    """Read through the gate's ``dataset_conf_methods``, as ``admit`` reads it."""
    _, table = cell_adapter_surface(
        (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    schema = graph_schema()
    names = dataset_conf_methods(bacterial.case.dataset_cls, REPO_ROOT)
    assert [n for n in names if n not in table] == []
    conf = yaml.safe_load(
        (REPO_ROOT / "torchcell/adapters/conf" / bacterial.case.conf_name).read_text(
            encoding="utf-8"
        )
    )
    node_names = [m["method_name"] for m in conf["cell_adapter"]["node_methods"]]
    classes = {node_class(n) for n in node_names}
    assert sorted(c for c in classes if c not in schema) == []
    assert all(schema[c].kind == "node" for c in classes)
    phenotype = bacterial.case.shape.phenotype
    assert {n for n in names if n.endswith("phenotype (chunked)")} == {
        f"{phenotype} (chunked)"
    }
    # a bacterial leaf is never written under the served yeast class, and no conf here
    # serves a segregant genotype or a phage
    assert "perturbation (chunked)" not in names
    assert "segregant genotype (chunked)" not in names
    assert "phage perturbation (chunked)" not in names


def assert_gate_resolves_own_files(bacterial: Bacterial) -> None:
    """``kg_manifest`` fingerprints this dataset's own module and conf."""
    assert dataset_adapter_files(bacterial.case.dataset_cls, REPO_ROOT) == [
        f"torchcell/adapters/{bacterial.module}.py",
        f"torchcell/adapters/conf/{bacterial.case.conf_name}",
    ]


# --------------------------------------------------------------------------- #
# Data-gated: each adapter over its dev-tree LMDB
# --------------------------------------------------------------------------- #
RECORDS = 200
# Chunked node methods a conf enables only when the records carry that sub-object.
OPTIONAL_FAMILIES = (
    "bacterial perturbation (chunked)",
    "crispr construct (chunked)",
    "environment perturbation (chunked)",
    "phage perturbation (chunked)",
)
STORE_FILES = (
    "processed/lmdb",
    "preprocess/build_manifest.json",
    "preprocess/experiment_reference_index.json",
)


def _dev_root(dataset_cls: type[Any]) -> str:
    params = inspect.signature(dataset_cls.__init__).parameters
    return osp.join(os.environ["DATA_ROOT"], params["root"].default)


def _run(adapter: Any, view: Any, kind: str) -> list[Any]:
    """Every enabled method of ``kind``: reference methods, then ONE chunked pass."""
    if kind == "node":
        table, conf, pass_name = (
            adapter.node_methods,
            adapter.config.cell_adapter.node_methods,
            SINGLE_PASS_NODES,
        )
    else:
        table, conf, pass_name = (
            adapter.edge_methods,
            adapter.config.cell_adapter.edge_methods,
            SINGLE_PASS_EDGES,
        )
    enabled = {m["method_name"] for m in conf}
    out: list[Any] = []
    chunked = []
    for name, method in table:
        if name not in enabled:
            continue
        if method.__name__.startswith("_get_"):
            out.extend(method())
        else:
            chunked.append((name, method))
    adapter._single_pass_methods = chunked
    out.extend(adapter._all_chunked(view, pass_name, inprocess=True))
    return out


def assert_dev_store_graph(
    bacterial: Bacterial, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter over its dev-tree LMDB emits a closed, declared, lossless graph."""
    root = _dev_root(bacterial.case.dataset_cls)
    missing = [rel for rel in STORE_FILES if not osp.exists(osp.join(root, rel))]
    if missing:
        pytest.skip(f"dev store at {root} is absent or mid-rebuild: no {missing}")
    install_recorder(monkeypatch)
    dataset = bacterial.case.dataset_cls(root=root)
    adapter = bacterial.case.adapter_cls(
        dataset=dataset, process_workers=1, io_workers=0
    )
    view = dataset[0 : min(len(dataset), RECORDS)]

    nodes = _run(adapter, view, "node")
    edges = _run(adapter, view, "edge")

    schema = graph_schema()
    labels = {node.get_label() for node in nodes}
    assert sorted(label for label in labels if label not in schema) == []
    for node in nodes:
        # BioCypherNode.get_properties() also reports the node's id and preferred_id
        declared = set(schema[node.get_label()].properties) | {"id", "preferred_id"}
        assert set(node.get_properties()) <= declared, node.get_label()
    node_ids = {node.get_id() for node in nodes}
    dangling = [
        (edge.get_label(), edge.get_source_id(), edge.get_target_id())
        for edge in edges
        if edge.get_source_id() not in node_ids or edge.get_target_id() not in node_ids
    ]
    assert dangling == []
    shape = bacterial.case.shape
    assert shape.phenotype in labels
    assert "perturbation" not in labels
    assert ("bacterial perturbation" in labels) is shape.perturbation
    assert ("crispr construct" in labels) is shape.crispr
    assert ("environment perturbation" in labels) is shape.env_perturbation

    # The converse: a sub-object family the conf leaves OFF is absent from the records,
    # so the enable-list drops nothing they carry.
    enabled = {m["method_name"] for m in adapter.config.cell_adapter.node_methods}
    left_off = [
        (name, method)
        for name, method in adapter.node_methods
        if name in OPTIONAL_FAMILIES and name not in enabled
    ]
    adapter._single_pass_methods = left_off
    assert adapter._all_chunked(view, SINGLE_PASS_NODES, inprocess=True) == []
