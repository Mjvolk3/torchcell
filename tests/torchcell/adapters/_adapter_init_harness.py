# tests/torchcell/adapters/_adapter_init_harness.py
# [[tests.torchcell.adapters._adapter_init_harness]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/_adapter_init_harness.py
"""Shared checks for the single-dataset adapter constructors (not a test module).

Every ``torchcell/adapters/<name>_adapter.py`` class does the same three things: find
``conf/<file>.yaml`` next to the module (refusing when it is absent), load it into an
``OmegaConf`` config, and call ``CellAdapter.__init__`` with it. The conf is the
knowledge-graph contract: which node and edge methods run for that dataset. So the
checks here are:

* the dataset each adapter is checked against is the one the build pairs it with,
  ``torchcell.knowledge_graphs.dataset_adapter_map`` inverted, and each case also
  states the class it expects that pairing to be;
* the EXACT conf content, rebuilt by ``expected_conf`` from a short description of
  the dataset's graph shape (phenotype kind, whether per-perturbation nodes, CRISPR
  constructs or environment perturbations are served, whether the perturbation nodes
  are the bacterial class, and the memory-reduction factor written on chunked
  methods) plus whether the genotype is a segregant, which is
  DERIVED from the paired dataset's ``experiment_class`` genotype type hint
  (``SegregantGenotype``) rather than written per case. ``phage`` is the second
  environment-side node class, independent of ``env_perturbation`` since issue #756: it
  adds the ``phage perturbation`` node pair and rides the same two
  ``environment perturbation to environment`` edges, whose graph class declares both node
  classes as sources. The method order is the
  ``CellAdapter`` registration-table order restricted to the enabled set, written out
  by hand below;
* the conf order is the order the adapter will run the methods in:
  ``_yield_methods`` iterates the registration tables (``supported_node_methods`` /
  ``supported_edge_methods``) and keeps the enabled names, so the enabled names in
  registration order must equal the conf list exactly (this also implies every name
  is registered, which ``tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py``
  checks on its own);
* graph integrity: every enabled edge's endpoint nodes are enabled (``EDGE_ENDPOINTS``,
  written from the relationship each edge method emits), and every chunked
  entity node with its own id has its linking edge enabled;
* the enabled phenotype method matches the phenotype class of the paired dataset's
  ``experiment_class`` (``PHENOTYPE_METHOD``);
* the constructor wiring: attributes, one ``wandb.init``, the exact method table and
  start-time payloads ``CellAdapter.__init__`` logs, and stdout;
* the refusal when the conf path does not exist: the exact message naming
  ``<adapters dir>/conf/<file>``, raised before ``wandb.init``.

The dataset is a ``SimpleNamespace`` with only ``name`` (all ``__init__`` reads); no
real dataset is built and nothing touches ``DATA_ROOT``.

Limit: the Het and Hom Hillenmeyer 2008 confs differ only in their header comment,
and the two Lopez 2024 confs are byte-identical, so for those pairs a constructor
that loaded its sibling's conf would pass the content check; only the missing-conf
test, which pins the exact file name in the refusal path, catches the swap.
"""

from __future__ import annotations

import math
import os.path as osp
import re
import typing
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
from omegaconf import OmegaConf

import torchcell.adapters.cell_adapter as cell_adapter_module
from tests.torchcell.adapters._sga_adapter_harness import (
    FIXED_NOW,
    FixedDatetime,
    Table,
    WandbRecorder,
)
from torchcell.datamodels import schema as s

PHENOTYPE_METHOD: dict[type[Any], str] = {
    s.FitnessPhenotype: "fitness phenotype",
    s.GeneInteractionPhenotype: "gene interaction phenotype",
    s.GeneEssentialityPhenotype: "gene essentiality phenotype",
    s.SyntheticLethalityPhenotype: "synthetic lethality phenotype",
    s.SyntheticRescuePhenotype: "synthetic rescue phenotype",
    s.CalMorphPhenotype: "calmorph phenotype",
    s.MicroarrayExpressionPhenotype: "microarray expression phenotype",
    s.RNASeqExpressionPhenotype: "rnaseq expression phenotype",
    s.PseudobulkExpressionPhenotype: "pseudobulk expression phenotype",
    s.VisualScorePhenotype: "visual score phenotype",
    s.MetabolitePhenotype: "metabolite phenotype",
    s.ProteinAbundancePhenotype: "protein abundance phenotype",
    s.EnvironmentResponsePhenotype: "environment response phenotype",
    s.ProductTiterPhenotype: "product titer phenotype",
    s.PromoterActivityPhenotype: "promoter activity phenotype",
    s.ProteinTurnoverPhenotype: "protein turnover phenotype",
    s.FluxPhenotype: "flux phenotype",
}

PHENOTYPE_CHUNKED = "<phenotype> (chunked)"
# Edge -> the node methods it connects (each entry: any one of the alternatives).
EDGE_ENDPOINTS: dict[str, list[tuple[str, ...]]] = {
    "experiment reference to dataset": [("experiment reference",), ("dataset",)],
    "experiment to dataset (chunked)": [("experiment (chunked)",), ("dataset",)],
    "experiment reference to experiment (chunked)": [
        ("experiment reference",),
        ("experiment (chunked)",),
    ],
    "genotype to experiment (chunked)": [
        ("genotype (chunked)", "segregant genotype (chunked)"),
        ("experiment (chunked)",),
    ],
    "perturbation to genotype (chunked)": [
        ("perturbation (chunked)", "bacterial perturbation (chunked)"),
        ("genotype (chunked)",),
    ],
    "crispr construct to perturbation (chunked)": [
        ("crispr construct (chunked)",),
        ("perturbation (chunked)", "bacterial perturbation (chunked)"),
    ],
    "environment to experiment (chunked)": [
        ("environment (chunked)",),
        ("experiment (chunked)",),
    ],
    "environment to experiment reference": [
        ("environment reference",),
        ("experiment reference",),
    ],
    "phenotype to experiment (chunked)": [
        (PHENOTYPE_CHUNKED,),
        ("experiment (chunked)",),
    ],
    "media to environment (chunked)": [
        ("media (chunked)",),
        ("environment (chunked)",),
    ],
    "temperature to environment (chunked)": [
        ("temperature (chunked)",),
        ("environment (chunked)",),
    ],
    # `phage perturbation` is content-addressed by the same projection as `environment
    # perturbation` and its graph class declares both as sources, so these two edge
    # methods address either node class (cell_adapter, "Phage challenges").
    "environment perturbation to environment (chunked)": [
        ("environment perturbation (chunked)", "phage perturbation (chunked)"),
        ("environment (chunked)",),
    ],
    "environment perturbation to environment reference": [
        ("environment perturbation reference", "phage perturbation reference"),
        ("environment reference",),
    ],
    "genome to experiment reference": [("genome",), ("experiment reference",)],
    "phenotype to experiment reference": [
        ("<phenotype> reference",),
        ("experiment reference",),
    ],
    "publication to experiment (chunked)": [
        ("publication (chunked)",),
        ("experiment (chunked)",),
    ],
}
# Chunked entity node -> the edge that links it into the graph.
NODE_LINK: dict[str, str] = {
    "genotype (chunked)": "genotype to experiment (chunked)",
    "segregant genotype (chunked)": "genotype to experiment (chunked)",
    "perturbation (chunked)": "perturbation to genotype (chunked)",
    "bacterial perturbation (chunked)": "perturbation to genotype (chunked)",
    "crispr construct (chunked)": "crispr construct to perturbation (chunked)",
    "environment (chunked)": "environment to experiment (chunked)",
    "media (chunked)": "media to environment (chunked)",
    "temperature (chunked)": "temperature to environment (chunked)",
    "environment perturbation (chunked)": (
        "environment perturbation to environment (chunked)"
    ),
    "phage perturbation (chunked)": (
        "environment perturbation to environment (chunked)"
    ),
    PHENOTYPE_CHUNKED: "phenotype to experiment (chunked)",
    "publication (chunked)": "publication to experiment (chunked)",
}


class Shape(NamedTuple):
    """The graph shape of one dataset, from which its conf is rebuilt."""

    phenotype: str
    perturbation: bool = True
    crispr: bool = False
    env_perturbation: bool = False
    # A bacteriophage challenge is served as `phage perturbation`, its OWN node class.
    # Since issue #756 the served `_environment_perturbation_node` filters phages out, so
    # the two lanes partition the environment's perturbations and a conf may enable both;
    # this flag and `env_perturbation` are independent, each pinning one lane.
    phage: bool = False
    mrf: float | None = 1.0
    # A bacterial genotype's leaves are served as `bacterial perturbation`, never as the
    # yeast `perturbation` class (cell_adapter.BACTERIAL_PERTURBATION_LEAVES).
    bacterial: bool = False


class AdapterCase(NamedTuple):
    """One adapter class, the conf file it must load, its shape, and the dataset class
    the case expects ``dataset_adapter_map`` to pair it with.
    """

    adapter_cls: type[Any]
    conf_name: str
    shape: Shape
    dataset_cls: type[Any]
    prints: bool = False


def expected_methods(shape: Shape, segregant: bool) -> tuple[list[str], list[str]]:
    """Node and edge method names in registration-table order."""
    nodes = ["experiment reference", "genome", "experiment (chunked)"]
    nodes.append("segregant genotype (chunked)" if segregant else "genotype (chunked)")
    if shape.perturbation:
        nodes.append(
            "bacterial perturbation (chunked)"
            if shape.bacterial
            else "perturbation (chunked)"
        )
    if shape.crispr:
        nodes.append("crispr construct (chunked)")
    nodes += [
        "environment (chunked)",
        "environment reference",
        "media (chunked)",
        "media reference",
        "temperature (chunked)",
        "temperature reference",
    ]
    if shape.env_perturbation and shape.phage:
        raise ValueError(
            "a conf enables `environment perturbation` or `phage perturbation`, never "
            "both: the served environment-perturbation method does not filter phages out"
        )
    if shape.env_perturbation:
        nodes += [
            "environment perturbation (chunked)",
            "environment perturbation reference",
        ]
    if shape.phage:
        nodes += ["phage perturbation (chunked)", "phage perturbation reference"]
    nodes += [
        f"{shape.phenotype} (chunked)",
        f"{shape.phenotype} reference",
        "dataset",
        "publication (chunked)",
    ]
    edges = [
        "experiment reference to dataset",
        "experiment to dataset (chunked)",
        "experiment reference to experiment (chunked)",
        "genotype to experiment (chunked)",
    ]
    if shape.perturbation:
        edges.append("perturbation to genotype (chunked)")
    if shape.crispr:
        edges.append("crispr construct to perturbation (chunked)")
    edges += [
        "environment to experiment (chunked)",
        "environment to experiment reference",
        "phenotype to experiment (chunked)",
        "media to environment (chunked)",
        "temperature to environment (chunked)",
    ]
    if shape.env_perturbation or shape.phage:
        edges += [
            "environment perturbation to environment (chunked)",
            "environment perturbation to environment reference",
        ]
    edges += [
        "genome to experiment reference",
        "phenotype to experiment reference",
        "publication to experiment (chunked)",
    ]
    return nodes, edges


def _entry(name: str, mrf: float | None) -> dict[str, Any]:
    if "(chunked)" in name and mrf is not None:
        return {"method_name": name, "memory_reduction_factor": mrf}
    return {"method_name": name}


def expected_conf(shape: Shape, segregant: bool) -> dict[str, Any]:
    """The whole conf as a plain container."""
    nodes, edges = expected_methods(shape, segregant)
    return {
        "cell_adapter": {
            "node_methods": [_entry(n, shape.mrf) for n in nodes],
            "edge_methods": [_entry(e, shape.mrf) for e in edges],
        }
    }


def _generic(name: str, phenotype: str) -> str:
    return name.replace(phenotype, "<phenotype>")


def assert_graph_integrity(nodes: list[str], edges: list[str], phenotype: str) -> None:
    """No dangling edge, no unlinked chunked entity node, no duplicate method."""
    assert len(set(nodes)) == len(nodes) and len(set(edges)) == len(edges)
    generic_nodes = {_generic(n, phenotype) for n in nodes}
    for edge in edges:
        for alternatives in EDGE_ENDPOINTS[edge]:
            assert generic_nodes & set(alternatives), (edge, alternatives)
    for node in generic_nodes:
        if node in NODE_LINK:
            assert NODE_LINK[node] in edges, (node, NODE_LINK[node])


def hinted_dataset_class(adapter_cls: type[Any]) -> type[Any]:
    """The class the constructor's ``dataset`` parameter is annotated with."""
    hint: type[Any] = typing.get_type_hints(adapter_cls.__init__)["dataset"]
    return hint


def dataset_phenotype_class(dataset_cls: type[Any]) -> type[Any]:
    """The ``phenotype`` field type of the dataset's ``experiment_class``."""
    experiment_cls = dataset_cls.experiment_class.fget(None)
    phenotype: type[Any] = typing.get_type_hints(experiment_cls)["phenotype"]
    return phenotype


def paired_dataset_class(adapter_cls: type[Any]) -> type[Any]:
    """The one dataset class ``dataset_adapter_map`` pairs with ``adapter_cls``."""
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    paired = [ds for ds, ad in dataset_adapter_map.items() if ad is adapter_cls]
    assert len(paired) == 1, (adapter_cls, paired)
    return paired[0]


def dataset_genotype_is_segregant(dataset_cls: type[Any]) -> bool:
    """Whether the dataset's ``experiment_class`` types its genotype as a segregant."""
    experiment_cls = dataset_cls.experiment_class.fget(None)
    return typing.get_type_hints(experiment_cls)["genotype"] is s.SegregantGenotype


def _rows_equal(actual: list[list[Any]], expected: list[list[Any]]) -> bool:
    if len(actual) != len(expected):
        return False
    for a_row, e_row in zip(actual, expected, strict=True):
        if a_row[:3] != e_row[:3]:
            return False
        a, e = a_row[3], e_row[3]
        if isinstance(e, float) and math.isnan(e):
            if not (isinstance(a, float) and math.isnan(a)):
                return False
        elif a != e:
            return False
    return True


def install_recorder(monkeypatch: pytest.MonkeyPatch) -> WandbRecorder:
    """``cell_adapter``'s ``wandb`` becomes a recorder; ``datetime.now`` is pinned."""
    rec = WandbRecorder()
    monkeypatch.setattr(cell_adapter_module, "wandb", rec)
    monkeypatch.setattr(cell_adapter_module, "datetime", FixedDatetime)
    return rec


def assert_construction(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Build the adapter on a name-only dataset and check everything in the docstring."""
    rec = install_recorder(monkeypatch)
    capsys.readouterr()
    dataset = SimpleNamespace(name="FakeDataset")
    adapter = case.adapter_cls(
        dataset=dataset,
        process_workers=3,
        io_workers=2,
        chunk_size=500,
        loader_batch_size=50,
    )
    out = capsys.readouterr().out

    paired = paired_dataset_class(case.adapter_cls)
    assert paired is case.dataset_cls, (paired, case.dataset_cls)
    segregant = dataset_genotype_is_segregant(paired)
    expected = expected_conf(case.shape, segregant)
    assert OmegaConf.to_container(adapter.config) == expected
    nodes, edges = expected_methods(case.shape, segregant)
    enabled_nodes, enabled_edges = set(nodes), set(edges)
    assert [m for m in adapter.supported_node_methods if m in enabled_nodes] == nodes
    assert [m for m in adapter.supported_edge_methods if m in enabled_edges] == edges
    assert_graph_integrity(nodes, edges, case.shape.phenotype)
    assert PHENOTYPE_METHOD[dataset_phenotype_class(paired)] == case.shape.phenotype

    assert adapter.dataset is dataset
    assert (adapter.process_workers, adapter.io_workers) == (3, 2)
    assert (adapter.chunk_size, adapter.loader_batch_size) == (500, 50)
    assert adapter.event == 0

    assert rec.init_calls == 1
    table_payload, start_payload = rec.logged
    assert list(table_payload) == ["FakeDataset_method_table"]
    table = table_payload["FakeDataset_method_table"]
    assert isinstance(table, Table)
    assert table.columns == ["event", "method", "data_type", "memory_reduction_factor"]

    def factor(name: str) -> float:
        if "(chunked)" not in name:
            return float("nan")
        return 1.0 if case.shape.mrf is None else case.shape.mrf

    rows = [[i + 1, n, "node", factor(n)] for i, n in enumerate(nodes)]
    rows += [[len(nodes) + i + 1, e, "edge", factor(e)] for i, e in enumerate(edges)]
    assert _rows_equal(table.data, rows), table.data
    assert start_payload == {
        "current_adapter_dataset_name": "FakeDataset",
        "current_adapter_dataset_start_time": FIXED_NOW.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if case.prints:
        name = case.adapter_cls.__name__
        assert out == f"{name} initialized with config: {adapter.config}\n"
    else:
        assert out == ""


def assert_missing_conf(
    case: AdapterCase, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the constructor names its exact path, before wandb."""
    rec = install_recorder(monkeypatch)
    fake_osp = SimpleNamespace(
        dirname=osp.dirname,
        abspath=osp.abspath,
        join=osp.join,
        exists=lambda path: False,
    )
    monkeypatch.setattr(module, "osp", fake_osp)
    assert module.__file__ is not None
    conf_path = osp.join(
        osp.dirname(osp.abspath(module.__file__)), "conf", case.conf_name
    )
    with pytest.raises(
        FileNotFoundError, match=f"^{re.escape(f'Config file not found: {conf_path}')}$"
    ):
        case.adapter_cls(
            dataset=SimpleNamespace(name="x"), process_workers=1, io_workers=1
        )
    assert rec.init_calls == 0
