# tests/torchcell/adapters/test_volk2021_inhibitor_bioscreen_adapter.py
# [[tests.torchcell.adapters.test_volk2021_inhibitor_bioscreen_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_volk2021_inhibitor_bioscreen_adapter.py
"""The private Bioscreen adapter: its conf, and the nodes its records project onto.

The node methods are called on the loader's own records (``run_records`` on a synthetic
two-WT, one-inhibitor run): the genotype node of an empty genotype, one environment
perturbation node per inhibitor, the phenotype node of a grown and a no-growth well,
the publication node of the preliminary report, and the genome node carrying bAID's
integrated cassette. ``get_nodes()`` over a built dataset is not exercised here: the
dataset build is blocked on ``ExperimentDataset.gene_set`` refusing an empty gene set
(``tests/torchcell/datasets/private_torchcell/test_volk2021_inhibitor_bioscreen.py``).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os.path as osp
import re
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import torchcell.adapters.volk2021_inhibitor_bioscreen_adapter as adapter_module
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import StrainReferenceGenome
from torchcell.datasets.private_torchcell import bioscreen as b
from torchcell.datasets.private_torchcell import volk2021_inhibitor_bioscreen as v
from torchcell.knowledge_graphs.kg_manifest import adapter_conf_name

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "inhibitor_bioscreen_volk2021_adapter.yaml",
)


def _tables() -> tuple[set[str], set[str]]:
    """Every node and edge method name ``CellAdapter.__init__`` registers."""
    source = inspect.getsource(CellAdapter.__init__)
    node_block, edge_block = source.split("self.edge_methods = [", 1)
    node_block = node_block.split("self.node_methods = [", 1)[1]
    pattern = r'"([^"]+)",\s*\n?\s*self\._'
    return set(re.findall(pattern, node_block)), set(re.findall(pattern, edge_block))


def _conf() -> tuple[list[str], list[str]]:
    with open(CONF) as handle:
        conf = yaml.safe_load(handle)["cell_adapter"]
    return (
        [m["method_name"] for m in conf["node_methods"]],
        [m["method_name"] for m in conf["edge_methods"]],
    )


def test_the_adapter_class_names_its_conf() -> None:
    assert (
        adapter_conf_name(adapter_module.InhibitorBioscreenVolk2021Adapter)
        == "inhibitor_bioscreen_volk2021_adapter.yaml"
    )
    assert osp.isfile(CONF)


def test_every_configured_method_exists_and_no_gene_perturbation_is_served() -> None:
    node_names, edge_names = _tables()
    nodes, edges = _conf()
    assert set(nodes) <= node_names, set(nodes) - node_names
    assert set(edges) <= edge_names, set(edges) - edge_names
    assert "genotype (chunked)" in nodes and "genotype to experiment (chunked)" in edges
    assert "environment perturbation (chunked)" in nodes
    assert "environment response phenotype (chunked)" in nodes
    # an empty genotype has no perturbation to emit
    assert "perturbation (chunked)" not in nodes
    assert "perturbation to genotype (chunked)" not in edges


def _wells() -> list[b.Well]:
    """WT wells 1 and 101 (2.0 h, 2.5 h); FF+AA well 2 grown (4.0 h); well 3 flat."""
    out = []
    for well, doses, gt in (
        (1, {}, 2.0),
        (2, {b.Inhibitor.FF: 1.0, b.Inhibitor.AA: 0.4}, 4.0),
        (3, {b.Inhibitor.FF: 1.0}, None),
        (101, {}, 2.5),
    ):
        d = b.no_doses() | doses
        out.append(
            b.Well(
                run=b.Run.ex26,
                plate=b.plate_of(well),
                well=well,
                condition=b.WILD_TYPE if not doses else "x",
                doses_g_per_l=d,
                biological_replicate_id=b.plate_of(well),
                generation_time_h=gt,
                grew=gt is not None,
                trait_source=b.TraitSource.raw_curve,
            )
        )
    return out


@pytest.fixture
def records() -> v.RunRecords:
    return v.run_records(b.Run.ex26, _wells(), "InhibitorBioscreenVolk2021Dataset")


def _data(records: v.RunRecords, i: int) -> dict[str, Any]:
    return {
        "experiment": records.experiments[i],
        "experiment_reference": records.reference,
        "publication": v.publication(),
    }


def _bare() -> CellAdapter:
    return CellAdapter.__new__(CellAdapter)


def test_the_empty_genotype_is_one_genotype_node(records: v.RunRecords) -> None:
    node_fn = CellAdapter._genotype_node.__wrapped__  # type: ignore[attr-defined]  # functools.wraps
    nodes = [
        node_fn(_bare(), _data(records, i), "genotype (chunked)") for i in range(4)
    ]
    assert {n.get_id() for n in nodes} == {
        hashlib.sha256(json.dumps({"perturbations": []}).encode()).hexdigest()
    }
    props = nodes[0].get_properties()
    assert {k: props[k] for k in props if k not in ("id", "preferred_id")} == {
        "systematic_gene_names": [],
        "perturbed_gene_names": [],
        "perturbation_types": [],
    }
    perturbation_fn = CellAdapter._perturbation_node.__wrapped__  # type: ignore[attr-defined]  # functools.wraps
    assert perturbation_fn(_bare(), _data(records, 1), "perturbation (chunked)") == []


def test_each_inhibitor_is_an_environment_perturbation_node(
    records: v.RunRecords,
) -> None:
    environment = records.experiments[1].environment
    props = [
        CellAdapter._environment_perturbation_node_from(p).get_properties()
        for p in environment.perturbations
    ]
    assert [
        (p["compound_name"], p["inchikey"], p["concentration_value"]) for p in props
    ] == [
        ("furfural", "HYBBIBNJHNGZAN-UHFFFAOYSA-N", 1.0),
        ("acetic acid", "QTBSBXVTEAMEQO-UHFFFAOYSA-N", 0.4),
    ]
    assert {p["concentration_unit"] for p in props} == {"g/L"}
    assert records.experiments[0].environment.perturbations == []


def test_phenotype_nodes_carry_the_rate_or_the_call(records: v.RunRecords) -> None:
    node_fn = CellAdapter._environment_response_phenotype_node.__wrapped__  # type: ignore[attr-defined]  # functools.wraps
    grown = node_fn(_bare(), _data(records, 1), "x").get_properties()
    flat = node_fn(_bare(), _data(records, 2), "x").get_properties()
    assert grown["measurement_type"] == "relative_growth_rate"
    assert grown["environment_response"] == pytest.approx(2.25 / 4.0)
    assert grown["assay_type"] == "liquid_od_growth"
    assert grown["screen_id"] == "ex26:well2"
    assert flat["measurement_type"] == "categorical"
    assert flat["category"] == "severely_reduced"
    assert flat["category_label"] == "no growth within 96 h"
    assert flat["environment_response"] is None


def test_the_publication_node_is_keyed_by_the_deposited_report(
    records: v.RunRecords,
) -> None:
    node_fn = CellAdapter._publication_node.__wrapped__  # type: ignore[attr-defined]  # functools.wraps
    node = node_fn(_bare(), _data(records, 0), "publication (chunked)")
    props = node.get_properties()
    assert node.get_preferred_id() == (
        f"publication_paper.pdf sha256:{b.REPORT_PDF_SHA256}"
    )
    assert props["source_type"] == "preliminary_report"
    assert props["title"] == b.REPORT_TITLE
    assert props["doi"] is None and props["pubmed_id"] is None


def test_the_genome_node_carries_bAID_and_its_cassette(records: v.RunRecords) -> None:
    adapter = _bare()
    adapter.dataset = SimpleNamespace(
        experiment_reference_index=[SimpleNamespace(reference=records.reference)]
    )
    (node,) = adapter._get_genome_nodes()
    props = node.get_properties()
    assert props["strain"] == "bAID"
    genome = StrainReferenceGenome.model_validate_json(props["serialized_data"])
    assert [c.name for c in genome.background.integrations] == [
        "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]"
    ]
    assert genome.background.parents == ["BY4742"]
