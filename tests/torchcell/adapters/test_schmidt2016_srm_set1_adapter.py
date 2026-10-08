# tests/torchcell/adapters/test_schmidt2016_srm_set1_adapter.py
# [[tests.torchcell.adapters.test_schmidt2016_srm_set1_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_schmidt2016_srm_set1_adapter.py
"""``torchcell/adapters/schmidt2016_srm_set1_adapter.py``: the conf it loads, the
gate's view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Schmidt 2016 Table S2 SRM abundances: 11 BW25113 growth conditions,
BacterialProteinAbundanceExperiment, 41 protein keys per record. Every genotype is the
wild type, so neither `bacterial perturbation` nor `perturbation to genotype` is
enabled; the environment-perturbation pair is, because every record but the LB one and
the glucose reference carry a carbon-source or stress edit.

The two properties pinned here are the ones a reader could get wrong: that this arm
serves no gene perturbation although the paper does have deletion strains (they are
`GrowthRateSchmidt2016Dataset`'s records), and that its conf is its OWN -- a shared
module would make `kg_manifest` fingerprint a sibling's conf.
"""

from __future__ import annotations

import pytest
import yaml

from tests.torchcell.adapters._adapter_init_harness import (
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._bacterial_adapter_cases import (
    REPO_ROOT,
    adapter_module,
    assert_conf_registered_and_declared,
    assert_dev_store_graph,
    assert_gate_resolves_own_files,
    case_for,
)
from torchcell.datasets.ecoli.schmidt2016 import ProteomeSchmidt2016Dataset
from torchcell.datasets.ecoli.schmidt2016_srm import ProteomeSrmSet1Schmidt2016Dataset

CASE = case_for(ProteomeSrmSet1Schmidt2016Dataset)


def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_construction(CASE.case, monkeypatch, capsys)


def test_init_refuses_a_missing_conf_before_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_missing_conf(CASE.case, adapter_module(CASE), monkeypatch)


def test_conf_methods_are_registered_and_their_classes_declared() -> None:
    assert_conf_registered_and_declared(CASE)


def test_kg_manifest_resolves_the_dataset_to_its_own_module_and_conf() -> None:
    assert_gate_resolves_own_files(CASE)


def test_conf_is_not_the_stored_table_s6_blocks_conf() -> None:
    """Two confs, because two datasets: a shared one would fingerprint the wrong file."""
    stored = case_for(ProteomeSchmidt2016Dataset)
    assert CASE.case.conf_name != stored.case.conf_name
    assert CASE.module != stored.module


def test_conf_enables_no_gene_perturbation_method() -> None:
    """Every record is wild type with an empty genotype, so there is nothing to write."""
    conf = yaml.safe_load(
        (REPO_ROOT / "torchcell/adapters/conf" / CASE.case.conf_name).read_text(
            encoding="utf-8"
        )
    )
    names = [
        m["method_name"]
        for m in conf["cell_adapter"]["node_methods"]
        + conf["cell_adapter"]["edge_methods"]
    ]
    assert [n for n in names if "perturbation" in n and "environment" not in n] == []
    assert [n for n in names if "crispr" in n] == []
    assert "genotype (chunked)" in names
    assert "environment perturbation (chunked)" in names
    assert "protein abundance phenotype (chunked)" in names


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
