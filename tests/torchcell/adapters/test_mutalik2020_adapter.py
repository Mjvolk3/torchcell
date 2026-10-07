# tests/torchcell/adapters/test_mutalik2020_adapter.py
# [[tests.torchcell.adapters.test_mutalik2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_mutalik2020_adapter.py
"""``torchcell/adapters/mutalik2020_adapter.py``: the conf it loads, the gate's view of
it, and (``--data``) the graph it emits from its dev-tree LMDB.

Mutalik 2020 phage-resistance RB-TnSeq: BacterialEnvironmentResponseExperiment. One
TransposonInsertionPerturbation per record, served as `bacterial perturbation`. The only
environment perturbation the row carries is the challenge's PhagePerturbation, so the
conf enables `phage perturbation` and NOT `environment perturbation`; the shared cases
check that the two are never both on, and that the records' phages are served under the
phage class.
"""

from __future__ import annotations

import pytest

from tests.torchcell.adapters._adapter_init_harness import (
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._bacterial_adapter_cases import (
    adapter_module,
    assert_conf_registered_and_declared,
    assert_dev_store_graph,
    assert_gate_resolves_own_files,
    case_for,
)
from torchcell.datasets.ecoli.mutalik2020 import PhageRbTnseqMutalik2020Dataset

CASE = case_for(PhageRbTnseqMutalik2020Dataset)


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


def test_the_conf_serves_the_phage_class_and_not_the_served_environment_class() -> None:
    """The mutual exclusion, read off this conf rather than off the shape.

    ``_environment_perturbation_node`` does not filter phages out, so enabling both
    classes would write every phage twice under two labels on one content id. The
    ``environment perturbation to environment`` edges stay enabled: a phage node's id is
    the same composition projection, so they address it unchanged.
    """
    import os.path as osp

    import yaml

    import torchcell.adapters as adapters

    conf_path = osp.join(osp.dirname(adapters.__file__), "conf", CASE.case.conf_name)
    conf = yaml.safe_load(open(conf_path, encoding="utf-8"))
    nodes = [m["method_name"] for m in conf["cell_adapter"]["node_methods"]]
    edges = [m["method_name"] for m in conf["cell_adapter"]["edge_methods"]]
    assert "phage perturbation (chunked)" in nodes
    assert "phage perturbation reference" in nodes
    assert "environment perturbation (chunked)" not in nodes
    assert "environment perturbation reference" not in nodes
    assert "environment perturbation to environment (chunked)" in edges
    assert "environment perturbation to environment reference" in edges


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
