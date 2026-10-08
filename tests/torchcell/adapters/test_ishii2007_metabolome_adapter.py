# tests/torchcell/adapters/test_ishii2007_metabolome_adapter.py
# [[tests.torchcell.adapters.test_ishii2007_metabolome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_ishii2007_metabolome_adapter.py
"""``torchcell/adapters/ishii2007_metabolome_adapter.py``: the conf it loads, the gate's view of it,
and (``--data``) the graph it emits from its dev-tree LMDB.

Ishii 2007 CE-TOFMS metabolome: 24 (BW25113 Keio disruptant) records in one
glucose-limited chemostat environment, BacterialMetaboliteExperiment. One
BacterialDeletionPerturbation per record, served as `bacterial perturbation`
(never the yeast `perturbation` class). All 24 share one medium at one dilution
rate, so no record carries an environment perturbation (0 in 24).
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
from torchcell.datasets.ecoli.ishii2007 import MetabolomeIshii2007Dataset

CASE = case_for(MetabolomeIshii2007Dataset)


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


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
