# tests/torchcell/adapters/test_schastnaya2021_adapter.py
# [[tests.torchcell.adapters.test_schastnaya2021_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_schastnaya2021_adapter.py
"""``torchcell/adapters/schastnaya2021_adapter.py``: the conf it loads, the gate's view
of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Schastnaya 2021 deletion metabolome: 29 (KEIO deletion strain, carbon source) records,
BacterialMetaboliteExperiment. One BacterialDeletionPerturbation per record, served as
`bacterial perturbation` (never the yeast `perturbation` class). The carbon source is
part of the medium, so no record carries an environment perturbation (0 in 29).
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
from torchcell.datasets.ecoli.schastnaya2021 import MetabolomeSchastnaya2021Dataset

CASE = case_for(MetabolomeSchastnaya2021Dataset)


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
