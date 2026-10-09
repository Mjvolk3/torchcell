# tests/torchcell/adapters/test_yunus2026_titer_adapter.py
# [[tests.torchcell.adapters.test_yunus2026_titer_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_yunus2026_titer_adapter.py
"""``torchcell/adapters/yunus2026_titer_adapter.py``: the conf it loads, the gate's view of it,
and (``--data``) the graph it emits from its dev-tree LMDB.

Yunus 2026 isoprenol titer: 125 IY1452 strains, ProductTiterExperiment, from the
hand-deposited Supplementary Note 1 input table. One
BacterialCrisprInterferencePerturbation per record, served as `bacterial perturbation`,
carrying a CrisprConstruct, so the crispr-construct pair is enabled. No heterologous
pathway leaf. The CultureEnvironment carries EnvironmentPhysicalPerturbation and
SmallMoleculePerturbation edits, so the environment-perturbation pair is enabled.
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
from torchcell.datasets.pputida.yunus2026 import IsoprenolTiterYunus2026Dataset

CASE = case_for(IsoprenolTiterYunus2026Dataset)


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
