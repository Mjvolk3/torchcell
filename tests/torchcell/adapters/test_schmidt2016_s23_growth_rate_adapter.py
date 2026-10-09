# tests/torchcell/adapters/test_schmidt2016_s23_growth_rate_adapter.py
# [[tests.torchcell.adapters.test_schmidt2016_s23_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_schmidt2016_s23_growth_rate_adapter.py
"""``torchcell/adapters/schmidt2016_s23_growth_rate_adapter.py``: the conf it loads, the
gate's view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Schmidt 2016 Table S23: 15 records, one per kept (growth condition, BW25113) row,
``BacterialEnvironmentResponseExperiment`` carrying the absolute growth rate in h^-1.
Every released row is wild type (0 perturbations in 15 of 15 records), so neither
``bacterial perturbation`` nor ``perturbation to genotype`` is enabled. 14 of the 15
records carry an ``EnvironmentPhysicalPerturbation`` or a ``SmallMoleculePerturbation``
(carbon source, NaCl, pH), so the environment-perturbation pair is enabled; the LB
record carries none, because its edit is its complex medium.
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
from torchcell.datasets.ecoli.schmidt2016_s23_growth_rate import (
    GrowthRateS23Schmidt2016Dataset,
)

CASE = case_for(GrowthRateS23Schmidt2016Dataset)


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
