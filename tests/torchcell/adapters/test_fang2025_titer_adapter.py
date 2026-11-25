# tests/torchcell/adapters/test_fang2025_titer_adapter.py
# [[tests.torchcell.adapters.test_fang2025_titer_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_fang2025_titer_adapter.py
"""``torchcell/adapters/fang2025_titer_adapter.py``: the conf it loads, the gate's view
of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Fang 2025 per-strain FFA titers: 306 records, ProductTiterExperiment, one per released
biological replicate culture of 102 strains. 441 perturbations over three leaves served
as `perturbation`: CrisprInterferencePerturbation (each with its CrisprConstruct, so the
crispr-construct pair is enabled), GeneAdditionPerturbation and
DeletionStrainPerturbation; the two unmodulated control strains carry an empty genotype.
The environment carries 1 mM IPTG for every record and 10 mM arabinose for the P_BAD
fadR strain, so the environment-perturbation pair is enabled, and every record states
30 C and the one modified-M9 medium.
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
from torchcell.datasets.ecoli.fang2025 import FfaTiterFang2025Dataset

CASE = case_for(FfaTiterFang2025Dataset)


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
