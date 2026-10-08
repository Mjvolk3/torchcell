# tests/torchcell/adapters/test_fang2025_adapter.py
# [[tests.torchcell.adapters.test_fang2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_fang2025_adapter.py
"""``torchcell/adapters/fang2025_adapter.py``: the conf it loads, the gate's view of it,
and (``--data``) the graph it emits from its dev-tree LMDB.

Fang 2025 CRISPRi-FACS free-fatty-acid enrichment: 15,708 records,
BacterialEnvironmentResponseExperiment. 16,187 perturbations, every one a
BacterialCrisprInterferencePerturbation served as `bacterial perturbation`; a record
carries more than one where its guide targets a multi-member BLASTN cluster, and every
round-2 record carries one extra for the pcnBi host's constant pcnB repression. Every
leaf carries a CrisprConstruct, so the crispr-construct pair is enabled. The environment
carries one SmallMoleculePerturbation (1 mM IPTG), so the environment-perturbation pair
is enabled. Every record states 30 C and the one modified-M9 medium, so the temperature
and media methods emit for every record.
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
from torchcell.datasets.ecoli.fang2025 import CrispriGuideFfaEnrichmentFang2025Dataset

CASE = case_for(CrispriGuideFfaEnrichmentFang2025Dataset)


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
