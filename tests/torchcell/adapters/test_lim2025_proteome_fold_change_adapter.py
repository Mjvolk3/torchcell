# tests/torchcell/adapters/test_lim2025_proteome_fold_change_adapter.py
# [[tests.torchcell.adapters.test_lim2025_proteome_fold_change_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_lim2025_proteome_fold_change_adapter.py
"""``torchcell/adapters/lim2025_proteome_fold_change_adapter.py``: the conf it loads, the
gate's view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Lim 2025's four released evolved-isolate-over-IPL400 proteome contrasts: two isolates
(A10_F63_I1, A12_F53_I1) x two conditions (M9 at 4 g/L glucose, the same with 4 g/L
isoprenol), BacterialProteinFoldChangeExperiment. Each record's numerator genotype carries
IPL400's seven BacterialDeletionPerturbation leaves served as `bacterial perturbation`
plus that isolate's called variants served as `bacterial sequence variant perturbation`
(#731), and the per-protein log2 fold changes with their unadjusted and BH-adjusted
p-values as the already-served `protein fold change phenotype` -- never `protein
abundance phenotype`, which is the absolute sibling's class. Two of the four records carry
isoprenol as a SmallMoleculePerturbation, so the environment-perturbation pair is enabled
and emits nothing for the other two; every record gaps temperature. No phage and no
CRISPRi construct appears in any record.
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
from torchcell.datasets.pputida.lim2025 import ProteomeFoldChangeLim2025Dataset

CASE = case_for(ProteomeFoldChangeLim2025Dataset)


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
