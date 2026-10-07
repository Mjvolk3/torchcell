# tests/torchcell/adapters/test_rousset2018_adapter.py
# [[tests.torchcell.adapters.test_rousset2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_rousset2018_adapter.py
"""``torchcell/adapters/rousset2018_adapter.py``: the conf it loads, the gate's view of it,
and (``--data``) the graph it emits from its dev-tree LMDB.

Rousset 2018 CRISPR-dCas9 screens: BacterialEnvironmentResponseExperiment. One
BacterialCrisprInterferencePerturbation per record, served as `bacterial perturbation`
with its spacer on a `crispr construct` node. The four phage-derived screens carry
exactly one PhagePerturbation each, so this is the one bacterial conf that enables `phage
perturbation`; it does NOT enable `environment perturbation`, because the served
`_environment_perturbation_node` does not filter phages out. The aTc inducer is a
component of the two media rather than an environment perturbation.
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
from torchcell.datasets.ecoli.rousset2018 import CrispriScreenRousset2018Dataset

CASE = case_for(CrispriScreenRousset2018Dataset)


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


def test_the_conf_serves_a_phage_and_not_an_environment_perturbation() -> None:
    """The one bacterial conf whose environment edit is a phage.

    The two classes are mutually exclusive because the served
    ``_environment_perturbation_node`` emits a phage under the ``environment
    perturbation`` label, so a conf enabling both would write each phage twice, once
    under each label, on one content id.
    """
    assert CASE.case.shape.phage is True
    assert CASE.case.shape.env_perturbation is False
    assert CASE.case.shape.crispr is True


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
