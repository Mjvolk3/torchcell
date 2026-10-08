# tests/torchcell/adapters/test_borchert2023_adapter.py
# [[tests.torchcell.adapters.test_borchert2023_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_borchert2023_adapter.py
"""``torchcell/adapters/borchert2023_adapter.py``: the conf it loads, the gate's view of
it, and (``--data``) the graph it emits from its dev-tree LMDB.

Borchert 2023 KT2440 RB-TnSeq, the 271 loci the Borchert 2024 compendium dropped:
BacterialEnvironmentResponseExperiment. One TransposonInsertionPerturbation per record,
served as `bacterial perturbation`. The environment always carries D-glucose as an
EnvironmentPhysicalPerturbation and, for the twelve stressed cultures, the stressor as a
SmallMoleculePerturbation, so the environment-perturbation pair is enabled. Same record
family and same graph classes as the compendium adapter, so this adds no graph class.
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
from torchcell.datasets.pputida.borchert2023 import RbTnseqBorchert2023Dataset

CASE = case_for(RbTnseqBorchert2023Dataset)


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


def test_the_conf_matches_the_compendium_adapter_s_enable_list() -> None:
    """The record family is the compendium's, so the two confs enable the same methods.

    A new graph node class would show up here as a method the compendium's conf does
    not name; this dataset adds none, which is why nothing is declared in
    ``torchcell_schema_config.yaml`` for it.
    """
    import os.path as osp

    import yaml

    import torchcell.adapters as adapters

    directory = osp.join(osp.dirname(adapters.__file__), "conf")

    def methods(slug: str) -> dict[str, list[str]]:
        with open(osp.join(directory, f"{slug}_adapter.yaml")) as handle:
            conf = yaml.safe_load(handle)["cell_adapter"]
        return {
            kind: [entry["method_name"] for entry in conf[f"{kind}_methods"]]
            for kind in ("node", "edge")
        }

    assert methods("rbtnseq_borchert2023") == methods("rbtnseq_borchert2024")


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
