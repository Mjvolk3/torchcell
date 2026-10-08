# tests/torchcell/adapters/test_choe2019_tf_knockout_adapter.py
# [[tests.torchcell.adapters.test_choe2019_tf_knockout_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_choe2019_tf_knockout_adapter.py
"""``torchcell/adapters/choe2019_tf_knockout_adapter.py``: the conf it loads, the gate's
view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Choe 2019 Supplementary Fig. 6: two Keio BW25113 single deletions,
BacterialFitnessExperiment. One ``BacterialDeletionPerturbation`` per record, so the
perturbation pair is enabled; one medium at one temperature with no added compound, so
the environment-perturbation pair is off.
"""

from __future__ import annotations

import pytest
import yaml

from tests.torchcell.adapters._adapter_init_harness import (
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._bacterial_adapter_cases import (
    REPO_ROOT,
    adapter_module,
    assert_conf_registered_and_declared,
    assert_dev_store_graph,
    assert_gate_resolves_own_files,
    case_for,
)
from torchcell.datasets.ecoli.choe2019_growth_rate import (
    TranscriptionFactorKnockoutChoe2019Dataset,
)

CASE = case_for(TranscriptionFactorKnockoutChoe2019Dataset)


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


def test_conf_is_the_same_shape_as_the_designed_deletion_arm() -> None:
    """Two arms of one paper, two confs, one shape; the slugs are what differ.

    Both are single-medium bacterial fitness datasets with a gene perturbation, so the
    method lists are identical and only the conf file name distinguishes them. That is
    the point of one conf per module: the gate resolves each dataset to its own file.
    """
    conf_dir = REPO_ROOT / "torchcell/adapters/conf"
    mine = yaml.safe_load((conf_dir / CASE.case.conf_name).read_text(encoding="utf-8"))
    sibling = yaml.safe_load(
        (conf_dir / "growth_rate_choe2019_adapter.yaml").read_text(encoding="utf-8")
    )
    assert CASE.case.conf_name == "tf_knockout_growth_choe2019_adapter.yaml"
    assert mine == sibling
    names = [
        m["method_name"]
        for m in mine["cell_adapter"]["node_methods"]
        + mine["cell_adapter"]["edge_methods"]
    ]
    assert "bacterial perturbation (chunked)" in names
    assert "fitness phenotype (chunked)" in names
    assert [n for n in names if "environment perturbation" in n] == []


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
