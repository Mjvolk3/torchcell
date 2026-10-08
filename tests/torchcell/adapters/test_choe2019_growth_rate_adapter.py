# tests/torchcell/adapters/test_choe2019_growth_rate_adapter.py
# [[tests.torchcell.adapters.test_choe2019_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_choe2019_growth_rate_adapter.py
"""``torchcell/adapters/choe2019_growth_rate_adapter.py``: the conf it loads, the gate's
view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Choe 2019 Fig. 2c: two designed MS56 deletion strains, BacterialFitnessExperiment. One
``BacterialDeletionPerturbation`` per deleted gene, so the perturbation pair is enabled;
the panel adds no compound to its medium, so the environment-perturbation pair is off,
and the environment declares a ``ProvenanceGap`` on temperature.
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
from torchcell.datasets.ecoli.choe2019_growth_rate import GrowthRateChoe2019Dataset

CASE = case_for(GrowthRateChoe2019Dataset)


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


def test_conf_serves_the_deletions_and_no_environment_perturbation() -> None:
    """The deletions are served and the environment axis carries nothing extra.

    Both strains are "MS56, <lesion>::kan" in one M9 glucose medium with no added
    compound, so the environment-perturbation pair must stay off: enabling it would
    announce an axis no record populates.
    """
    conf = yaml.safe_load(
        (REPO_ROOT / "torchcell/adapters/conf" / CASE.case.conf_name).read_text(
            encoding="utf-8"
        )
    )
    names = [
        m["method_name"]
        for m in conf["cell_adapter"]["node_methods"]
        + conf["cell_adapter"]["edge_methods"]
    ]
    assert "bacterial perturbation (chunked)" in names
    assert "perturbation to genotype (chunked)" in names
    assert "fitness phenotype (chunked)" in names
    assert [n for n in names if "environment perturbation" in n] == []
    assert [n for n in names if "crispr" in n] == []
    assert "perturbation (chunked)" not in names


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
