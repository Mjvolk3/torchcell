# tests/torchcell/adapters/test_schmidt2016_growth_rate_adapter.py
# [[tests.torchcell.adapters.test_schmidt2016_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_schmidt2016_growth_rate_adapter.py
"""``torchcell/adapters/schmidt2016_growth_rate_adapter.py``: the conf it loads, the
gate's view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Schmidt 2016 Table S24: six (KEIO deletion strain, medium) records,
BacterialFitnessExperiment. One ``BacterialDeletionPerturbation`` per record, so the
perturbation pair is enabled -- this is the one Schmidt 2016 dataset that serves a gene
perturbation, which is why the perturbation methods are pinned ON here and OFF in the
three proteome confs of the same paper.
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
from torchcell.datasets.ecoli.schmidt2016_growth_rate import (
    GrowthRateSchmidt2016Dataset,
)

CASE = case_for(GrowthRateSchmidt2016Dataset)


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


def test_conf_serves_the_deletion_perturbation_and_the_carbon_source() -> None:
    """The one Schmidt 2016 dataset with a gene perturbation, and both axes are served.

    The genotype leaf is a ``bacterial perturbation`` node joined to its genotype, and
    the medium's carbon source is an environment perturbation, so both pairs are on.
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
    assert "environment perturbation (chunked)" in names
    assert "fitness phenotype (chunked)" in names
    assert [n for n in names if "crispr" in n] == []
    assert "perturbation (chunked)" not in names


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)
