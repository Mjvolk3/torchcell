# tests/torchcell/adapters/test_messner2021_adapter.py
# [[tests.torchcell.adapters.test_messner2021_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_messner2021_adapter.py
"""``torchcell.adapters.messner2021_adapter`` constructor tests on a name-only fake dataset.

The adapter adds no node or edge builder of its own; what it adds is the conf it loads
(the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The expected conf is
rebuilt from the dataset's graph shape by ``_adapter_init_harness.expected_conf``: a
protein-abundance phenotype, no gene perturbation (every genotype is the reference
strain), and environment perturbations (the compound and the DMSO control).
"""

from __future__ import annotations

import pytest

import torchcell.adapters.messner2021_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.messner2021_adapter import ProteomeMessner2021Adapter
from torchcell.datasets.scerevisiae.messner2021 import ProteomeMessner2021Dataset

_INIT_CASES = [
    AdapterCase(
        ProteomeMessner2021Adapter,
        "proteome_messner2021_adapter.yaml",
        Shape("protein abundance phenotype", perturbation=False, env_perturbation=True),
        ProteomeMessner2021Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``conf/proteome_messner2021_adapter.yaml``: 16 node and 14 edge methods.

    The phenotype method matches the dataset's ``experiment_class``, no perturbation or
    perturbation->genotype method is enabled, both environment perturbation methods
    are, and memory_reduction_factor is 1.0 on every chunked method. The adapter keeps
    the dataset and the worker / chunk sizes it was given, calls ``wandb.init`` once and
    logs the method table; nothing is printed.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
