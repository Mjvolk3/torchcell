# tests/torchcell/adapters/test_piotrowski2017_adapter.py
# [[tests.torchcell.adapters.test_piotrowski2017_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_piotrowski2017_adapter.py
"""``torchcell.adapters.piotrowski2017_adapter`` constructor tests on a name-only fake dataset.

2026.10.11. The adapter adds no node or edge builder of its own; what it adds is the
conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The
expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``: an environment-response phenotype, the
gene-keyed genotype WITH perturbation nodes (the barcoded diagnostic-pool deletion), and
environment-perturbation nodes for the dosed compound, the shape the Vanacloig 2022 conf
serves.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.piotrowski2017_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.piotrowski2017_adapter import EnvChemgenPiotrowski2017Adapter
from torchcell.datasets.scerevisiae.piotrowski2017 import (
    EnvChemgenPiotrowski2017Dataset,
)

_INIT_CASES = [
    AdapterCase(
        EnvChemgenPiotrowski2017Adapter,
        "env_chemgen_piotrowski2017_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=True, env_perturbation=True
        ),
        EnvChemgenPiotrowski2017Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``EnvChemgenPiotrowski2017Adapter`` loads
    ``conf/env_chemgen_piotrowski2017_adapter.yaml``: 17 node and 15 edge methods
    (``environment response phenotype``; gene-keyed genotype; perturbation nodes;
    environment perturbations; memory_reduction_factor 1.0 on every chunked method),
    checked against the dataset ``dataset_adapter_map`` pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
