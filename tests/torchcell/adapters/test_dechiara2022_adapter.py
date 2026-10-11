# tests/torchcell/adapters/test_dechiara2022_adapter.py
# [[tests.torchcell.adapters.test_dechiara2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_dechiara2022_adapter.py
"""``torchcell.adapters.dechiara2022_adapter`` constructor tests on name-only datasets.

The three adapters add no node or edge builder of their own; what each adds is the conf
it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The checks
are ``_adapter_init_harness``'s: the exact conf rebuilt from the dataset's graph shape,
the wiring, and the refusal of a missing conf.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.dechiara2022_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.dechiara2022_adapter import (
    CellYieldDeChiara2022Adapter,
    DoublingTimeDeChiara2022Adapter,
    LifeCycleDeChiara2022Adapter,
)
from torchcell.datasets.scerevisiae.dechiara2022 import (
    CellYieldDeChiara2022Dataset,
    DoublingTimeDeChiara2022Dataset,
    LifeCycleDeChiara2022Dataset,
)

_INIT_CASES = [
    AdapterCase(
        LifeCycleDeChiara2022Adapter,
        "life_cycle_dechiara2022_adapter.yaml",
        Shape("life cycle phenotype", perturbation=False),
        LifeCycleDeChiara2022Dataset,
    ),
    AdapterCase(
        DoublingTimeDeChiara2022Adapter,
        "doubling_time_dechiara2022_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=False, env_perturbation=True
        ),
        DoublingTimeDeChiara2022Dataset,
    ),
    AdapterCase(
        CellYieldDeChiara2022Adapter,
        "cell_yield_dechiara2022_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=False, env_perturbation=True
        ),
        CellYieldDeChiara2022Dataset,
    ),
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf each constructor loads is exactly the one its dataset's shape needs.

    * ``LifeCycleDeChiara2022Adapter``: ``life cycle phenotype``; gene-keyed genotype
      with NO perturbation nodes (each isolate carries ~5,000); no environment
      perturbation (the five protocols carry none).
    * ``DoublingTimeDeChiara2022Adapter`` and ``CellYieldDeChiara2022Adapter``:
      ``environment response phenotype``; no perturbation nodes; the carbon or nitrogen
      source served as ``environment perturbation`` nodes.

    Each adapter is checked against the dataset ``dataset_adapter_map`` pairs it with,
    the conf order is the registration order, no edge dangles, every chunked entity
    node is linked and the phenotype method matches the dataset's ``experiment_class``.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
