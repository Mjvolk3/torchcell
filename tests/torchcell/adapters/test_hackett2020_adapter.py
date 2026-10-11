# tests/torchcell/adapters/test_hackett2020_adapter.py
# [[tests.torchcell.adapters.test_hackett2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_hackett2020_adapter.py
"""``torchcell.adapters.hackett2020_adapter`` constructor tests on a name-only fake dataset.

2026.10.11. The adapter adds no node or edge builder of its own; what it adds is the
conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The
expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``: an induction time-course expression phenotype,
the inducible promoter insertion as a first-class perturbation node, and the
beta-estradiol of every post-induction array as an environment-perturbation node.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.hackett2020_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.hackett2020_adapter import InductionTimecourseHackett2020Adapter
from torchcell.datasets.scerevisiae.hackett2020 import (
    InductionTimecourseHackett2020Dataset,
)

_INIT_CASES = [
    AdapterCase(
        InductionTimecourseHackett2020Adapter,
        "induction_timecourse_hackett2020_adapter.yaml",
        Shape(
            "induction timecourse expression phenotype",
            perturbation=True,
            env_perturbation=True,
        ),
        InductionTimecourseHackett2020Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``InductionTimecourseHackett2020Adapter`` loads
    ``conf/induction_timecourse_hackett2020_adapter.yaml``: 17 node and 15 edge methods
    (``induction timecourse expression phenotype``; perturbation nodes; environment
    perturbations; memory_reduction_factor 1.0 on every chunked method), checked against
    the dataset ``dataset_adapter_map`` pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
