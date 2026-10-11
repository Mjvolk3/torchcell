# tests/torchcell/adapters/test_hu2007_adapter.py
# [[tests.torchcell.adapters.test_hu2007_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_hu2007_adapter.py
"""``torchcell.adapters.hu2007_adapter`` constructor tests on a name-only fake dataset.

2026.10.11 (#1022). The adapter adds no node or edge builder of its own; what it adds
is the conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring.
The expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``: the Kemmeren 2014 shape with the error-model
score phenotype in place of the log-ratio one.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.hu2007_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.hu2007_adapter import MicroarrayScoreHu2007Adapter
from torchcell.datasets.scerevisiae.hu2007 import MicroarrayScoreHu2007Dataset

_INIT_CASES = [
    AdapterCase(
        MicroarrayScoreHu2007Adapter,
        "microarray_score_hu2007_adapter.yaml",
        Shape("microarray expression score phenotype"),
        MicroarrayScoreHu2007Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``MicroarrayScoreHu2007Adapter`` loads ``conf/microarray_score_hu2007_adapter.yaml``:
    15 node and 13 edge methods (``microarray expression score phenotype``; gene-keyed
    genotype; perturbation nodes; memory_reduction_factor 1.0 on every chunked method),
    checked against the dataset ``dataset_adapter_map`` pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
