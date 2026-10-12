# tests/torchcell/adapters/test_vaishnav2022_adapter.py
# [[tests.torchcell.adapters.test_vaishnav2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_vaishnav2022_adapter.py
"""``torchcell.adapters.vaishnav2022_adapter`` constructor tests on a name-only fake dataset.

The adapter adds no node or edge builder of its own; what it adds is the conf it loads
(the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The expected conf is
rebuilt from the dataset's graph shape by ``_adapter_init_harness.expected_conf``: a
pooled sort-seq expression phenotype and the variable-promoter reporter construct as a
first-class perturbation node, with NO environment-perturbation lane, since the assay
varies the genome's content and never the medium.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.vaishnav2022_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.vaishnav2022_adapter import (
    PooledPromoterExpressionVaishnav2022Adapter,
)
from torchcell.datasets.scerevisiae.vaishnav2022 import (
    PooledPromoterExpressionVaishnav2022Dataset,
)

_INIT_CASES = [
    AdapterCase(
        PooledPromoterExpressionVaishnav2022Adapter,
        "pooled_promoter_expression_vaishnav2022_adapter.yaml",
        Shape("pooled sort seq expression phenotype", perturbation=True),
        PooledPromoterExpressionVaishnav2022Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``PooledPromoterExpressionVaishnav2022Adapter`` loads
    ``conf/pooled_promoter_expression_vaishnav2022_adapter.yaml``: the pooled sort-seq
    expression phenotype and its reference, perturbation nodes carrying the released
    sequence, and ``memory_reduction_factor`` 1.0 on every chunked method, checked
    against the dataset ``dataset_adapter_map`` pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
