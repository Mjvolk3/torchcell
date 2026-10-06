# tests/torchcell/adapters/test_cachera2023_adapter.py
# [[tests.torchcell.adapters.test_cachera2023_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_cachera2023_adapter.py
"""``torchcell.adapters.cachera2023_adapter`` constructor tests on a name-only fake dataset.

2026.10.06, Phase 21. The adapter adds no node or edge builder of its own; what it adds
is the conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring.
The dataset is ``SimpleNamespace(name="FakeDataset")``; ``wandb`` and ``datetime`` in
``cell_adapter`` are replaced by the recorder and the pinned clock of
``_sga_adapter_harness``. The expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.cachera2023_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cachera2023_adapter import BetaxanthinCachera2023Adapter
from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset

# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        BetaxanthinCachera2023Adapter,
        "betaxanthin_cachera2023_adapter.yaml",
        Shape("metabolite phenotype"),
        BetaxanthinCachera2023Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``BetaxanthinCachera2023Adapter`` loads ``conf/betaxanthin_cachera2023_adapter.yaml``: 15 node and 13 edge methods (``metabolite phenotype``; gene-keyed genotype; perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

    The adapter is checked against the dataset ``dataset_adapter_map`` pairs it with
    (asserted to be the class above). The conf lists its methods in the order the
    adapter runs them, no edge dangles, every chunked entity node is linked, and the
    phenotype method matches that dataset's ``experiment_class``. The adapter keeps the dataset and the worker / chunk sizes
    it was given (3, 2, 500, 50), calls ``wandb.init`` once and logs the method table
    (event number, name, node/edge, factor or NaN for a non-chunked method) then the
    dataset name and the pinned start time; nothing is printed.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
