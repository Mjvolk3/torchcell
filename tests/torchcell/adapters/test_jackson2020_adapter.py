# tests/torchcell/adapters/test_jackson2020_adapter.py
# [[tests.torchcell.adapters.test_jackson2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_jackson2020_adapter.py
"""``torchcell.adapters.jackson2020_adapter`` constructor tests on a name-only fake dataset.

2026.10.11. The adapter adds no node or edge builder of its own; what it adds is the
conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The
expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``: a pseudobulk UMI-count phenotype, one
first-class perturbation node per barcoded deletion, and environment-perturbation nodes
for the RAPA condition's rapamycin.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.jackson2020_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.jackson2020_adapter import (
    TfDeletionPseudobulkJackson2020Adapter,
)
from torchcell.datasets.scerevisiae.jackson2020 import (
    TfDeletionPseudobulkJackson2020Dataset,
)

_INIT_CASES = [
    AdapterCase(
        TfDeletionPseudobulkJackson2020Adapter,
        "tf_deletion_pseudobulk_jackson2020_adapter.yaml",
        Shape(
            "pseudobulk umi count phenotype", perturbation=True, env_perturbation=True
        ),
        TfDeletionPseudobulkJackson2020Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``TfDeletionPseudobulkJackson2020Adapter`` loads
    ``conf/tf_deletion_pseudobulk_jackson2020_adapter.yaml`` (``pseudobulk umi count
    phenotype``; per-perturbation nodes; environment perturbations; memory_reduction_factor
    1.0 on every chunked method), checked against the dataset ``dataset_adapter_map``
    pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
