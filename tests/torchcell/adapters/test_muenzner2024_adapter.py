# tests/torchcell/adapters/test_muenzner2024_adapter.py
# [[tests.torchcell.adapters.test_muenzner2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_muenzner2024_adapter.py
"""``torchcell.adapters.muenzner2024_adapter`` constructor tests on a name-only dataset.

The two adapters add no node or edge builder of their own; what they add is the conf they
load (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. Both confs are the
compact-genotype shape Caudal 2024 uses (no per-perturbation node or edge), since every
natural isolate carries thousands of perturbations held on its genotype node. The
expected conf is rebuilt from that shape by ``_adapter_init_harness.expected_conf``.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.muenzner2024_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.muenzner2024_adapter import (
    ProteinTurnoverMuenzner2024Adapter,
    ProteomeMuenzner2024Adapter,
)
from torchcell.datasets.scerevisiae.muenzner2024 import (
    ProteinTurnoverMuenzner2024Dataset,
    ProteomeMuenzner2024Dataset,
)

_INIT_CASES = [
    AdapterCase(
        ProteomeMuenzner2024Adapter,
        "proteome_muenzner2024_adapter.yaml",
        Shape("protein abundance phenotype", perturbation=False),
        ProteomeMuenzner2024Dataset,
    ),
    AdapterCase(
        ProteinTurnoverMuenzner2024Adapter,
        "protein_turnover_muenzner2024_adapter.yaml",
        Shape("protein turnover phenotype", perturbation=False),
        ProteinTurnoverMuenzner2024Dataset,
    ),
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf each constructor loads is exactly the one its dataset's shape needs.

    The adapter is checked against the dataset ``dataset_adapter_map`` pairs it with; the
    conf lists its methods in run order, no edge dangles, every chunked entity node is
    linked, and the phenotype method matches the dataset's ``experiment_class``. The
    adapter keeps the dataset and the worker / chunk sizes it was given, calls
    ``wandb.init`` once and logs the method table; nothing is printed.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
