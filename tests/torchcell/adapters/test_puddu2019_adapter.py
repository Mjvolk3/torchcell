# tests/torchcell/adapters/test_puddu2019_adapter.py
# [[tests.torchcell.adapters.test_puddu2019_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_puddu2019_adapter.py
"""``torchcell.adapters.puddu2019_adapter`` constructor tests on a name-only fake dataset.

The adapter adds no node or edge builder of its own; what it adds is the conf it loads
(the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The expected conf is
rebuilt from the dataset's graph shape by ``_adapter_init_harness.expected_conf``: a
genome-architecture phenotype, perturbation nodes (a colony carries exactly one deletion)
and no environment perturbation (one medium at one temperature).
"""

from __future__ import annotations

import pytest

import torchcell.adapters.puddu2019_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.puddu2019_adapter import GenomeArchitecturePuddu2019Adapter
from torchcell.datasets.scerevisiae.puddu2019 import GenomeArchitecturePuddu2019Dataset

_INIT_CASES = [
    AdapterCase(
        GenomeArchitecturePuddu2019Adapter,
        "genome_architecture_puddu2019_adapter.yaml",
        Shape("genome architecture phenotype"),
        GenomeArchitecturePuddu2019Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    ``conf/genome_architecture_puddu2019_adapter.yaml``: 15 node and 13 edge methods
    (``genome architecture phenotype``; gene-keyed genotype WITH perturbation nodes,
    since the designed deletion and its detected barcodes are what a query joins on; no
    environment perturbation; memory_reduction_factor 1.0 on every chunked method),
    checked against the dataset ``dataset_adapter_map`` pairs it with.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
