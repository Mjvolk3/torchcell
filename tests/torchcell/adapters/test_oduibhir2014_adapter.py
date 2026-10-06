# tests/torchcell/adapters/test_oduibhir2014_adapter.py
# [[tests.torchcell.adapters.test_oduibhir2014_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_oduibhir2014_adapter.py
"""``torchcell.adapters.oduibhir2014_adapter`` constructor tests on a name-only fake dataset.

2026.10.06, Phase 21. The adapter adds no node or edge builder of its own; what it adds
is the conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring.
The dataset is ``SimpleNamespace(name="FakeDataset")``; ``wandb`` and ``datetime`` in
``cell_adapter`` are replaced by the recorder and the pinned clock of
``_sga_adapter_harness``. The expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.oduibhir2014_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.oduibhir2014_adapter import SmfODuibhir2014Adapter
from torchcell.datasets.scerevisiae.oduibhir2014 import SmfODuibhir2014Dataset

# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        SmfODuibhir2014Adapter,
        "smf_oduibhir2014_adapter.yaml",
        Shape("fitness phenotype", mrf=None),
        SmfODuibhir2014Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``SmfODuibhir2014Adapter`` loads ``conf/smf_oduibhir2014_adapter.yaml``: 15 node and 13 edge methods (``fitness phenotype``; gene-keyed genotype; perturbation nodes; no memory_reduction_factor keys).

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
