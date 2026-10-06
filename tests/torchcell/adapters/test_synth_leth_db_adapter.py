# tests/torchcell/adapters/test_synth_leth_db_adapter.py
# [[tests.torchcell.adapters.test_synth_leth_db_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_synth_leth_db_adapter.py
"""``torchcell.adapters.synth_leth_db_adapter`` constructor tests on a name-only fake dataset.

2026.10.06, Phase 21. The adapter adds no node or edge builder of its own; what it adds
is the conf it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring.
The dataset is ``SimpleNamespace(name="FakeDataset")``; ``wandb`` and ``datetime`` in
``cell_adapter`` are replaced by the recorder and the pinned clock of
``_sga_adapter_harness``. The expected conf is rebuilt from the dataset's graph shape by
``_adapter_init_harness.expected_conf``.
"""

from __future__ import annotations

import pytest

import torchcell.adapters.synth_leth_db_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.synth_leth_db_adapter import (
    SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbAdapter,
)
from torchcell.datasets.scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
)

# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        SynthLethalityYeastSynthLethDbAdapter,
        "synth_lethality_yeast_synth_leth_db_adapter.yaml",
        Shape("synthetic lethality phenotype", mrf=None),
        SynthLethalityYeastSynthLethDbDataset,
        prints=True,
    ),
    AdapterCase(
        SynthRescueYeastSynthLethDbAdapter,
        "synth_rescue_yeast_synth_leth_db_adapter.yaml",
        Shape("synthetic rescue phenotype", mrf=None),
        SynthRescueYeastSynthLethDbDataset,
        prints=True,
    ),
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``SynthLethalityYeastSynthLethDbAdapter`` loads ``conf/synth_lethality_yeast_synth_leth_db_adapter.yaml``: 15 node and 13 edge methods (``synthetic lethality phenotype``; gene-keyed genotype; perturbation nodes; no memory_reduction_factor keys).
    * ``SynthRescueYeastSynthLethDbAdapter`` loads ``conf/synth_rescue_yeast_synth_leth_db_adapter.yaml``: 15 node and 13 edge methods (``synthetic rescue phenotype``; gene-keyed genotype; perturbation nodes; no memory_reduction_factor keys).

    The adapter is checked against the dataset ``dataset_adapter_map`` pairs it with
    (asserted to be the class above). The conf lists its methods in the order the
    adapter runs them, no edge dangles, every chunked entity node is linked, and the
    phenotype method matches that dataset's ``experiment_class``. The adapter keeps the dataset and the worker / chunk sizes
    it was given (3, 2, 500, 50), calls ``wandb.init`` once and logs the method table
    (event number, name, node/edge, factor or NaN for a non-chunked method) then the
    dataset name and the pinned start time; it prints the config once.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)


def test_rescue_adapter_is_annotated_with_the_lethality_dataset() -> None:
    """Finding: the rescue adapter's ``dataset`` parameter names the wrong class.

    synth_leth_db_adapter.py:73 annotates ``SynthRescueYeastSynthLethDbAdapter``'s
    ``dataset`` as ``SynthLethalityYeastSynthLethDbDataset``, whose experiment phenotype
    is ``SyntheticLethalityPhenotype``, while its conf enables the
    ``synthetic rescue phenotype`` methods that ``SynthRescueYeastSynthLethDbDataset``
    (phenotype ``SyntheticRescuePhenotype``) needs. Type checking would therefore accept
    the lethality dataset here and reject the right one. Pinned until the annotation is
    ``SynthRescueYeastSynthLethDbDataset``.
    """
    from tests.torchcell.adapters._adapter_init_harness import (
        dataset_phenotype_class,
        hinted_dataset_class,
    )
    from torchcell.datamodels.schema import (
        SyntheticLethalityPhenotype,
        SyntheticRescuePhenotype,
    )

    hinted = hinted_dataset_class(SynthRescueYeastSynthLethDbAdapter)
    assert hinted is SynthLethalityYeastSynthLethDbDataset
    assert dataset_phenotype_class(hinted) is SyntheticLethalityPhenotype
    assert dataset_phenotype_class(SynthRescueYeastSynthLethDbDataset) is (
        SyntheticRescuePhenotype
    )
    assert hinted_dataset_class(SynthLethalityYeastSynthLethDbAdapter) is (
        SynthLethalityYeastSynthLethDbDataset
    )
