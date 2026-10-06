# tests/torchcell/adapters/test_baryshnikova2010_adapter.py
"""The Baryshnikova 2010 adapter conf: a verbatim clone of the Costanzo 2016 SMF pair.

No new ``CellAdapter`` method and no new graph class is introduced, so the tests are
list comparisons against the Costanzo conf and a name check against the adapter's own
method tables. The temperature split is the one shape difference, and it is checked to
produce two distinct temperature nodes through the shared method.
"""

from __future__ import annotations

import ast
import inspect
import os.path as osp
from typing import Any

import pytest
import yaml

import torchcell.adapters.baryshnikova2010_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.baryshnikova2010_adapter import SmfBaryshnikova2010Adapter
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.baryshnikova2010 import SmfBaryshnikova2010Dataset

ADAPTERS_DIR = osp.dirname(osp.abspath(inspect.getsourcefile(CellAdapter) or ""))
ADAPTER_CONF_DIR = osp.join(ADAPTERS_DIR, "conf")
CELL_ADAPTER_PY = osp.join(ADAPTERS_DIR, "cell_adapter.py")


def _conf(name: str) -> dict[str, Any]:
    with open(osp.join(ADAPTER_CONF_DIR, name)) as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)["cell_adapter"]
    return loaded


def _declared_methods() -> dict[str, set[str]]:
    """``node_methods``/``edge_methods`` names, read from the source, not from an instance.

    ``CellAdapter.__init__`` builds the tables, and building one needs a dataset, so the
    names are parsed out of the assignment instead.
    """
    tree = ast.parse(open(CELL_ADAPTER_PY).read())
    found: dict[str, set[str]] = {"node_methods": set(), "edge_methods": set()}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in found
                and isinstance(node.value, ast.List)
            ):
                for element in node.value.elts:
                    if not isinstance(element, ast.Tuple):
                        continue
                    head = element.elts[0]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str):
                        found[target.attr].add(head.value)
    return found


def test_conf_is_the_costanzo_smf_conf_verbatim() -> None:
    mine = _conf("smf_baryshnikova2010_adapter.yaml")
    costanzo = _conf("smf_costanzo2016_adapter.yaml")
    assert mine == costanzo
    assert len(mine["node_methods"]) == 15 and len(mine["edge_methods"]) == 13
    # no environment-perturbation or crispr methods: this dataset has neither
    names = {m["method_name"] for m in mine["node_methods"] + mine["edge_methods"]}
    assert not {n for n in names if "environment perturbation" in n or "crispr" in n}


def test_every_enabled_method_exists_in_cell_adapter() -> None:
    mine = _conf("smf_baryshnikova2010_adapter.yaml")
    declared = _declared_methods()
    assert declared["node_methods"] and declared["edge_methods"]
    for key in ("node_methods", "edge_methods"):
        requested = [m["method_name"] for m in mine[key]]
        assert not [m for m in requested if m not in declared[key]]


def test_adapter_points_at_its_own_conf_file() -> None:
    from torchcell.adapters.baryshnikova2010_adapter import SmfBaryshnikova2010Adapter

    assert osp.exists(osp.join(ADAPTER_CONF_DIR, "smf_baryshnikova2010_adapter.yaml"))
    source = inspect.getsource(SmfBaryshnikova2010Adapter)
    assert '"smf_baryshnikova2010_adapter.yaml"' in source
    assert issubclass(SmfBaryshnikova2010Adapter, CellAdapter)


def test_the_temperature_split_yields_two_environment_nodes() -> None:
    thirty = SmfBaryshnikova2010Dataset.environment("deletion")
    twenty_six = SmfBaryshnikova2010Dataset.environment("ts")
    damp = SmfBaryshnikova2010Dataset.environment("damp")
    assert thirty.model_dump() == damp.model_dump()
    assert thirty.model_dump() != twenty_six.model_dump()
    # one medium across both, so only the temperature node forks
    assert thirty.media.model_dump() == twenty_six.media.model_dump()


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        SmfBaryshnikova2010Adapter,
        "smf_baryshnikova2010_adapter.yaml",
        Shape("fitness phenotype", mrf=None),
        SmfBaryshnikova2010Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``SmfBaryshnikova2010Adapter`` loads ``conf/smf_baryshnikova2010_adapter.yaml``: 15 node and 13 edge methods (``fitness phenotype``; gene-keyed genotype; perturbation nodes; no memory_reduction_factor keys).

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
