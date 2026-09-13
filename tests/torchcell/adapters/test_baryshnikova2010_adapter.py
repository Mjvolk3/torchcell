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

import yaml

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
