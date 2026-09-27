# tests/torchcell/knowledge_graphs/test_package_lazy_imports.py
# [[tests.torchcell.knowledge_graphs.test_package_lazy_imports]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_package_lazy_imports.py
"""``torchcell.knowledge_graphs`` resolves its two submodules lazily.

Importing the package must not import ``dataset_adapter_map`` or
``create_scerevisiae_kg_small`` (BioCypher, every loader, the adapters; the seven-second
import that made ``make ops`` slow). The lazy path is checked in a subprocess so this
process's module cache cannot mask an eager import: after
``import torchcell.knowledge_graphs`` neither submodule is in ``sys.modules``, attribute
access imports the mapping module (a module whose ``dataset_adapter_map`` is the dict of
loader classes to adapters), the import binds it on the package so a second access is
the same object, and an unknown attribute raises ``AttributeError`` naming the package.
In this process the mapping module has usually been imported directly by another test
already, which is exactly the case that shadows a re-exported dict: the attribute is the
module either way. The build module is never imported here (it mutates the SSL
environment at import; ``NEVER_IMPORT`` in test_import_all).
"""

import json
import subprocess
import sys
import types

import pytest

import torchcell.knowledge_graphs as kg
from torchcell.knowledge_graphs import dataset_adapter_map as mapping_module

PROBE = r"""
import json, sys, types
import torchcell.knowledge_graphs as kg
heavy = ("torchcell.knowledge_graphs.create_scerevisiae_kg_small",
         "torchcell.knowledge_graphs.dataset_adapter_map")
before = [m in sys.modules for m in heavy]
module = kg.dataset_adapter_map
after = [m in sys.modules for m in heavy]
try:
    kg.no_such_name
    error = None
except AttributeError as exc:
    error = str(exc)
print(json.dumps({
    "before": before,
    "after": after,
    "is_module": isinstance(module, types.ModuleType),
    "module_name": module.__name__,
    "same_object": kg.dataset_adapter_map is module,
    "bound_in_dict": "dataset_adapter_map" in vars(kg),
    "mapping_type": type(module.dataset_adapter_map).__name__,
    "n_datasets": len(module.dataset_adapter_map),
    "error": error,
}))
"""


def test_package_import_is_light_and_attribute_access_imports_the_submodule() -> None:
    """Neither heavy module loads with the package; the first access loads the mapping only."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["before"] == [False, False]
    assert probe["after"] == [False, True]
    assert probe["is_module"] is True
    assert probe["module_name"] == "torchcell.knowledge_graphs.dataset_adapter_map"
    assert probe["same_object"] is True
    assert probe["bound_in_dict"] is True
    assert probe["mapping_type"] == "dict"
    assert probe["n_datasets"] == len(mapping_module.dataset_adapter_map)
    assert probe["error"] == (
        "module 'torchcell.knowledge_graphs' has no attribute 'no_such_name'"
    )


def test_attribute_is_the_submodule_in_this_process_too() -> None:
    """After a direct submodule import the package attribute is that module, not a dict;
    an unlisted name raises with the package name (covered here as well as in the probe,
    since create_kg.py, the other changed file, cannot be imported under the test contract).
    """
    with pytest.raises(
        AttributeError,
        match=r"^module 'torchcell.knowledge_graphs' has no attribute 'nope'$",
    ):
        kg.nope
    assert isinstance(kg.dataset_adapter_map, types.ModuleType)
    assert kg.dataset_adapter_map is mapping_module
    assert isinstance(mapping_module.dataset_adapter_map, dict)
    assert sorted(kg.__all__) == ["create_scerevisiae_kg_small", "dataset_adapter_map"]
    assert kg.maps == ["dataset_adapter_map"]
    assert kg.scerevisiae_builds == ["create_scerevisiae_kg_small"]
