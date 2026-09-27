# tests/torchcell/knowledge_graphs/test_package_lazy_imports.py
# [[tests.torchcell.knowledge_graphs.test_package_lazy_imports]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_package_lazy_imports.py
"""``torchcell.knowledge_graphs`` resolves its two public names lazily.

Importing the package must not import ``create_scerevisiae_kg_small`` (BioCypher, every
loader, the adapters; the ten-second import that made ``make ops`` slow). The names are
checked in a subprocess so the parent's module cache cannot mask an eager import:
after ``import torchcell.knowledge_graphs`` the heavy module is absent from
``sys.modules``, ``dataset_adapter_map`` resolves to the mapping (the adapters import
is the price of that name and is paid only then), and an unknown attribute raises
``AttributeError`` with the package name. The build module itself is never imported
here: it mutates the SSL environment at import (``NEVER_IMPORT`` in test_import_all).
"""

import json
import subprocess
import sys

import torchcell.knowledge_graphs as kg

PROBE = r"""
import json, sys
import torchcell.knowledge_graphs as kg
before = "torchcell.knowledge_graphs.create_scerevisiae_kg_small" in sys.modules
adapters_before = "torchcell.knowledge_graphs.dataset_adapter_map" in sys.modules
mapping = kg.dataset_adapter_map
adapters_after = "torchcell.knowledge_graphs.dataset_adapter_map" in sys.modules
cached = "dataset_adapter_map" in vars(kg)
try:
    kg.no_such_name
    error = None
except AttributeError as exc:
    error = str(exc)
print(json.dumps({
    "heavy_before": before,
    "heavy_after": "torchcell.knowledge_graphs.create_scerevisiae_kg_small" in sys.modules,
    "adapters_before": adapters_before,
    "adapters_after": adapters_after,
    "cached": cached,
    "mapping_type": type(mapping).__name__,
    "n_datasets": len(mapping),
    "error": error,
}))
"""


def test_package_import_does_not_import_the_build_module() -> None:
    """The heavy module stays out of sys.modules; the mapping loads on first access only."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["heavy_before"] is False
    assert probe["heavy_after"] is False
    assert probe["adapters_before"] is False
    assert probe["adapters_after"] is True
    assert probe["cached"] is True
    assert probe["mapping_type"] == "dict"
    assert probe["n_datasets"] == len(kg.dataset_adapter_map)
    assert probe["error"] == (
        "module 'torchcell.knowledge_graphs' has no attribute 'no_such_name'"
    )


def test_public_names_and_lazy_table_agree() -> None:
    """``__all__`` is exactly the lazily resolved names."""
    assert (
        sorted(kg.__all__)
        == sorted(kg._LAZY)
        == ["create_scerevisiae_kg_small", "dataset_adapter_map"]
    )
