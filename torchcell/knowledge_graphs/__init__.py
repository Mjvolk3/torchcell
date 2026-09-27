"""Knowledge-graph builders and dataset-to-adapter mappings.

The two public names are resolved lazily (PEP 562 ``__getattr__``): importing
``create_scerevisiae_kg_small`` pulls in BioCypher, every dataset loader and the
adapters (about ten seconds), and the light submodules such as ``releases`` (the
``make ops`` release table) and ``kg_manifest`` must not pay for it. ``from
torchcell.knowledge_graphs import dataset_adapter_map`` and ``python -m
torchcell.knowledge_graphs.create_scerevisiae_kg_small`` behave as before.
"""

from importlib import import_module
from typing import Any

maps = ["dataset_adapter_map"]

scerevisiae_builds = ["create_scerevisiae_kg_small"]

__all__ = scerevisiae_builds + maps

_LAZY: dict[str, tuple[str, str | None]] = {
    "create_scerevisiae_kg_small": (".create_scerevisiae_kg_small", None),
    "dataset_adapter_map": (".dataset_adapter_map", "dataset_adapter_map"),
}


def __getattr__(name: str) -> Any:
    """Import a public name on first access and cache it on the package."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY[name]
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
