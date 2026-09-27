"""Knowledge-graph builders and dataset-to-adapter mappings.

The two submodules named here are resolved lazily (PEP 562 ``__getattr__``): each of
them imports BioCypher, every dataset loader and every adapter (about seven seconds),
and the light submodules such as ``releases`` (the ``make ops`` release table) and
``kg_manifest`` must not pay for it. ``python -m
torchcell.knowledge_graphs.create_scerevisiae_kg_small`` is unchanged.

Both names refer to the SUBMODULE. The package used to re-export the
``dataset_adapter_map`` dict under the same name as its module; that cannot coexist
with lazy loading, because a direct ``import torchcell.knowledge_graphs.dataset_adapter_map``
binds the submodule onto the package and shadows any re-export. The dict is imported
from its module: ``from torchcell.knowledge_graphs.dataset_adapter_map import
dataset_adapter_map``.
"""

from importlib import import_module
from types import ModuleType

maps = ["dataset_adapter_map"]

scerevisiae_builds = ["create_scerevisiae_kg_small"]

__all__ = scerevisiae_builds + maps


def __getattr__(name: str) -> ModuleType:
    """Import a listed submodule on first access; the import binds it on the package."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module(f".{name}", __name__)
