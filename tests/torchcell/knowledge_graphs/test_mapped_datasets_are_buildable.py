# tests/torchcell/knowledge_graphs/test_mapped_datasets_are_buildable.py
# [[tests.torchcell.knowledge_graphs.test_mapped_datasets_are_buildable]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_mapped_datasets_are_buildable.py
"""Every dataset the knowledge-graph build maps can be built by name.

The dev-store build (``torchcell.database.build_dataset_lmdb``) resolves a loader through
the dataset registry and, for a bacterial loader, reads ``REFERENCE_STRAIN`` off the
class. On 2026-10-08 the full-rebuild array refused five mapped datasets for exactly
these two reasons: three Ishii 2007 loaders and Banerjee 2025 carried
``@register_dataset`` but their modules were never imported by their package, and Foo
2014 declared its reference strain as a module constant. These tests close that gap for
every class in the public and private maps, so a loader that cannot be built by name
fails here instead of in a slurm array the night of a build.
"""

from __future__ import annotations

import inspect

from torchcell.database.build_dataset_lmdb import resolve_dataset_class
from torchcell.datasets.bacteria_common import (
    BACTERIAL_GENOME_PARAMETERS,
    declared_reference_strain,
)
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map


def _mapped() -> list[type]:
    return list(build_adapter_map(include_private=True))


def test_every_mapped_dataset_resolves_by_name_to_the_mapped_class() -> None:
    """The registry knows every mapped class under its own name, as the same object."""
    unresolved: list[str] = []
    for cls in _mapped():
        try:
            resolved = resolve_dataset_class(cls.__name__)
        except KeyError:
            unresolved.append(cls.__name__)
            continue
        assert resolved is cls, f"{cls.__name__} resolves to another object"
    assert unresolved == [], f"mapped but not registered: {unresolved}"


def test_every_mapped_bacterial_loader_declares_its_reference_strain_on_the_class() -> (
    None
):
    """A loader naming a bacterial genome parameter carries ``REFERENCE_STRAIN``, and
    the strain it declares is one the genome injector can build.
    """
    missing: list[str] = []
    declared: dict[str, str] = {}
    for cls in _mapped():
        params = inspect.signature(cls.__init__).parameters  # type: ignore[misc]
        if not any(name in params for name in BACTERIAL_GENOME_PARAMETERS):
            continue
        try:
            declared[cls.__name__] = declared_reference_strain(cls)
        except TypeError:
            missing.append(cls.__name__)
    assert missing == [], (
        f"bacterial loaders without a class REFERENCE_STRAIN: {missing}"
    )
    assert declared, "no mapped loader names a bacterial genome parameter"
    assert set(declared.values()) <= {"MG1655", "BW25113", "KT2440", "REL606"}, declared
    assert declared["IsopentenolTiterFoo2014Dataset"] == "MG1655"
