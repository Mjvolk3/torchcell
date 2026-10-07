# torchcell/artifacts/walk.py
# [[torchcell.artifacts.walk]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/walk.py
# Test file: tests/torchcell/artifacts/test_artifact_walk.py

"""Find every ``ArtifactRef`` inside a pydantic model tree. Pure: no I/O.

The walk descends declared fields, extra fields (``model_extra``), nested models, lists,
tuples, sets and dict values. A ref is yielded and not descended into. Computed fields
are not walked: they are derived from the declared ones, which are.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel

from torchcell.artifacts.ref import ArtifactRef

#: ``(tier, key, path, sha256)``: what one resolvability check answers for. Refs that
#: differ only in ``member``, ``bytes`` or ``media_type`` name the same file.
RefKey = tuple[str, str, str, str]


def ref_key(ref: ArtifactRef) -> RefKey:
    """``(tier, key, path, sha256)`` of ``ref``."""
    return (ref.tier, ref.key, ref.path, ref.sha256)


def _iter_value(value: Any) -> Iterator[ArtifactRef]:
    """Yield the refs inside one field value."""
    if isinstance(value, ArtifactRef):
        yield value
    elif isinstance(value, BaseModel):
        yield from iter_refs(value)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _iter_value(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_value(item)


def iter_refs(model: BaseModel) -> Iterator[ArtifactRef]:
    """Yield every ``ArtifactRef`` in ``model``'s tree, in field order, with repeats.

    ``model`` itself is yielded when it is a ref.
    """
    if isinstance(model, ArtifactRef):
        yield model
        return
    for name in type(model).model_fields:
        yield from _iter_value(getattr(model, name))
    extra = model.model_extra
    if extra:
        for value in extra.values():
            yield from _iter_value(value)


def distinct_refs(models: Iterable[BaseModel]) -> dict[RefKey, ArtifactRef]:
    """The refs in ``models`` keyed by ``ref_key``; the first ref met per key is kept."""
    found: dict[RefKey, ArtifactRef] = {}
    for model in models:
        for ref in iter_refs(model):
            found.setdefault(ref_key(ref), ref)
    return found
