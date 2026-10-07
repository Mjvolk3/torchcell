# torchcell/artifacts/__init__.py
# [[torchcell.artifacts]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/__init__.py
# Test file: tests/torchcell/artifacts/test_artifact_resolve.py

"""The artifact tier: ``ArtifactRef`` pointers and the one resolver that dereferences them.

Bytes that do not belong in the graph (sequence tarballs, embeddings, per-cell matrices)
live in four sha256-manifested tiers under ``DATA_ROOT`` (``raw``, ``genomes``,
``library``, ``objects``); a graph record points at them with an ``ArtifactRef``, and
``resolve`` / ``materialize`` / ``check`` turn the pointer into verified bytes on disk.
``deposit`` writes the ``objects`` tier; ``iter_refs`` / ``distinct_refs`` find the refs
inside a record.
"""

from torchcell.artifacts.deposit import deposit
from torchcell.artifacts.ref import ArtifactRef
from torchcell.artifacts.resolve import (
    ArtifactIntegrityError,
    ArtifactUnresolvableError,
    RemoteSource,
    ResolvedArtifact,
    TcDataSource,
    check,
    materialize,
    resolve,
)
from torchcell.artifacts.walk import distinct_refs, iter_refs, ref_key

__all__ = [
    "ArtifactIntegrityError",
    "ArtifactRef",
    "ArtifactUnresolvableError",
    "RemoteSource",
    "ResolvedArtifact",
    "TcDataSource",
    "check",
    "deposit",
    "distinct_refs",
    "iter_refs",
    "materialize",
    "ref_key",
    "resolve",
]
