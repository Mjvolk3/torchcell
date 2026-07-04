# torchcell/literature/provenance.py
# [[torchcell.literature.provenance]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/provenance.py
# Test file: tests/torchcell/literature/test_provenance.py
"""Retrieval + processing provenance behavior (run / verify / check).

Answers, for any artifact, "where did this come from and how was it done?" from
our own records -- the documentation is authoritative; we never re-chase authors.

The provenance RECORD types (``RetrievalMethod``, ``SourceCheck``,
``RetrievalRecord``, ``ProcessingRecord``, ``ArtifactRecord``) live in
``manifest``: they are pure pydantic data with no I/O dependency, so they sit at
the bottom of the import graph and ``Manifest.files`` is typed
``list[ArtifactRecord]`` without a circular import. This module holds the
BEHAVIOR that acts on those records and therefore depends on the network
retrievers in ``retrieve``. The records are re-exported here for convenience.

Two principles:
- The stored artifact + its ``sha256`` is canonical; the URL is historical
  retrieval metadata, not a live dependency. On rebuild we run the recorded
  retriever, then verify sha256 -- upstream drift (mismatch) or rot (failure) is
  DETECTED and falls back to the mirror, never silently followed.
- Retrieval/processing reference VERSIONED SOURCE CODE (a registry key + params),
  not a free command string -- so the recipe is testable and re-runnable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from torchcell.literature.manifest import (
    ArtifactRecord,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
    sha256_file,
)
from torchcell.literature.retrieve import RETRIEVERS

__all__ = [
    "ArtifactRecord",
    "ProcessingRecord",
    "RetrievalMethod",
    "RetrievalRecord",
    "SourceCheck",
    "run_retriever",
    "verify_artifact",
    "check_source",
]


def run_retriever(
    record: RetrievalRecord, *, registry: dict[str, Callable[..., bytes]] | None = None
) -> bytes:
    """Resolve the record's retriever via the registry key and call it with params."""
    reg = RETRIEVERS if registry is None else registry
    fn = reg.get(record.retriever)
    if fn is None:
        raise KeyError(f"unknown retriever: {record.retriever!r}")
    return fn(**record.params)


def verify_artifact(record: ArtifactRecord, artifact_dir: str | Path) -> bool:
    """True if the stored file's sha256 matches the record (mirror is intact)."""
    return sha256_file(Path(artifact_dir) / record.path) == record.sha256


def check_source(
    record: RetrievalRecord,
    *,
    now: str,
    registry: dict[str, Callable[..., bytes]] | None = None,
) -> SourceCheck:
    """Re-run the retriever and report whether it still yields our recorded bytes.

    ``matches=False`` means upstream drifted (a new version); it does NOT overwrite
    our canonical artifact -- the caller decides whether to version a new record.
    """
    produced = hashlib.sha256(run_retriever(record, registry=registry)).hexdigest()
    return SourceCheck(
        checked_at=now, produced_sha256=produced, matches=(produced == record.sha256)
    )
