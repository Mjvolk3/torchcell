# torchcell/literature/provenance.py
# [[torchcell.literature.provenance]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/provenance.py
# Test file: tests/torchcell/literature/test_provenance.py
"""Retrieval + processing provenance records (pydantic).

Answers, for any artifact, "where did this come from and how was it done?" from
our own records -- the documentation is authoritative; we never re-chase authors.

Two principles:
- The stored artifact + its ``sha256`` is canonical; the URL is historical
  retrieval metadata, not a live dependency. On rebuild we run the recorded
  retriever, then verify sha256 -- upstream drift (mismatch) or rot (failure) is
  DETECTED and falls back to the mirror, never silently followed.
- Retrieval/processing reference VERSIONED SOURCE CODE (a dotted-path function +
  params), not a free command string -- so the recipe is testable and re-runnable.

These extend ``manifest.FileRecord``; the same records serve papers, supplements,
software, and dataset raw files.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import FileRecord, sha256_file
from torchcell.literature.retrieve import RETRIEVERS


class RetrievalMethod(StrEnum):
    """How an artifact is fetched. ``radiant_endpoint`` is a future slot (issue
    #20) for un-scriptable sources served from the Radiant VM; there is no
    ``manual_browser`` -- manual-once artifacts are deposited then served.
    """

    springer_esm = "springer_esm"
    zotero_attachment = "zotero_attachment"
    pmc_oa_api = "pmc_oa_api"
    direct_url = "direct_url"
    radiant_endpoint = "radiant_endpoint"


class SourceCheck(BaseModel):
    """Result of re-running the retriever to test the source still yields our bytes."""

    model_config = ConfigDict(extra="forbid")

    checked_at: str = Field(description="ISO timestamp of the check (caller-supplied).")
    produced_sha256: str = Field(description="sha256 the retriever produced this run.")
    matches: bool = Field(description="Did produced_sha256 equal the record's sha256?")


class RetrievalRecord(BaseModel):
    """How an artifact was obtained, referencing a versioned retriever function."""

    model_config = ConfigDict(extra="forbid")

    method: RetrievalMethod
    source_url: str | None = Field(
        default=None, description="Canonical URL (historical retrieval metadata)."
    )
    retriever: str = Field(
        description="Dotted path into torchcell.literature.retrieve (the source code)."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="kwargs passed to the retriever function."
    )
    sha256: str = Field(description="sha256 of the retrieved bytes (canonical anchor).")
    retrieved_at: str = Field(description="ISO date the artifact was retrieved.")
    last_check: SourceCheck | None = Field(
        default=None, description="Last time the retriever was re-run + verified."
    )


class ProcessingRecord(BaseModel):
    """How a derived artifact (OCR markdown, extracted data) was produced."""

    model_config = ConfigDict(extra="forbid")

    processor: str = Field(description="Dotted path to the processing function.")
    tool: str = Field(description="mineru | pdftotext | llm-extract | ...")
    version: str = Field(description="Tool/library version used.")
    params: dict[str, Any] = Field(default_factory=dict)
    input_sha256: list[str] = Field(
        default_factory=list, description="sha256 of the inputs this was derived from."
    )


class ArtifactRecord(FileRecord):
    """A ``FileRecord`` plus how it was retrieved and/or processed."""

    retrieval: RetrievalRecord | None = None
    processing: ProcessingRecord | None = None


def run_retriever(
    record: RetrievalRecord, *, registry: dict[str, Callable[..., bytes]] | None = None
) -> bytes:
    """Resolve the record's retriever by dotted path and call it with its params."""
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
