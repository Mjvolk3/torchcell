# torchcell/verification/report
# [[torchcell.verification.report]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/report
"""Provenance + L0-L4 verification report models (roadmap WS3).

These are pure data models describing WHERE a dataset came from (``Provenance``)
and WHETHER it passed the universal L0-L4 record-level gate (``VerificationReport``
of ``LevelResult``). The level-check building blocks that produce ``LevelResult``s
live in ``torchcell.verification.levels``.
"""

from __future__ import annotations

import hashlib
from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Level(IntEnum):
    """The five universal verification levels (ordered: a gate, not a set)."""

    L0 = 0  # structural: record instantiates its schema model
    L1 = 1  # completeness: count / known-keys / contiguity vs an oracle
    L2 = 2  # value fidelity: cross-method agreement, type/range/NaN
    L3 = 3  # semantic: units/conventions (fitness = ko/wt; log2(sample/ref))
    L4 = 4  # cross-source consistency: overlapping entities agree


class Provenance(BaseModel):
    """Where a dataset's records came from, recorded alongside verification."""

    model_config = ConfigDict(extra="forbid")

    source_uri: str = Field(
        description="URL, DOI, or file path of the source artifact."
    )
    citation_key: str | None = Field(
        default=None, description="Zotero/library identity (e.g. authorTitleYear)."
    )
    sha256: str | None = Field(
        default=None, description="Hex sha256 of the source artifact, if hashed."
    )
    method: str | None = Field(
        default=None, description="Extraction method (e.g. 'pdftotext -layout')."
    )
    page: str | None = Field(
        default=None, description="Page/table/sheet reference within the source."
    )
    retrieved: str | None = Field(
        default=None,
        description="ISO date the artifact was retrieved (caller-supplied).",
    )


class LevelResult(BaseModel):
    """The outcome of a single L-level check."""

    model_config = ConfigDict(extra="forbid")

    level: Level
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """Provenance + the L-level results for one dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    provenance: Provenance
    results: list[LevelResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if every recorded level passed (empty report is NOT passed)."""
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def levels_covered(self) -> set[Level]:
        """The set of L-levels that have a recorded result."""
        return {r.level for r in self.results}

    def add(self, result: LevelResult) -> VerificationReport:
        """Append a result and return self (chainable)."""
        self.results.append(result)
        return self

    def summary(self) -> str:
        """One-line-per-level human-readable summary of the report."""
        head = f"{self.dataset_name}: {'PASS' if self.passed else 'FAIL'}"
        lines = [head]
        for r in sorted(self.results, key=lambda x: x.level):
            mark = "ok" if r.passed else "XX"
            lines.append(f"  [{mark}] {r.level.name} {r.name}: {r.message}")
        return "\n".join(lines)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks (large artifacts)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
