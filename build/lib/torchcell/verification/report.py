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
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class DerivationMethod(StrEnum):
    """How a non-sourced statistic's value was fixed (ordered by preference).

    ``back_solve`` and the range fallbacks apply only when the source does NOT
    state the value per-record. Policy (CLAUDE.md "Adding Datasets"): prefer
    ``back_solve`` from another reported statistic; else ``conservative_low``;
    ``median`` only if a central estimate is explicitly wanted over conservatism.
    """

    sourced = "sourced"  # stated explicitly in the source (constant + citation)
    back_solve = "back_solve"  # inferred from another reported statistic
    conservative_low = "conservative_low"  # lower end of a stated range
    median = "median"  # median of a stated range


class StatDerivation(BaseModel):
    """How a derived (non-per-record-sourced) statistic was fixed, as data.

    For values the source does not give per record (e.g. an effective
    ``n_samples`` that only appears as a range): records the method, chosen value,
    the source range, and diagnostics, so the number and its justification travel
    together and stay re-checkable. Intended to be emitted as a post-process
    artifact alongside ``experiment_reference_index.json`` (roadmap WS3), and to
    back the loader constant (e.g. Kuzmin ``N_SAMPLES_COMBINED_MUTANT``).
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        description="Statistic derived, e.g. 'combined_mutant_n_samples'."
    )
    method: DerivationMethod
    value: float = Field(description="The chosen value the loader uses.")
    range_low: float | None = Field(
        default=None, description="Low end of the source range (range methods)."
    )
    range_high: float | None = Field(
        default=None, description="High end of the source range (range methods)."
    )
    statistic: str | None = Field(
        default=None,
        description="For back_solve: the reported statistic matched "
        "(e.g. 'reported interaction p-value median').",
    )
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Supporting numbers (e.g. per-candidate match, spearman).",
    )
    provenance: Provenance | None = Field(
        default=None, description="Source of the range/statistic (citation + sha256)."
    )
    rationale: str = Field(default="", description="Human-readable why.")

    @model_validator(mode="after")
    def _check_method_inputs(self) -> StatDerivation:
        """Enforce that each method carries the inputs that justify it."""
        if self.method in (DerivationMethod.conservative_low, DerivationMethod.median):
            if self.range_low is None or self.range_high is None:
                raise ValueError(f"{self.method} requires range_low and range_high")
            if not (self.range_low <= self.value <= self.range_high):
                raise ValueError("value must lie within [range_low, range_high]")
        if self.method is DerivationMethod.back_solve and self.statistic is None:
            raise ValueError("back_solve requires a `statistic` (what was matched)")
        return self


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
