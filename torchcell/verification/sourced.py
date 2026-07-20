# torchcell/verification/sourced
# [[torchcell.verification.sourced]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/sourced
"""Provenance-bound constants: bind ONE extracted value to its source (WS2/WS3).

A ``SourcedValue`` pins a hardcoded number (e.g. ``n_samples = 2``) to the exact
paper artifact it came from -- ``citation_key`` + a path inside the MinerU
``torchcell-library`` + the file ``sha256`` + a verbatim ``quote``. Reading
``.value`` needs nothing but Python (the ML env never touches MinerU or the
library). An OPTIONAL audit (``audit_sourced_value``) re-opens the source, checks
the hash still matches, and confirms the quote is still present -- run on demand,
never at dataset load.

Design notes:
- **No line numbers.** OCR output reflows when the extractor updates, so a line
  reference silently rots. The file ``sha256`` is the reproducibility anchor: if
  the source is re-OCR'd the hash changes and the audit flags it for re-location;
  the ``quote`` substring is what actually locates the value.
- This layers on ``torchcell.literature.manifest.Manifest`` (which already stores
  the per-file ``sha256``) once that subsystem is in scope. Until then it is
  self-contained.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.verification.report import Level, LevelResult, Provenance, sha256_file


# NOTE: intentionally NON-generic. A parametrized pydantic generic (``SourcedValue
# [Any]``) has a bracketed ``__qualname__`` that ``pickle`` cannot resolve by name,
# so any object embedding one (a Media provenance list inside a dataset's reference
# index) fails to pickle across a ``multiprocessing.Queue`` / ``ProcessPoolExecutor``
# -- surfacing as a silent adapter hang. ``value: Any`` matches every real usage
# (all were ``SourcedValue[Any]``) with zero loss of typing.
class SourcedValue(BaseModel):
    """A hardcoded value plus the provenance that justifies it.

    ``provenance.source_uri`` is interpreted as the path RELATIVE to
    ``<library_root>/<citation_key>/`` (e.g. ``"si/si1.md"``). ``citation_key``
    and ``sha256`` are required so the value is always auditable.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any
    provenance: Provenance
    quote: str = Field(
        description="Verbatim substring from the source that justifies the value."
    )
    note: str | None = Field(
        default=None, description="Optional reasoning linking the quote to the value."
    )

    @model_validator(mode="after")
    def _require_auditable_provenance(self) -> SourcedValue:
        if not self.provenance.citation_key:
            raise ValueError("SourcedValue.provenance.citation_key is required")
        if not self.provenance.sha256:
            raise ValueError(
                "SourcedValue.provenance.sha256 is required (audit anchor)"
            )
        if not self.quote.strip():
            raise ValueError("SourcedValue.quote cannot be empty")
        return self

    def source_path(self, library_root: str | Path) -> Path:
        """Resolve the artifact path within the MinerU library."""
        return (
            Path(library_root)
            / str(self.provenance.citation_key)
            / self.provenance.source_uri
        )


def audit_sourced_value(sv: SourcedValue, library_root: str | Path) -> LevelResult:
    """Verify a ``SourcedValue`` against its source artifact.

    Checks (1) the file hash still equals the pinned ``sha256`` and (2) the quote
    is still present in the file. Raises ``FileNotFoundError`` when the artifact
    is not on disk -- callers skip when the library isn't mounted (see
    ``library_available``), so CI without the library stays green.
    """
    path = sv.source_path(library_root)
    if not path.exists():
        raise FileNotFoundError(f"source artifact not found: {path}")

    actual_sha = sha256_file(path)
    integrity = actual_sha == sv.provenance.sha256

    quote_present = False
    if integrity:
        # Only trust the text if the bytes are what we pinned.
        quote_present = sv.quote in path.read_text(encoding="utf-8", errors="replace")

    passed = integrity and quote_present
    if not integrity:
        message = f"sha256 drift: source re-OCR'd or edited ({path.name})"
    elif not quote_present:
        message = f"quote no longer found in {path.name}"
    else:
        message = f"value backed by verbatim quote in {path.name}"

    return LevelResult(
        level=Level.L3,
        name="provenance_audit",
        passed=passed,
        message=message,
        details={
            "citation_key": sv.provenance.citation_key,
            "source": sv.provenance.source_uri,
            "value": repr(sv.value),
            "sha256_ok": integrity,
            "quote_present": quote_present,
        },
    )


def library_available(library_root: str | Path) -> bool:
    """True if the MinerU library root exists (audits skip when it doesn't)."""
    return Path(library_root).is_dir()


class ProvenanceGapReason(StrEnum):
    """WHERE in the provenance chain a value died -- an honest, typed absence.

    Ordered by how actionable the gap is. Only ``deferred_pending_source_review``
    is recoverable work (comb the primary paper's SI); the other two are terminal
    (there is nothing to fetch). This split is what turns the gap-set into a
    worklist rather than a pile of nulls.
    """

    not_reported_by_primary = (
        "not_reported_by_primary"  # original screen never measured/reported it
    )
    not_carried_by_curation = "not_carried_by_curation"  # primary had it; the secondary DB (YeastPhenome/SPELL) dropped it in aggregation
    deferred_pending_source_review = "deferred_pending_source_review"  # recoverable -- the per-paper SI comb just isn't done yet


class ProvenanceGap(BaseModel):
    """The complement of :class:`SourcedValue`: a documented, typed ABSENCE.

    ``SourcedValue`` binds a value to ``(citation_key, sha256, quote)``.
    ``ProvenanceGap`` binds a MISSING value to a typed ``reason``, so an unfilled
    field (``n_samples``, an uncertainty, a replicate design) is an HONEST typed
    absence -- never a guess, never a silent ``None`` the reader cannot tell apart
    from "not applicable". The set of gaps across a build is a queryable worklist
    (see :func:`provenance_gap_census`).

    Consuming a secondary curation layer (YeastPhenome, SPELL) is the motivating
    case: we capture the whole record as presented and gap-mark every field the
    curation does not carry, rather than dropping the record or fabricating a value.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        description="name of the field on THIS record that is unsourced, e.g. 'n_samples'"
    )
    reason: ProvenanceGapReason
    looked_in: Provenance | None = Field(
        default=None,
        description="the secondary source we DID consult (e.g. the YeastPhenome "
        "Zenodo re-host) -- anchors even the absence to where we looked",
    )
    resolve_with: Provenance | None = Field(
        default=None,
        description="deferred_pending_source_review only: the primary artifact "
        "whose SI would close the gap (the worklist target)",
    )
    note: str | None = Field(
        default=None, description="optional human reasoning for the gap."
    )

    @model_validator(mode="after")
    def _require_field(self) -> ProvenanceGap:
        if not self.field.strip():
            raise ValueError("ProvenanceGap.field cannot be empty")
        return self


class ProvenanceGapCensus(BaseModel):
    """Aggregate view of the ``ProvenanceGap``s across a built dataset -- the worklist.

    A documented gap is a PASS, not a failure: this census is informational. Its
    payoff is ``worklist_fields`` -- the distinct fields carrying a
    ``deferred_pending_source_review`` gap, i.e. exactly what a future per-paper SI
    comb (or OCR fan-out) would close.
    """

    model_config = ConfigDict(extra="forbid")

    n_records: int
    n_records_with_gaps: int
    n_gaps: int
    by_reason: dict[str, int] = Field(default_factory=dict)
    by_field: dict[str, int] = Field(default_factory=dict)
    worklist_fields: list[str] = Field(
        default_factory=list,
        description="distinct fields with a deferred_pending_source_review gap",
    )


def provenance_gap_census(
    phenotypes: Iterable[Mapping[str, Any]],
) -> ProvenanceGapCensus:
    """Tally the ``provenance_gaps`` carried by a stream of phenotype records.

    Accepts phenotype MAPPINGS (the JSON dicts a built LMDB yields), reading
    ``phenotype["provenance_gaps"]`` (absent/empty -> no gaps). Single-pass and
    memory-light: keeps only counters + the distinct deferred worklist fields, so
    it scales to a streaming verifier over a large dataset.
    """
    n_records = 0
    n_records_with_gaps = 0
    n_gaps = 0
    by_reason: Counter[str] = Counter()
    by_field: Counter[str] = Counter()
    worklist: set[str] = set()
    for phenotype in phenotypes:
        n_records += 1
        gaps = phenotype.get("provenance_gaps") or []
        if gaps:
            n_records_with_gaps += 1
        for gap in gaps:
            n_gaps += 1
            reason = str(gap["reason"])
            field = str(gap["field"])
            by_reason[reason] += 1
            by_field[field] += 1
            if reason == ProvenanceGapReason.deferred_pending_source_review:
                worklist.add(field)
    return ProvenanceGapCensus(
        n_records=n_records,
        n_records_with_gaps=n_records_with_gaps,
        n_gaps=n_gaps,
        by_reason=dict(by_reason),
        by_field=dict(by_field),
        worklist_fields=sorted(worklist),
    )


def provenance_gap_level_result(census: ProvenanceGapCensus) -> LevelResult:
    """Wrap a census into the L1 ``provenance_gaps`` result -- ALWAYS passes.

    A documented gap is honest, not a defect (the honesty invariant is already
    enforced at L0 by the ``Phenotype`` model validators), so this level never
    fails a build. It surfaces the gap census + the deferred worklist in the report
    so "what still needs sourcing" is visible next to the pass/fail gate. Shared by
    the eager and streaming verifiers so both emit an identical result.
    """
    if census.n_gaps == 0:
        message = (
            f"no provenance gaps across {census.n_records} records (fully sourced)"
        )
    else:
        message = (
            f"{census.n_gaps} documented provenance gaps over "
            f"{census.n_records_with_gaps}/{census.n_records} records; "
            f"{len(census.worklist_fields)} deferred field(s): "
            f"{census.worklist_fields}"
        )
    return LevelResult(
        level=Level.L1,
        name="provenance_gaps",
        passed=True,
        message=message,
        details=census.model_dump(),
    )


def l1_provenance_gaps(phenotypes: Iterable[Mapping[str, Any]]) -> LevelResult:
    """L1 informational census of documented provenance gaps over a phenotype stream."""
    return provenance_gap_level_result(provenance_gap_census(phenotypes))
