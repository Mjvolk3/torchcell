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
