# torchcell/candidates/verdict.py
# [[torchcell.candidates.verdict]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/verdict.py
# Test file: tests/torchcell/candidates/test_verdict.py
"""The candidate verdict: a typed record of the five gates a dataset passes before a loader.

``CandidateVerdict`` is the contract between the deterministic gate evaluators
(:mod:`torchcell.candidates.gates`), the agents of the ``/add-dataset`` pipeline (which
receive :func:`verdict_json_schema` in their prompt and return JSON the CLI validates with
``model_validate_json``), and the tracked store under ``database/candidates/``. It is named
for the CANDIDATE on purpose: ``AdmissionReport`` in
``torchcell/knowledge_graphs/kg_manifest.py`` already means serving admission, which stays
the last stage of the program and is not touched here.

The five gates, in order:

- **G1 primary measurement.** Classifies the source as ``primary``, ``aggregation``,
  ``transcription`` or ``prediction``. A primary source passes; an aggregation passes when
  it carries an :class:`AggregationRecord` (owner decision 2026-10-10: aggregations are
  marked and counted, not refused); a transcription or a prediction fails, because no
  stored value could be checked against a byte of the measuring paper.
- **G2 sequence route.** Every host resolves through the genomes tier and is readable
  (:class:`G2Record`).
- **G3 deposit inventory.** What the mirrors hold, off the bytes (:class:`InventoryRecord`).
- **G4 overlap.** Keys against what is registered (:class:`G4KeyRecord`), values against a
  dev store (:class:`G4ValueRecord`).
- **G5 schema fit.** A phenotype class exists and every compound resolves (:class:`G5Record`).

Ordering and the stop rule: after a ``fail`` or a ``blocked`` every later gate is
``unmeasured``; a ``gap`` (which names its issue) does not stop the run. The verdict's
:attr:`CandidateVerdict.outcome` is derived from the gates, never stored, so a stored
verdict cannot disagree with its own gates.

Pinned to pydantic 2.12.3 semantics: nothing here depends on ``PydanticUserError`` being a
``RuntimeError`` or on discriminated-union serialization falling back (both change in
2.13.0).
"""

from __future__ import annotations

import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.verification.sourced import SourcedValue

GateName = Literal["G1", "G2", "G3", "G4", "G5"]
GATE_ORDER: Final[tuple[GateName, ...]] = ("G1", "G2", "G3", "G4", "G5")

GateOutcome = Literal["pass", "fail", "blocked", "gap", "unmeasured"]

#: Why a gate is blocked rather than failed: the row is not refused, a named prerequisite
#: stands between it and a loader.
#:   wall         the release sits behind a retrieval nothing can script (manual_browser)
#:   provision    a named host has no assembly set in the genomes tier yet
#:   ingest       the host's assembly set is in the tier but no genome class reads it
#:   unfiled_gap  a schema gap was found and no issue names it yet
#:   deposit      a scripted file the manifest lists is not on disk: re-run its retrieval
BlockedKind = Literal["wall", "provision", "ingest", "unfiled_gap", "deposit"]

SourceKind = Literal["primary", "aggregation", "transcription", "prediction"]
ValueOrigin = Literal["re_measured", "derived_by_aggregator"]

G2Branch = Literal["assembly_set", "haplotype_mosaic"]
G2Outcome = Literal[
    "resolvable_readable",
    "resolvable_not_ingestible",
    "absent_provisionable",
    "absent_unnamed",
]

#: The gate outcome each G2 outcome implies (decision 5 of the plan).
G2_GATE: Final[dict[str, tuple[GateOutcome, BlockedKind | None]]] = {
    "resolvable_readable": ("pass", None),
    "resolvable_not_ingestible": ("blocked", "ingest"),
    "absent_provisionable": ("blocked", "provision"),
    "absent_unnamed": ("fail", None),
}

CandidateTable = Literal["bacteria", "yeast"]
ZoteroItem = Literal["present", "absent"]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


class _Frozen(BaseModel):
    """Frozen, closed records: a verdict is evidence, never edited in place."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class GateResult(_Frozen):
    """One gate's outcome, the reason in words, and the quoted evidence behind it."""

    gate: GateName
    outcome: GateOutcome
    reason: str = Field(min_length=1)
    evidence: tuple[SourcedValue, ...] = Field(
        default=(),
        description="file + sha256 + verbatim quote for every judgment read off a source; "
        "empty for a gate decided from the registry or the mirror manifests alone",
    )
    issue: int | None = Field(
        default=None, description="the GitHub issue; required when outcome == 'gap'"
    )
    blocked_kind: BlockedKind | None = Field(
        default=None, description="required when, and only when, outcome == 'blocked'"
    )

    @model_validator(mode="after")
    def _issue_and_kind(self) -> GateResult:
        if self.outcome == "gap" and self.issue is None:
            raise ValueError(f"{self.gate}: a 'gap' must name its issue")
        if (self.outcome == "blocked") != (self.blocked_kind is not None):
            raise ValueError(
                f"{self.gate}: blocked_kind is required for, and only for, 'blocked'"
            )
        return self


class AggregationRecord(_Frozen):
    """What an aggregation passes G1 with: how many studies, named where, copied or not.

    ``n_sources_mirrored`` and ``net_new_vs_served`` are ``None`` when they have not been
    measured; ``None`` is never a stand-in for zero.
    """

    n_source_studies: int = Field(ge=1)
    attribution_field: str = Field(
        min_length=1,
        description="the per-record field (or per-file unit) that names the source study",
    )
    value_origin: ValueOrigin = Field(
        description="re_measured: the aggregator recomputed from raw data; "
        "derived_by_aggregator: it copied or rescaled published numbers"
    )
    n_sources_mirrored: int | None = Field(
        ge=0,
        description="source studies whose own paper or release is in a mirror; None = "
        "not measured",
    )
    net_new_vs_served: int | None = Field(
        ge=0,
        description="source studies whose DOI is not the primary DOI of any dataset "
        "module; None = not measured",
    )
    evidence: tuple[SourcedValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _counts_bounded(self) -> AggregationRecord:
        for name in ("n_sources_mirrored", "net_new_vs_served"):
            value = getattr(self, name)
            if value is not None and value > self.n_source_studies:
                raise ValueError(
                    f"{name}={value} exceeds n_source_studies={self.n_source_studies}"
                )
        return self


class G2Host(_Frozen):
    """One host strain and what the genomes tier says about it."""

    name: str
    assembly_set: str | None = Field(
        description="the tier's assembly-set id; None when no set is deposited for it"
    )
    resolvable: bool = Field(
        description="every manifest member resolves sha256-verified"
    )
    readable: bool = Field(
        description="a genome class of the schema vocabulary reads it"
    )

    @model_validator(mode="after")
    def _readable_implies_resolvable(self) -> G2Host:
        if self.readable and not self.resolvable:
            raise ValueError(f"{self.name}: readable but not resolvable")
        if self.resolvable and self.assembly_set is None:
            raise ValueError(f"{self.name}: resolvable without an assembly set")
        return self


class G2Record(_Frozen):
    """G2 in full: the branch, the hosts (both parents for a mosaic), the outcome."""

    branch: G2Branch
    outcome: G2Outcome
    hosts: tuple[G2Host, ...]

    @model_validator(mode="after")
    def _shape(self) -> G2Record:
        if self.outcome == "absent_unnamed":
            return self
        if self.branch == "haplotype_mosaic" and len(self.hosts) != 2:
            raise ValueError("a haplotype mosaic names exactly two parents")
        if not self.hosts:
            raise ValueError(f"outcome {self.outcome!r} needs at least one host")
        expected = _g2_outcome(self.hosts)
        if expected != self.outcome:
            raise ValueError(f"hosts imply {expected!r}, record says {self.outcome!r}")
        return self


def _g2_outcome(hosts: tuple[G2Host, ...]) -> G2Outcome:
    """The worst host decides: absent before not-ingestible before readable."""
    if any(h.assembly_set is None or not h.resolvable for h in hosts):
        return "absent_provisionable"
    if any(not h.readable for h in hosts):
        return "resolvable_not_ingestible"
    return "resolvable_readable"


class SheetCount(_Frozen):
    """Rows of one worksheet, counted with openpyxl in read_only mode."""

    sheet: str
    rows: int = Field(ge=0)


class InventoryFile(_Frozen):
    """One manifest entry under ``si/`` or ``data/`` and what its bytes say."""

    tree: Literal["library", "raw"]
    path: str
    role: str
    bytes: int = Field(ge=0)
    sha256_manifest: str
    sha256_disk: str | None = Field(description="None when the file is not on disk")
    retrieval_method: str | None
    is_zip: bool
    zip_ok: bool | None = Field(description="zipfile.testzip() found no bad member")
    leading_archive_sha256: str | None = Field(
        description="sha256 of the bytes up to the first end-of-central-directory record, "
        "set only when that is shorter than the file (a concatenated archive)"
    )
    sheets: tuple[SheetCount, ...] = ()

    @property
    def intact(self) -> bool:
        """On disk, hash as pinned, and (for a zip) every member reads."""
        return self.sha256_disk == self.sha256_manifest and self.zip_ok is not False


class InventoryRecord(_Frozen):
    """G3: the deposit, walked separately in the literature and the raw mirror (#495)."""

    citation_key: str
    trees_read: tuple[Literal["library", "raw"], ...]
    files: tuple[InventoryFile, ...]
    walls: tuple[str, ...] = Field(
        default=(),
        description="the manual_browser recipe of every manifest entry not on disk",
    )


G4KeyKind = Literal["doi", "accession", "pmid"]
G4Relation = Literal["own", "primary_of_other", "mentioned", "aggregator_source"]


class G4KeyHit(_Frozen):
    """One key of the candidate found in a dataset module or an aggregator's sources."""

    key_kind: G4KeyKind
    key: str
    where: str = Field(description="the module path, or the aggregator name")
    relation: G4Relation = Field(
        description="own: the module whose primary DOI it is the candidate's; "
        "primary_of_other: another module's primary DOI or a shared accession; "
        "mentioned: cited in another module's text; aggregator_source: a PMID an "
        "aggregator re-serves"
    )


class G4KeyRecord(_Frozen):
    """G4-key: the candidate's identifiers and every place they were found."""

    doi: str | None
    pmid: str | None
    accessions: tuple[str, ...]
    hits: tuple[G4KeyHit, ...]
    pmid_checked: bool = Field(
        description="False when no PMID resolved offline, so no aggregator was consulted"
    )

    @property
    def overlapping(self) -> tuple[G4KeyHit, ...]:
        """The hits that mean the candidate is already held under another name."""
        return tuple(
            h
            for h in self.hits
            if h.relation in ("primary_of_other", "aggregator_source")
        )


class G4ValueMatch(_Frozen):
    """One released column against every served sample, best and runner-up kept."""

    released_column: str
    best_sample: str
    best_max_abs_diff: float = Field(ge=0)
    runner_up_sample: str | None
    runner_up_max_abs_diff: float | None
    keys_compared: int = Field(ge=1)

    @property
    def margin(self) -> float | None:
        """How much worse the nearest wrong sample is (Thompson: 0.046 vs >= 0.58)."""
        if self.runner_up_max_abs_diff is None:
            return None
        return self.runner_up_max_abs_diff - self.best_max_abs_diff


class G4ValueRecord(_Frozen):
    """G4-value: a released table against a dev store's records, read-only."""

    store: str = Field(description="the dev store directory that was read")
    store_commit: str | None = Field(
        description="torchcell_commit of its build manifest"
    )
    records_read: int = Field(ge=0)
    tolerance: float = Field(gt=0)
    matches: tuple[G4ValueMatch, ...]

    @property
    def subsumed(self) -> bool:
        """Every released column has a served sample within tolerance."""
        return bool(self.matches) and all(
            m.best_max_abs_diff <= self.tolerance for m in self.matches
        )


class G5Record(_Frozen):
    """G5: the phenotype class the released columns need, and the compounds."""

    phenotype_class: str
    class_exists: bool
    unresolved_compounds: tuple[str, ...] = ()
    proposed_class: str | None = Field(
        default=None, description="the agent's proposed name when the class is missing"
    )


class CandidateVerdict(_Frozen):
    """The five gates for one candidate row, as recorded in ``database/candidates/``."""

    citation_key: str = Field(min_length=1)
    table: CandidateTable
    row_name: str = Field(min_length=1)
    gates: tuple[GateResult, ...]
    zotero_item: ZoteroItem
    source_kind: SourceKind | None = Field(
        description="None only while G1 is unmeasured (a borderline row awaiting the "
        "agentic judgment)"
    )
    aggregation: AggregationRecord | None
    g2: G2Record | None
    g3: InventoryRecord | None
    g4_key: G4KeyRecord | None
    g4_value: G4ValueRecord | None
    g5: G5Record | None
    decided_at: str = Field(description="ISO date the verdict was computed")
    torchcell_commit: str = Field(description="the 40-hex commit the gates read")

    @model_validator(mode="after")
    def _consistent(self) -> CandidateVerdict:
        _check_order_and_stop(self.gates)
        by_gate = {g.gate: g for g in self.gates}
        _check_g1(by_gate["G1"], self.source_kind, self.aggregation)
        _check_record("G2", by_gate["G2"], self.g2)
        _check_record("G3", by_gate["G3"], self.g3)
        _check_record("G4", by_gate["G4"], self.g4_key)
        _check_record("G5", by_gate["G5"], self.g5)
        if self.g4_value is not None and self.g4_key is None:
            raise ValueError("g4_value without g4_key: G4-value runs after G4-key")
        if self.g2 is not None:
            outcome, kind = G2_GATE[self.g2.outcome]
            if (by_gate["G2"].outcome, by_gate["G2"].blocked_kind) != (outcome, kind):
                raise ValueError(
                    f"G2 record {self.g2.outcome!r} implies {outcome!r}, gate says "
                    f"{by_gate['G2'].outcome!r}"
                )
        if not _COMMIT.match(self.torchcell_commit):
            raise ValueError("torchcell_commit must be a 40-hex commit")
        if not _DATE.match(self.decided_at):
            raise ValueError("decided_at must start with an ISO date (YYYY-MM-DD)")
        return self

    def gate(self, name: GateName) -> GateResult:
        """The result of one gate."""
        return next(g for g in self.gates if g.gate == name)

    @property
    def outcome(self) -> str:
        """Derived: refused, blocked:<kind>, pending, admissible_with_gaps, admissible."""
        return derive_outcome(self.gates)

    @property
    def passing(self) -> bool:
        """Admissible, with or without gaps: what the enforcement test requires."""
        return self.outcome in ("admissible", "admissible_with_gaps")


def _check_order_and_stop(gates: tuple[GateResult, ...]) -> None:
    names = tuple(g.gate for g in gates)
    if names != GATE_ORDER:
        raise ValueError(f"gates must be exactly {GATE_ORDER} in order, got {names}")
    stopped_at: str | None = None
    for g in gates:
        if stopped_at is not None and g.outcome != "unmeasured":
            raise ValueError(
                f"{g.gate} is {g.outcome!r} after {stopped_at} stopped the run; "
                "every later gate must be 'unmeasured'"
            )
        if stopped_at is None and g.outcome in ("fail", "blocked"):
            stopped_at = g.gate


def _check_g1(
    g1: GateResult, kind: SourceKind | None, aggregation: AggregationRecord | None
) -> None:
    allowed: dict[str, tuple[SourceKind | None, ...]] = {
        "pass": ("primary", "aggregation"),
        "fail": ("transcription", "prediction"),
        "unmeasured": (None,),
    }
    if g1.outcome not in allowed:
        raise ValueError(f"G1 is pass, fail or unmeasured, not {g1.outcome!r}")
    if kind not in allowed[g1.outcome]:
        raise ValueError(f"G1 {g1.outcome!r} with source_kind {kind!r}")
    if (kind == "aggregation") != (aggregation is not None):
        raise ValueError(
            "an AggregationRecord is required for, and only for, aggregations"
        )


def _check_record(name: str, result: GateResult, record: Any) -> None:
    if result.outcome == "unmeasured" and record is not None:
        raise ValueError(f"{name} is unmeasured but carries a record")
    if result.outcome != "unmeasured" and record is None:
        raise ValueError(f"{name} is {result.outcome!r} without its record")


def derive_outcome(gates: tuple[GateResult, ...]) -> str:
    """The verdict outcome from its gates (the stop rule is assumed to hold)."""
    for g in gates:
        if g.outcome == "fail":
            return "refused"
        if g.outcome == "blocked":
            return f"blocked:{g.blocked_kind}"
    if any(g.outcome == "unmeasured" for g in gates):
        return "pending"
    if any(g.outcome == "gap" for g in gates):
        return "admissible_with_gaps"
    return "admissible"


def verdict_json_schema() -> dict[str, Any]:
    """The JSON schema an agent prompt carries; its output is validated against it."""
    return CandidateVerdict.model_json_schema()
