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
  stored value could be checked against a byte of the measuring paper. Owner decision
  16 (2026-10-10, revising 15) adds the decomposition: ``AggregationRecord.sources``
  names each source study, whether it has its own per-record release, and the L4
  cross-check of that release against the compendium's copy (:class:`SourceDecomposition`).
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


# --------------------------------------------------------------------------- #
# Aggregation decomposition (owner decision 2026-10-10, plan decision 16)
# --------------------------------------------------------------------------- #
#: original_study     a paper the compendium attributes samples to, other than itself
#: compendium_paper   the compendium's own paper: its samples were first described there
#: unattributed_release  samples another artifact releases with no study named for them
#:                    (a Fitness Browser experiment with an empty ``pubId``, a BioProject
#:                    deposited with no publication)
SourceRole = Literal["original_study", "compendium_paper", "unattributed_release"]
#: per_record_values  the source's own release carries the per-record values
#: subset_values      it carries values for a subset (some genes, some contrasts)
#: raw_reads_only     only per-sample sequencing reads: values must be recomputed, with
#:                    the compendium's processing as the reference
#: none               the compendium is the only artifact holding these samples
OwnReleaseKind = Literal["per_record_values", "subset_values", "raw_reads_only", "none"]
ReleaseAccess = Literal["scriptable", "walled"]
CrossCheckStatistic = Literal["max_abs_diff", "min_pearson_r", "min_r_squared"]


class SourceCrossCheck(_Frozen):
    """L4: a source's own values against the compendium's copy of the same samples."""

    statistic: CrossCheckStatistic
    value: float
    tolerance: float | None = Field(
        description="the bound the statistic must meet; None when no stated "
        "reprocessing implies one (the value is then a measured pin, not a test)"
    )
    tolerance_basis: str = Field(
        min_length=1,
        description="where the tolerance comes from (the compendium's stated "
        "rounding or replicate filter), or why there is none",
    )
    n_samples: int = Field(ge=1)
    n_values: int = Field(ge=1)
    against: str = Field(
        min_length=1, description="the own-release artifact compared, with its sha256"
    )

    @property
    def passes(self) -> bool | None:
        """Whether the statistic meets the tolerance; None without a tolerance."""
        if self.tolerance is None:
            return None
        if self.statistic == "max_abs_diff":
            return self.value <= self.tolerance
        return self.value >= self.tolerance


class SourceDecomposition(_Frozen):
    """One source study of an aggregation and what its OWN release holds.

    A served compendium is a processing reference and a sanity check, not the source of
    record (owner, 2026-10-10): each original study is loaded from its own per-record
    release when it has one, cross-checked against the compendium's copy, and the
    compendium stays served only for samples whose only artifact is the compendium.
    One entry per (source, served dataset class); a study feeding two classes of one
    compendium has two entries with the same ``source_label``.
    """

    source_label: str = Field(
        min_length=1,
        description="the compendium's attribution value for the study "
        "(doi:/pmid:/bioproject:/a Fitness Browser pubId, or the compendium's name)",
    )
    role: SourceRole
    citation_key: str | None = Field(
        description="the literature- or raw-mirror key holding the study's paper or "
        "release; None when neither mirror holds it"
    )
    doi: str | None
    dataset_class: str = Field(
        min_length=1, description="the served class whose records these samples are"
    )
    n_samples_in_compendium: int = Field(ge=1)
    n_samples_served: int = Field(
        ge=0, description="of those, the samples the served loader keeps"
    )
    own_release: bool
    own_release_kind: OwnReleaseKind
    own_release_url: str | None
    own_release_access: ReleaseAccess | None
    mirrored: bool
    loaded_as_own_dataset: bool = Field(
        description="a registered dataset class serves these samples' values from "
        "the study's own release"
    )
    own_dataset_class: str | None
    cross_check: SourceCrossCheck | None
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SourceDecomposition:
        name = self.source_label
        if self.role == "compendium_paper" and self.own_release:
            raise ValueError(
                f"{name}: the compendium's own paper has no release apart from it"
            )
        if self.own_release != (self.own_release_kind != "none"):
            raise ValueError(f"{name}: own_release disagrees with own_release_kind")
        has_url = self.own_release_url is not None
        has_access = self.own_release_access is not None
        if (has_url, has_access) != (self.own_release, self.own_release):
            raise ValueError(
                f"{name}: own_release_url and own_release_access are required for, "
                "and only for, an own release"
            )
        if self.mirrored != (self.citation_key is not None):
            raise ValueError(f"{name}: mirrored requires, and only with, citation_key")
        if self.loaded_as_own_dataset != (self.own_dataset_class is not None):
            raise ValueError(
                f"{name}: loaded_as_own_dataset requires, and only with, "
                "own_dataset_class"
            )
        if self.n_samples_served > self.n_samples_in_compendium:
            raise ValueError(
                f"{name}: {self.n_samples_served} served of "
                f"{self.n_samples_in_compendium} compendium samples"
            )
        if self.cross_check is not None and not self.own_release:
            raise ValueError(f"{name}: a cross-check needs an own release")
        return self

    @property
    def loadable_from_own_release(self) -> bool:
        """An own release exists, it is not the compendium's, and nothing loads it yet."""
        return (
            self.own_release
            and self.role != "compendium_paper"
            and not self.loaded_as_own_dataset
        )

    @property
    def compendium_only(self) -> bool:
        """The compendium is the only artifact holding these samples."""
        return not self.own_release


class AggregationRecord(_Frozen):
    """What an aggregation passes G1 with: how many studies, named where, copied or not.

    ``n_sources_mirrored`` and ``net_new_vs_served`` are ``None`` when they have not been
    measured; ``None`` is never a stand-in for zero. ``sources`` is the per-source
    decomposition of plan decision 16, ``None`` until it is measured; when present,
    ``n_source_studies`` and ``n_sources_mirrored`` are its counts.
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
    sources: tuple[SourceDecomposition, ...] | None = Field(
        default=None,
        description="one entry per (source study, served class), measured against each "
        "study's own release; None = not decomposed yet",
    )

    @model_validator(mode="after")
    def _counts_bounded(self) -> AggregationRecord:
        for name in ("n_sources_mirrored", "net_new_vs_served"):
            value = getattr(self, name)
            if value is not None and value > self.n_source_studies:
                raise ValueError(
                    f"{name}={value} exceeds n_source_studies={self.n_source_studies}"
                )
        if self.sources is not None:
            _check_sources(self, self.sources)
        return self

    def studies(self) -> dict[str, tuple[SourceDecomposition, ...]]:
        """The decomposition grouped by source label (empty when not decomposed)."""
        grouped: dict[str, list[SourceDecomposition]] = {}
        for source in self.sources or ():
            grouped.setdefault(source.source_label, []).append(source)
        return {label: tuple(entries) for label, entries in grouped.items()}


def _check_sources(
    record: AggregationRecord, sources: tuple[SourceDecomposition, ...]
) -> None:
    """The decomposition is the count: one label per study, consistent across classes."""
    if not sources:
        raise ValueError("an empty decomposition: pass None until it is measured")
    seen: set[tuple[str, str]] = set()
    for s in sources:
        pair = (s.source_label, s.dataset_class)
        if pair in seen:
            raise ValueError(f"{pair} appears twice in the decomposition")
        seen.add(pair)
    studies = record.studies()
    for label, entries in studies.items():
        identity = {(e.role, e.citation_key, e.doi) for e in entries}
        if len(identity) != 1:
            raise ValueError(f"{label}: entries disagree on role, citation key or DOI")
    if record.n_source_studies != len(studies):
        raise ValueError(
            f"n_source_studies={record.n_source_studies} but the decomposition names "
            f"{len(studies)} studies"
        )
    mirrored = sum(1 for entries in studies.values() if entries[0].mirrored)
    if record.n_sources_mirrored is not None and record.n_sources_mirrored != mirrored:
        raise ValueError(
            f"n_sources_mirrored={record.n_sources_mirrored} but {mirrored} "
            "decomposed studies are mirrored"
        )


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


class CompendiumDerivation(_Frozen):
    """The candidate's records are a derived copy taken from a compendium.

    Owner, 2026-10-12: "in all cases where we collect compendium data / aggregated data
    from db or follow up study - the answer is to alwayst try to eventually collect it
    from source." A compendium, a database aggregate or a follow-up study can embed its
    own preprocessing (normalization, filtering, re-keying, merged conditions) and can
    lack ontological representation the primary release carries (per-sample metadata,
    replicate structure, the released statistic). The flag marks the records as that
    derived copy, so the primary releases stay on the collection list. It decides no gate.
    """

    citation_key: str = Field(
        min_length=1, description="citation key of the compendium the records come from"
    )
    note: str = Field(
        min_length=1,
        description="what the compendium's copy may hide relative to the primary release",
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
    derived_from_compendium: CompendiumDerivation | None = Field(
        default=None,
        description="set when the records are a compendium's, a database aggregate's or "
        "a follow-up study's copy rather than the primary release; None = not flagged",
    )
    decided_at: str = Field(description="ISO date the verdict was computed")
    torchcell_commit: str = Field(description="the 40-hex commit the gates read")

    @model_validator(mode="after")
    def _consistent(self) -> CandidateVerdict:
        _check_order_and_stop(self.gates)
        by_gate = {g.gate: g for g in self.gates}
        _check_g1(by_gate["G1"], self.source_kind, self.aggregation)
        _check_record("G2", by_gate["G2"], self.g2)
        _check_record("G3", by_gate["G3"], self.g3)
        _check_g4(by_gate["G4"], self.g4_key, self.g4_value)
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


def _check_g4(
    result: GateResult, key: G4KeyRecord | None, value: G4ValueRecord | None
) -> None:
    """G4's record rule, with the one case the content rule adds (#913).

    A key shared with another module leaves G4 ``unmeasured`` WITH its key record until
    the released values are compared (``gates.g4_result``); that is the only unmeasured
    G4 that carries a record.
    """
    if result.outcome == "unmeasured" and key is not None:
        if not key.overlapping or value is not None:
            raise ValueError(
                "G4 is unmeasured with a key record only while a shared key awaits "
                "its content comparison"
            )
        return
    _check_record("G4", result, key)


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
