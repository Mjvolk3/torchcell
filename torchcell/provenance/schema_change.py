# torchcell/provenance/schema_change.py
# [[torchcell.provenance.schema_change]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/provenance/schema_change.py
# Test file: tests/torchcell/provenance/test_schema_change.py
"""Schema-change records: a change to a fingerprinted schema symbol carries its argument.

The policy is [[torchcell.datamodels.schema-change-policy]]. In short: a served class,
field or enum member is presumed correct, the burden is on the change, and the criterion
is correctness, never convenience for a new source. Additive changes need a correctness
argument; a removal, a rename, a loosened required field, a changed field meaning or a
removed enum member also need the owner's sign-off.

A :class:`SchemaChangeRecord` is the durable form of that argument, one JSON file per
change under ``database/schema_changes/<date>-<slug>.json``. Each changed symbol carries
the contract fingerprint before and after (``torchcell.provenance.schema_deps``), the
structural reasons :mod:`torchcell.provenance.schema_impact` gives for the change, and the
kinds the author claims for it. The claim is checked against the reasons in both
directions: every reason must be covered by a claimed kind (a removed field cannot be
called ``additive_field``), and a kind that needs a structural witness must have one (an
``enum_member_added`` with no gained member is refused).

Three checks use the records:

- **The tripwire** (:func:`branch_problems`, run by ``scripts/check_schema_change_records
  .py`` as a pre-commit hook): the index against the merge base with ``origin/main``.
  Every symbol whose fingerprint moved must be named by exactly one record staged on the
  branch, with matching fingerprints, status and reasons. The one exemption is a NEW
  symbol that no served dataset's closure contains; it is printed so the PR body lists it.
- **The enforcement test** (:func:`enforcement_problems`, run by
  ``tests/torchcell/datamodels/test_schema_change_records.py``): every symbol whose
  fingerprint differs from the grandfathered baseline is carried from the baseline to its
  current fingerprint by a chain of records (``before`` of one equals ``after`` of the
  previous), so a record whose ``after`` no longer matches is stale and fails.
- **Model validation**: sign-off for the sign-off kinds, a verbatim sourced quote for a
  measurement-type decision, a rebuild scope that agrees with the affected loaders.

"Served" means the datasets of the newest committed release snapshot
(``database/releases/<release>.json``, :mod:`torchcell.knowledge_graphs.release_snapshot`),
the git-tracked description of the store that is actually up.

The grandfathered baseline (``database/schema_changes/grandfathered.json``) pins every
surface symbol's fingerprint at the commit the policy landed on, so the tripwire and the
test bite from that commit on and no record is backfilled.

CLI (``python -m torchcell.provenance.schema_change`` or ``schema-change`` once the
package is reinstalled)::

    schema-change template                         # statement skeleton for this branch
    schema-change new --statement statement.json   # compute + write the record
    schema-change new --statement s.json --refresh # rewrite it after the schema moved
    schema-change check                            # the enforcement test, from the CLI
    schema-change list                             # one line per record
    schema-change tripwire                         # the pre-commit hook's check
    schema-change baseline --ref <commit> --write  # (one time) grandfather a commit

Override: ``TORCHCELL_SCHEMA_RECORD_ACK=1`` lets the tripwire pass a commit and skips the
enforcement test, in the style of ``TORCHCELL_SCHEMA_ACK``. It is a separate variable on
purpose: ``TORCHCELL_SCHEMA_ACK=1`` is set on every breaking schema commit to acknowledge
the rebuild, and reusing it would switch this check off exactly when it matters. CI never
sets it, so it lets a commit through but never a landing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from torchcell.knowledge_graphs.kg_manifest import (
    SURFACE_RELPATHS,
    surface_at_ref,
    surface_in_worktree,
)
from torchcell.knowledge_graphs.release_snapshot import load_snapshots
from torchcell.provenance.schema_deps import SchemaSurface, load_surface_from_sources
from torchcell.provenance.schema_impact import (
    ChangeKind,
    SymbolChange,
    _loader_paths,
    diff_surfaces,
    map_impacts,
)
from torchcell.verification.sourced import SourcedValue

__all__ = [
    "ACK_ENV",
    "BASELINE_FILENAME",
    "RECORDS_RELPATH",
    "SIGN_OFF_KINDS",
    "AffectedLoader",
    "GrandfatheredBaseline",
    "OwnerSignOff",
    "RebuildScope",
    "RejectedAlternative",
    "SchemaChangeKind",
    "SchemaChangeRecord",
    "SchemaChangeStatement",
    "SymbolChangeEntry",
    "admissible_kinds",
    "affected_loaders",
    "branch_problems",
    "build_record",
    "enforcement_problems",
    "kind_problems",
    "load_baseline",
    "load_records",
    "main",
    "rebuild_scope_of",
    "served_release",
    "symbol_chain_problems",
    "write_record",
]

REPO: Final = Path(__file__).resolve().parents[2]
RECORDS_RELPATH: Final = "database/schema_changes"
BASELINE_FILENAME: Final = "grandfathered.json"
ACK_ENV: Final = "TORCHCELL_SCHEMA_RECORD_ACK"

SLUG_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
RECORD_NAME_RE: Final = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.json$"
)
_FINGERPRINT_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_GITHUB_PREFIX: Final = "https://github.com/"

AuthorKind = Literal["owner", "agent"]
SymbolStatus = Literal["added", "removed", "modified"]


class SchemaChangeKind(StrEnum):
    """What a change does to one symbol, in the policy's terms.

    The first six need a correctness argument; the last five also need the owner's
    sign-off (:data:`SIGN_OFF_KINDS`).
    """

    additive_leaf = "additive_leaf"  # a new class or binding: leaf, sibling, pair, enum
    additive_field = "additive_field"  # a new optional-by-construction field
    typed_gap = "typed_gap"  # a required field becomes value-or-ProvenanceGap
    enum_member_added = "enum_member_added"  # a vocabulary gains a member
    equivalent_refactor = (
        "equivalent_refactor"  # fingerprint moved, accepted set did not
    )
    contract_tightened = "contract_tightened"  # a new required field, a stricter check
    field_meaning = "field_meaning"  # same name, different unit/statistic/semantics
    field_required_loosened = "field_required_loosened"  # accepts more, no typed gap
    rename = "rename"  # a class, field, member or binding under a new name
    removal = "removal"  # a class, field, member or binding is gone
    enum_member_removed = "enum_member_removed"  # a vocabulary loses a member


K = SchemaChangeKind

#: Kinds that need the owner's sign-off on top of the correctness argument.
SIGN_OFF_KINDS: Final[frozenset[SchemaChangeKind]] = frozenset(
    {
        K.field_meaning,
        K.field_required_loosened,
        K.rename,
        K.removal,
        K.enum_member_removed,
    }
)

_MODIFIED_KINDS: Final[frozenset[SchemaChangeKind]] = frozenset(K) - {K.additive_leaf}

# schema_impact reason text -> the kinds that can describe it. First match wins; a type
# change is classified by :func:`admissible_kinds` before this table is read, because
# whether it is a typed gap depends on the two annotations.
_REASON_RULES: Final[tuple[tuple[re.Pattern[str], frozenset[SchemaChangeKind]], ...]] = (
    (re.compile(r"^(new symbol|new module-level binding)$"), frozenset({K.additive_leaf, K.rename})),
    (re.compile(r"^(symbol removed|module-level binding removed)$"), frozenset({K.removal, K.rename})),
    (re.compile(r"^base classes changed: "), frozenset({K.typed_gap, K.contract_tightened, K.field_meaning, K.field_required_loosened, K.rename})),
    (re.compile(r"^model Config changed$"), frozenset({K.contract_tightened, K.field_meaning, K.field_required_loosened})),
    (re.compile(r"^added required field '"), frozenset({K.contract_tightened, K.field_meaning, K.rename})),
    (re.compile(r"^added optional field '"), frozenset({K.additive_field, K.typed_gap, K.rename})),
    (re.compile(r"^removed field '"), frozenset({K.removal, K.rename})),
    (re.compile(r"^field '[^']+' became required$"), frozenset({K.contract_tightened})),
    (re.compile(r"^field '[^']+' became optional$"), frozenset({K.field_required_loosened})),
    (re.compile(r"^field '[^']+' default changed$"), frozenset({K.contract_tightened, K.field_meaning, K.field_required_loosened})),
    (re.compile(r"^added member/const '"), frozenset({K.enum_member_added, K.rename})),
    (re.compile(r"^removed member/const '"), frozenset({K.enum_member_removed, K.removal, K.rename})),
    (re.compile(r"^member/const '[^']+' value changed$"), frozenset({K.field_meaning, K.rename})),
    (re.compile(r"^validator/serializer changed: "), _MODIFIED_KINDS),
    (re.compile(r"^contract fingerprint changed$"), _MODIFIED_KINDS),
    (re.compile(r"^module-level '[^']+' changed$"), frozenset({K.equivalent_refactor, K.contract_tightened, K.field_meaning, K.field_required_loosened, K.rename})),
    (re.compile(r"^lost members "), frozenset({K.enum_member_removed, K.removal, K.rename})),
    (re.compile(r"^gained members "), frozenset({K.enum_member_added})),
    (re.compile(r"^members reordered$"), frozenset({K.equivalent_refactor})),
)  # fmt: skip

_TYPE_CHANGE_RE: Final = re.compile(
    r"^field '(?P<field>[^']+)' type changed: (?P<old>.+?) -> (?P<new>.+)$"
)

# A kind that claims a structural event needs a reason that shows it. Kinds absent here
# (contract_tightened, field_meaning, field_required_loosened) can be a validator edit
# alone, which no reason distinguishes, so they carry only the coverage check.
_WITNESSES: Final[dict[SchemaChangeKind, re.Pattern[str]]] = {
    K.additive_leaf: re.compile(r"^(new symbol|new module-level binding)$"),
    K.additive_field: re.compile(r"^added optional field '"),
    K.enum_member_added: re.compile(r"^(added member/const '|gained members )"),
    K.enum_member_removed: re.compile(r"^(removed member/const '|lost members )"),
    K.removal: re.compile(
        r"^(symbol removed|module-level binding removed|removed field '"
        r"|removed member/const '|lost members )"
    ),
    K.rename: re.compile(
        r"^(new symbol|new module-level binding|symbol removed"
        r"|module-level binding removed|added (required|optional) field '"
        r"|removed field '|added member/const '|removed member/const '"
        r"|member/const '[^']+' value changed|gained members |lost members "
        r"|field '[^']+' type changed: |base classes changed: "
        r"|module-level '[^']+' changed)"
    ),
    K.equivalent_refactor: re.compile(
        r"^(validator/serializer changed: |module-level '[^']+' changed$"
        r"|members reordered$|contract fingerprint changed$)"
    ),
    K.typed_gap: re.compile(
        r"^(base classes changed: |added optional field '|validator/serializer changed: )"
    ),
}


def _is_gap_widening(old: str, new: str) -> bool:
    """True when ``new`` is ``old`` made nullable (``X -> X | None``), nothing else."""
    old_n = old.replace(" ", "")
    return new.replace(" ", "") in {
        f"{old_n}|None",
        f"None|{old_n}",
        f"Optional[{old_n}]",
    }


def admissible_kinds(reason: str) -> frozenset[SchemaChangeKind]:
    """The kinds that can describe one ``schema_impact`` reason.

    A type change ``X -> X | None`` is admissible as a typed gap; any other type change
    is a meaning change, a tightening, a loosening or a rename. Raises on a reason this
    table does not know, so a new reason in ``schema_impact`` fails loudly here.
    """
    type_change = _TYPE_CHANGE_RE.match(reason)
    if type_change is not None:
        if _is_gap_widening(type_change["old"], type_change["new"]):
            return frozenset({K.typed_gap, K.field_required_loosened, K.field_meaning})
        return frozenset(
            {K.field_meaning, K.contract_tightened, K.field_required_loosened, K.rename}
        )
    for pattern, kinds in _REASON_RULES:
        if pattern.match(reason):
            return kinds
    raise ValueError(
        f"unclassified schema-impact reason {reason!r}: extend _REASON_RULES in "
        "torchcell/provenance/schema_change.py"
    )


def _witnesses(kind: SchemaChangeKind, reason: str) -> bool:
    if kind == K.typed_gap:
        type_change = _TYPE_CHANGE_RE.match(reason)
        if type_change is not None:
            return _is_gap_widening(type_change["old"], type_change["new"])
    pattern = _WITNESSES.get(kind)
    return pattern is None or pattern.match(reason) is not None


def kind_problems(
    kinds: Sequence[SchemaChangeKind], reasons: Sequence[str]
) -> list[str]:
    """Why ``kinds`` does not describe ``reasons``; empty when it does.

    Every reason must be admissible under at least one claimed kind, and every claimed
    kind that names a structural event must be witnessed by at least one reason.
    """
    chosen = set(kinds)
    problems: list[str] = []
    for reason in reasons:
        admissible = admissible_kinds(reason)
        if not chosen & admissible:
            problems.append(
                f"no claimed kind covers {reason!r} "
                f"(admissible: {', '.join(sorted(admissible))})"
            )
    for kind in sorted(chosen):
        if kind in _WITNESSES and not any(_witnesses(kind, r) for r in reasons):
            problems.append(f"kind {kind} has no structural witness among the reasons")
    return problems


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} cannot be blank")


class SymbolChangeEntry(BaseModel):
    """One changed schema symbol: fingerprints, the structural reasons, the claimed kinds.

    ``fingerprint_before`` is ``None`` for an added symbol and ``fingerprint_after`` for a
    removed one. ``impact`` and ``reasons`` are ``schema_impact``'s, verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    status: SymbolStatus
    kinds: list[SchemaChangeKind]
    fingerprint_before: str | None
    fingerprint_after: str | None
    impact: ChangeKind
    reasons: list[str]

    @model_validator(mode="after")
    def _consistent(self) -> SymbolChangeEntry:
        """Status agrees with the fingerprints; the kinds describe the reasons."""
        _require_text(self.symbol, "symbol")
        expected = {
            "added": (False, True),
            "removed": (True, False),
            "modified": (True, True),
        }[self.status]
        present = (
            self.fingerprint_before is not None,
            self.fingerprint_after is not None,
        )
        if present != expected:
            raise ValueError(
                f"{self.symbol}: status {self.status} needs before/after present = "
                f"{expected}, got {present}"
            )
        for fingerprint in (self.fingerprint_before, self.fingerprint_after):
            if fingerprint is not None and not _FINGERPRINT_RE.match(fingerprint):
                raise ValueError(
                    f"{self.symbol}: {fingerprint!r} is not a 64-hex fingerprint"
                )
        if self.fingerprint_before == self.fingerprint_after:
            raise ValueError(f"{self.symbol}: before and after fingerprints are equal")
        if not self.kinds or len(set(self.kinds)) != len(self.kinds):
            raise ValueError(f"{self.symbol}: kinds must be non-empty and unique")
        if not self.reasons:
            raise ValueError(f"{self.symbol}: reasons cannot be empty")
        problems = kind_problems(self.kinds, self.reasons)
        if problems:
            raise ValueError(f"{self.symbol}: " + "; ".join(problems))
        return self


class RejectedAlternative(BaseModel):
    """A design the change did not take, and why."""

    model_config = ConfigDict(extra="forbid")

    alternative: str
    why_rejected: str

    @model_validator(mode="after")
    def _non_blank(self) -> RejectedAlternative:
        """Both halves are stated."""
        _require_text(self.alternative, "alternative")
        _require_text(self.why_rejected, "why_rejected")
        return self


class AffectedLoader(BaseModel):
    """A dataset loader whose closure holds a changed symbol (``schema_impact``).

    ``served_classes`` are the loader's dataset classes in the served release.
    """

    model_config = ConfigDict(extra="forbid")

    loader: str
    dataset_classes: list[str]
    served_classes: list[str]
    impact: ChangeKind
    via: list[str]

    @model_validator(mode="after")
    def _served_subset(self) -> AffectedLoader:
        """A served class is one of the loader's classes."""
        stray = sorted(set(self.served_classes) - set(self.dataset_classes))
        if stray:
            raise ValueError(f"{self.loader}: served {stray} not among its classes")
        if not self.via:
            raise ValueError(f"{self.loader}: via cannot be empty")
        return self


class OwnerSignOff(BaseModel):
    """The owner's approval: their words verbatim, or the URL of the comment holding them."""

    model_config = ConfigDict(extra="forbid")

    signed: bool
    quote: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _evidence(self) -> OwnerSignOff:
        """A signature carries evidence; an absent one carries none."""
        if self.signed and self.quote is None and self.url is None:
            raise ValueError("a sign-off needs the owner's quote or a comment URL")
        if not self.signed and (self.quote is not None or self.url is not None):
            raise ValueError("an unsigned sign-off carries no quote or URL")
        if self.quote is not None:
            _require_text(self.quote, "owner_sign_off.quote")
        if self.url is not None and not self.url.startswith(_GITHUB_PREFIX):
            raise ValueError(f"owner_sign_off.url must start with {_GITHUB_PREFIX}")
        return self


class RebuildScope(StrEnum):
    """What a change forces to be rebuilt.

    ``none``: no loader's closure holds a changed symbol. ``dev_stores``: the affected
    dev stores rebuild and the served graph does not change. ``full_kg_rebuild``: a
    served closure changes, so the change lands with a full KG build and the PR carries
    the ``schema-breaking`` label.
    """

    none = "none"
    dev_stores = "dev_stores"
    full_kg_rebuild = "full_kg_rebuild"


def rebuild_scope_of(loaders: Sequence[AffectedLoader]) -> RebuildScope:
    """Full rebuild when a served class is hit, dev stores when only unserved ones are."""
    if any(loader.served_classes for loader in loaders):
        return RebuildScope.full_kg_rebuild
    if loaders:
        return RebuildScope.dev_stores
    return RebuildScope.none


class SchemaChangeStatement(BaseModel):
    """The human-authored half of a record; ``schema-change new`` computes the rest.

    ``kinds`` maps each symbol this record covers to the kinds claimed for it; a symbol
    left out is not in the record (the tripwire then decides whether it needed one).
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    pr: int | None
    author_kind: AuthorKind
    kinds: dict[str, list[SchemaChangeKind]]
    correctness_argument: str
    measurement_type_decision: bool
    source_quotes: list[SourcedValue]
    alternatives_rejected: list[RejectedAlternative]
    owner_sign_off: OwnerSignOff


class SchemaChangeRecord(BaseModel):
    """One schema change and the argument for it (``database/schema_changes/``).

    Every field is required in the JSON. ``base_commit`` is the merge base the before
    fingerprints were read at; ``python_version`` the interpreter they were computed
    under (fingerprints hash ``ast.unparse`` text, which differs across 3.13 patch
    releases). ``served_release`` is the release snapshot the served flags were read
    against. ``pr`` is ``None`` only until the PR is opened; the enforcement test fails a
    record without one, so a record lands naming its PR.
    """

    model_config = ConfigDict(extra="forbid")

    record_version: Literal[1]
    slug: str
    date: dt.date
    pr: int | None
    author_kind: AuthorKind
    base_commit: str
    python_version: str
    symbols: list[SymbolChangeEntry]
    correctness_argument: str
    measurement_type_decision: bool
    source_quotes: list[SourcedValue]
    alternatives_rejected: list[RejectedAlternative]
    served_release: str
    affected_loaders: list[AffectedLoader]
    served_datasets: list[str]
    rebuild_scope: RebuildScope
    owner_sign_off: OwnerSignOff

    @property
    def filename(self) -> str:
        """``<date>-<slug>.json``."""
        return f"{self.date.isoformat()}-{self.slug}.json"

    @property
    def schema_breaking(self) -> bool:
        """True when a served closure changes: the PR carries ``schema-breaking``."""
        return self.rebuild_scope == RebuildScope.full_kg_rebuild

    @model_validator(mode="after")
    def _consistent(self) -> SchemaChangeRecord:
        """Identity, argument, sign-off and rebuild scope agree with the symbols."""
        if not SLUG_RE.match(self.slug):
            raise ValueError(f"slug {self.slug!r} is not kebab-case [a-z0-9-]")
        if not _COMMIT_RE.match(self.base_commit):
            raise ValueError(f"base_commit {self.base_commit!r} is not a 40-hex sha")
        if self.pr is not None and self.pr < 1:
            raise ValueError(f"pr must be a positive PR number, got {self.pr}")
        names = [entry.symbol for entry in self.symbols]
        if not names:
            raise ValueError("a record names at least one symbol")
        if names != sorted(set(names)):
            raise ValueError("symbols must be unique and sorted by name")
        _require_text(self.correctness_argument, "correctness_argument")
        if self.measurement_type_decision and not self.source_quotes:
            raise ValueError(
                "a measurement-type decision carries at least one verbatim source "
                "quote with its sha256"
            )
        signed_kinds = sorted(
            f"{entry.symbol}:{kind}"
            for entry in self.symbols
            for kind in entry.kinds
            if kind in SIGN_OFF_KINDS
        )
        if signed_kinds and not self.owner_sign_off.signed:
            raise ValueError(
                f"{', '.join(signed_kinds)} need the owner's sign-off "
                "(owner_sign_off.signed with a quote or a comment URL)"
            )
        if (signed_kinds or self.measurement_type_decision) and not (
            self.alternatives_rejected
        ):
            raise ValueError(
                "a sign-off change or a measurement-type decision names at least one "
                "rejected alternative"
            )
        for loader in self.affected_loaders:
            stray = sorted(set(loader.via) - set(names))
            if stray:
                raise ValueError(f"{loader.loader}: via {stray} not in this record")
        served = sorted(
            {name for loader in self.affected_loaders for name in loader.served_classes}
        )
        if self.served_datasets != served:
            raise ValueError(
                f"served_datasets {self.served_datasets} != the affected loaders' "
                f"served classes {served}"
            )
        scope = rebuild_scope_of(self.affected_loaders)
        if self.rebuild_scope != scope:
            raise ValueError(
                f"rebuild_scope {self.rebuild_scope} != {scope} implied by the "
                "affected loaders"
            )
        return self


class GrandfatheredBaseline(BaseModel):
    """Every surface symbol's fingerprint at the commit the policy landed on.

    Written once (``schema-change baseline``). A symbol at its baseline fingerprint
    needs no record; one that moved needs a chain of records from here.
    """

    model_config = ConfigDict(extra="forbid")

    base_commit: str
    python_version: str
    fingerprints: dict[str, str]

    @model_validator(mode="after")
    def _shape(self) -> GrandfatheredBaseline:
        """A real commit and well-formed fingerprints."""
        if not _COMMIT_RE.match(self.base_commit):
            raise ValueError(f"base_commit {self.base_commit!r} is not a 40-hex sha")
        bad = sorted(
            name
            for name, fp in self.fingerprints.items()
            if not _FINGERPRINT_RE.match(fp)
        )
        if bad:
            raise ValueError(f"malformed fingerprints for {bad}")
        return self


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2) + "\n"


def records_dir(repo: Path) -> Path:
    """``<repo>/database/schema_changes``."""
    return repo / RECORDS_RELPATH


def load_baseline(repo: Path) -> GrandfatheredBaseline:
    """The grandfathered baseline committed beside the records."""
    path = records_dir(repo) / BASELINE_FILENAME
    return GrandfatheredBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def _check_name(name: str, record: SchemaChangeRecord) -> None:
    if name != record.filename:
        raise ValueError(
            f"{name}: a record with this date and slug is {record.filename}"
        )


def load_records(repo: Path) -> dict[str, SchemaChangeRecord]:
    """Every record on disk, keyed by file name, in name order.

    Any other JSON file in the directory but the baseline is refused, as is a file whose
    name disagrees with its record or a slug used twice.
    """
    records: dict[str, SchemaChangeRecord] = {}
    for path in sorted(records_dir(repo).glob("*.json")):
        if path.name == BASELINE_FILENAME:
            continue
        if not RECORD_NAME_RE.match(path.name):
            raise ValueError(f"{path.name} is not <YYYY-MM-DD>-<slug>.json")
        record = SchemaChangeRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        _check_name(path.name, record)
        records[path.name] = record
    slugs = [record.slug for record in records.values()]
    repeated = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if repeated:
        raise ValueError(f"slugs used by more than one record: {repeated}")
    return records


def write_record(record: SchemaChangeRecord, repo: Path) -> Path:
    """Write ``record`` to ``database/schema_changes/<date>-<slug>.json``."""
    path = records_dir(repo) / record.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(record), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Git, surfaces, served set
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def merge_base(repo: Path, base: str) -> str:
    """The merge base of ``base`` and ``HEAD``."""
    return _git(repo, "merge-base", base, "HEAD").strip()


def surface_in_index(repo: Path) -> SchemaSurface:
    """The schema surface as staged (``git show :<path>``)."""
    return load_surface_from_sources(
        {relpath: _git(repo, "show", f":{relpath}") for relpath in SURFACE_RELPATHS}
    )


def served_release(repo: Path) -> tuple[str, frozenset[str]]:
    """``(release, dataset classes)`` of the newest committed release snapshot."""
    snapshots = load_snapshots(repo)
    if not snapshots:
        raise FileNotFoundError(f"no release snapshot under {repo}/database/releases")
    newest = snapshots[-1]
    return newest.release, frozenset(newest.datasets)


def affected_loaders(
    changes: Sequence[SymbolChange],
    old: SchemaSurface,
    new: SchemaSurface,
    repo: Path,
    served: frozenset[str],
) -> list[AffectedLoader]:
    """``schema_impact``'s loader map for ``changes``, with each loader's served classes."""
    impacts = map_impacts(list(changes), _loader_paths(repo), old, new, repo)
    return [
        AffectedLoader(
            loader=impact.loader,
            dataset_classes=impact.dataset_classes,
            served_classes=sorted(set(impact.dataset_classes) & served),
            impact=impact.kind,
            via=impact.changed_symbols,
        )
        for impact in impacts
    ]


# --------------------------------------------------------------------------- #
# Building a record
# --------------------------------------------------------------------------- #
class UsageError(Exception):
    """An input the CLI cannot act on (exit 2)."""


def build_record(
    statement: SchemaChangeStatement, *, repo: Path, base: str, record_date: dt.date
) -> tuple[SchemaChangeRecord, list[SymbolChange]]:
    """The record for ``statement``: working tree against the merge base with ``base``.

    Returns the record and the changed symbols it leaves out (for the caller to report).
    Raises :class:`UsageError` when the statement names a symbol that did not change.
    """
    base_commit = merge_base(repo, base)
    old = surface_at_ref(repo, base_commit)
    new = surface_in_worktree(repo)
    changes = {change.symbol: change for change in diff_surfaces(old, new)}
    unchanged = sorted(set(statement.kinds) - set(changes))
    if unchanged:
        raise UsageError(
            f"the statement names {unchanged}, which did not change since {base_commit[:12]}"
        )
    if not statement.kinds:
        raise UsageError("the statement names no symbol")
    covered = [changes[symbol] for symbol in sorted(statement.kinds)]
    release, served = served_release(repo)
    loaders = affected_loaders(covered, old, new, repo, served)
    record = SchemaChangeRecord(
        record_version=1,
        slug=statement.slug,
        date=record_date,
        pr=statement.pr,
        author_kind=statement.author_kind,
        base_commit=base_commit,
        python_version=platform.python_version(),
        symbols=[
            SymbolChangeEntry(
                symbol=change.symbol,
                status=change.status,
                kinds=statement.kinds[change.symbol],
                fingerprint_before=old.fingerprints.get(change.symbol),
                fingerprint_after=new.fingerprints.get(change.symbol),
                impact=change.kind,
                reasons=change.reasons,
            )
            for change in covered
        ],
        correctness_argument=statement.correctness_argument,
        measurement_type_decision=statement.measurement_type_decision,
        source_quotes=statement.source_quotes,
        alternatives_rejected=statement.alternatives_rejected,
        served_release=release,
        affected_loaders=loaders,
        served_datasets=sorted(
            {name for loader in loaders for name in loader.served_classes}
        ),
        rebuild_scope=rebuild_scope_of(loaders),
        owner_sign_off=statement.owner_sign_off,
    )
    left_out = [
        change for name, change in changes.items() if name not in statement.kinds
    ]
    return record, sorted(left_out, key=lambda change: change.symbol)


# --------------------------------------------------------------------------- #
# Enforcement: chains of records from the baseline to the current schema
# --------------------------------------------------------------------------- #
def _short(fingerprint: str | None) -> str:
    return "absent" if fingerprint is None else fingerprint[:12]


def symbol_chain_problems(
    symbol: str,
    entries: Sequence[tuple[str, SymbolChangeEntry]],
    start: str | None,
    end: str | None,
    *,
    anchored: bool,
) -> list[str]:
    """Whether ``entries`` carry ``symbol`` from ``start`` to ``end`` link by link.

    ``entries`` are ``(record name, entry)`` pairs naming ``symbol``. Anchored, the first
    link starts at ``start``; unanchored (a symbol added after the baseline without a
    record, then changed), it starts at the one entry no other entry leads into.
    """
    remaining = list(entries)
    current = start
    if not anchored:
        afters = {entry.fingerprint_after for _, entry in remaining}
        heads = [item for item in remaining if item[1].fingerprint_before not in afters]
        if len(heads) != 1:
            names = sorted(name for name, _ in remaining)
            return [f"{symbol}: records {names} do not form one chain"]
        current = heads[0][1].fingerprint_before
    last = ""
    while remaining:
        following = [
            item for item in remaining if item[1].fingerprint_before == current
        ]
        if not following:
            names = sorted(name for name, _ in remaining)
            return [
                f"{symbol}: no record carries it on from {_short(current)}; "
                f"{names} start elsewhere (stale after a rebase or an edit: rerun "
                "schema-change new --refresh)"
            ]
        if len(following) > 1:
            names = sorted(name for name, _ in following)
            return [
                f"{symbol}: {names} all start from {_short(current)}; refresh the "
                "later one after rebasing (schema-change new --refresh)"
            ]
        last, entry = following[0]
        remaining.remove(following[0])
        current = entry.fingerprint_after
    if current != end:
        return [
            f"{symbol}: {last} leaves it at {_short(current)} but the schema has "
            f"{_short(end)}: the record is stale (schema-change new --refresh)"
        ]
    return []


def enforcement_problems(
    baseline: GrandfatheredBaseline,
    records: Mapping[str, SchemaChangeRecord],
    current: SchemaSurface,
) -> list[str]:
    """Every way the records fail to explain the current schema; empty when they do.

    A symbol that moved from its baseline fingerprint (or was removed) needs records
    chaining it there; a symbol a record names must end at its current fingerprint; a
    record must name its PR and share the baseline's interpreter.
    """
    problems: list[str] = []
    by_symbol: dict[str, list[tuple[str, SymbolChangeEntry]]] = {}
    for name, record in records.items():
        if record.pr is None:
            problems.append(f"{name}: pr is not set (set it once the PR is open)")
        if record.python_version != baseline.python_version:
            problems.append(
                f"{name}: fingerprints computed under Python {record.python_version}, "
                f"the baseline under {baseline.python_version}"
            )
        for entry in record.symbols:
            by_symbol.setdefault(entry.symbol, []).append((name, entry))
    symbols = set(baseline.fingerprints) | set(current.fingerprints) | set(by_symbol)
    for symbol in sorted(symbols):
        start = baseline.fingerprints.get(symbol)
        end = current.fingerprints.get(symbol)
        entries = by_symbol.get(symbol, [])
        if not entries:
            if symbol in baseline.fingerprints and start != end:
                change = "removed" if end is None else "changed"
                problems.append(
                    f"{symbol}: {change} since the grandfathered baseline "
                    f"{baseline.base_commit[:12]} and no record names it"
                )
            continue
        anchored = symbol in baseline.fingerprints or any(
            entry.fingerprint_before is None for _, entry in entries
        )
        problems.extend(
            symbol_chain_problems(symbol, entries, start, end, anchored=anchored)
        )
    return problems


# --------------------------------------------------------------------------- #
# The tripwire: index against the merge base
# --------------------------------------------------------------------------- #
def _staged_record_paths(repo: Path, base_commit: str) -> list[str]:
    out = _git(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "--diff-filter=AM",
        base_commit,
        "--",
        RECORDS_RELPATH,
    )
    return sorted(
        line for line in out.splitlines() if RECORD_NAME_RE.match(Path(line).name)
    )


def branch_problems(repo: Path, base: str) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` for the staged schema against the merge base with ``base``.

    A changed symbol needs exactly one staged record naming it, unless it is new and no
    served dataset's closure holds it (a note then asks for it in the PR body). Each
    staged record is validated, must match the diff symbol by symbol, and must agree
    with the loaders and the served release the diff implies.
    """
    base_commit = merge_base(repo, base)
    old = surface_at_ref(repo, base_commit)
    new = surface_in_index(repo)
    changes = diff_surfaces(old, new)
    problems: list[str] = []
    notes: list[str] = []
    staged: dict[str, SchemaChangeRecord] = {}
    for relpath in _staged_record_paths(repo, base_commit):
        name = Path(relpath).name
        try:
            record = SchemaChangeRecord.model_validate_json(
                _git(repo, "show", f":{relpath}")
            )
        except ValidationError as error:
            problems.append(f"{name}: invalid record: {error}")
            continue
        if name != record.filename:
            problems.append(
                f"{name}: a record with this date and slug is {record.filename}"
            )
            continue
        staged[name] = record
    if not changes and not staged:
        return problems, notes
    release, served = served_release(repo)
    all_loaders = affected_loaders(changes, old, new, repo, served)
    by_change = {change.symbol: change for change in changes}
    naming: dict[str, list[tuple[str, SymbolChangeEntry]]] = {}
    for name, record in staged.items():
        for entry in record.symbols:
            naming.setdefault(entry.symbol, []).append((name, entry))
    for change in changes:
        named = naming.get(change.symbol, [])
        served_hits = sorted(
            {
                cls
                for loader in all_loaders
                if change.symbol in loader.via
                for cls in loader.served_classes
            }
        )
        if not named:
            if change.status == "added" and not served_hits:
                notes.append(
                    f"{change.symbol}: new and in no served closure, so no record is "
                    "required; list it in the PR body"
                )
            else:
                hit = f"; served via {served_hits}" if served_hits else ""
                problems.append(
                    f"{change.symbol} ({change.status}, {change.kind}{hit}) changed since "
                    f"the merge base {base_commit[:12]} and no staged record names it "
                    "(schema-change template / new)"
                )
            continue
        if len(named) > 1:
            problems.append(
                f"{change.symbol}: named by {sorted(n for n, _ in named)}; one record per "
                "symbol per branch"
            )
            continue
        name, entry = named[0]
        expected = (
            change.status,
            old.fingerprints.get(change.symbol),
            new.fingerprints.get(change.symbol),
            change.reasons,
            change.kind,
        )
        recorded = (
            entry.status,
            entry.fingerprint_before,
            entry.fingerprint_after,
            entry.reasons,
            entry.impact,
        )
        if recorded != expected:
            problems.append(
                f"{name}: stale for {change.symbol} (the staged schema differs from what "
                "the record measured; schema-change new --refresh)"
            )
    for symbol, named in sorted(naming.items()):
        if symbol not in by_change:
            for name, _ in named:
                problems.append(
                    f"{name}: names {symbol}, which did not change since the merge base "
                    f"{base_commit[:12]}"
                )
    for name, record in staged.items():
        covered = [by_change[e.symbol] for e in record.symbols if e.symbol in by_change]
        loaders = affected_loaders(covered, old, new, repo, served)
        if (record.served_release, record.affected_loaders) != (release, loaders):
            problems.append(
                f"{name}: affected loaders or served release ({release}) differ from "
                "the staged tree (schema-change new --refresh)"
            )
    return problems, notes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _template(repo: Path, base: str) -> tuple[dict[str, object], list[str]]:
    base_commit = merge_base(repo, base)
    changes = diff_surfaces(
        surface_at_ref(repo, base_commit), surface_in_worktree(repo)
    )
    skeleton: dict[str, object] = {
        "slug": "<kebab-case-slug>",
        "pr": None,
        "author_kind": "agent",
        "kinds": {change.symbol: ["<kind>"] for change in changes},
        "correctness_argument": "",
        "measurement_type_decision": False,
        "source_quotes": [],
        "alternatives_rejected": [],
        "owner_sign_off": {"signed": False, "quote": None, "url": None},
    }
    lines = [f"# changed since {base_commit[:12]} (kinds: {', '.join(K)})"]
    for change in changes:
        lines.append(f"# {change.symbol} ({change.status}, {change.kind})")
        lines.extend(f"#   - {reason}" for reason in change.reasons)
    return skeleton, lines


def cmd_template(args: argparse.Namespace) -> int:
    """Print a statement skeleton (stdout) and the structural reasons (stderr)."""
    skeleton, lines = _template(args.repo, args.base)
    print("\n".join(lines), file=sys.stderr)
    print(json.dumps(skeleton, indent=2))
    return 0


def _existing_with_slug(repo: Path, slug: str) -> Path | None:
    for path in sorted(records_dir(repo).glob(f"*-{slug}.json")):
        match = RECORD_NAME_RE.match(path.name)
        if match is not None and match["slug"] == slug:
            return path
    return None


def cmd_new(args: argparse.Namespace) -> int:
    """Build the record from a statement and write it."""
    statement = SchemaChangeStatement.model_validate_json(
        Path(args.statement).read_text(encoding="utf-8")
    )
    existing = _existing_with_slug(args.repo, statement.slug)
    if existing is not None and not args.refresh:
        raise UsageError(f"{existing.name} exists; pass --refresh to rewrite it")
    record_date = (
        SchemaChangeRecord.model_validate_json(
            existing.read_text(encoding="utf-8")
        ).date
        if existing is not None
        else dt.date.fromisoformat(args.date)
    )
    record, left_out = build_record(
        statement, repo=args.repo, base=args.base, record_date=record_date
    )
    path = write_record(record, args.repo)
    print(f"wrote {path.relative_to(args.repo)}")
    print(f"rebuild scope: {record.rebuild_scope}")
    if record.served_datasets:
        print(f"served datasets that rebuild: {', '.join(record.served_datasets)}")
    if record.schema_breaking:
        print("label the PR: gh pr edit <N> --add-label schema-breaking")
    for change in left_out:
        print(f"not in this record: {change.symbol} ({change.status})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """The enforcement test's check, from the CLI."""
    baseline = load_baseline(args.repo)
    records = load_records(args.repo)
    problems = enforcement_problems(baseline, records, surface_in_worktree(args.repo))
    for problem in problems:
        print(f"schema-change check: {problem}")
    if problems:
        return 1
    print(
        f"schema-change check: {len(records)} record(s) carry every change since "
        f"{baseline.base_commit[:12]}"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """One line per record."""
    for name, record in load_records(args.repo).items():
        symbols = ", ".join(
            f"{entry.symbol}[{'+'.join(entry.kinds)}]" for entry in record.symbols
        )
        pr = f"#{record.pr}" if record.pr is not None else "no-pr"
        signed = "signed" if record.owner_sign_off.signed else "unsigned"
        print(
            f"{name}  {pr}  {record.author_kind}  {record.rebuild_scope}  {signed}  "
            f"{symbols}"
        )
    return 0


def cmd_tripwire(args: argparse.Namespace) -> int:
    """The pre-commit check: staged schema against the merge base with ``--base``."""
    problems, notes = branch_problems(args.repo, args.base)
    for note in notes:
        print(f"schema-change-records: note: {note}")
    for problem in problems:
        print(f"schema-change-records: {problem}")
    if not problems:
        print(f"schema-change-records: every schema change vs {args.base} is recorded")
        return 0
    if os.environ.get(ACK_ENV):
        print(
            f"schema-change-records: {ACK_ENV} set -> passing this commit; CI still "
            "requires the record before landing"
        )
        return 0
    print(
        f"schema-change-records: {len(problems)} problem(s); see "
        "notes/torchcell.datamodels.schema-change-policy.md"
    )
    return 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """Grandfather every symbol at ``--ref`` (one time, when the policy lands)."""
    path = records_dir(args.repo) / BASELINE_FILENAME
    if path.exists() and not args.force:
        raise UsageError(
            f"{path.name} exists; the baseline is written once (--force rewrites it, and "
            "the pinned commit in tests/torchcell/datamodels/"
            "test_schema_change_records.py must change with it)"
        )
    commit = _git(args.repo, "rev-parse", args.ref).strip()
    surface = surface_at_ref(args.repo, commit)
    baseline = GrandfatheredBaseline(
        base_commit=commit,
        python_version=platform.python_version(),
        fingerprints=dict(sorted(surface.fingerprints.items())),
    )
    if not args.write:
        print(_dump(baseline), end="")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(baseline), encoding="utf-8")
    print(f"wrote {path.relative_to(args.repo)} ({len(baseline.fingerprints)} symbols)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``schema-change`` argument parser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, default=REPO, help="repository root")
    parser = argparse.ArgumentParser(
        prog="schema-change", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common])

    template = add("template", "statement skeleton for this branch")
    template.add_argument("--base", default="origin/main")
    template.set_defaults(func=cmd_template)

    new = add("new", "compute and write a record from a statement")
    new.add_argument("--statement", required=True, help="statement JSON path")
    new.add_argument("--base", default="origin/main")
    new.add_argument(
        "--date", default=dt.date.today().isoformat(), help="record date (YYYY-MM-DD)"
    )
    new.add_argument(
        "--refresh", action="store_true", help="rewrite the record with this slug"
    )
    new.set_defaults(func=cmd_new)

    check = add("check", "records explain every change since the baseline")
    check.set_defaults(func=cmd_check)

    listing = add("list", "one line per record")
    listing.set_defaults(func=cmd_list)

    tripwire = add("tripwire", "the pre-commit check")
    tripwire.add_argument("--base", default="origin/main")
    tripwire.set_defaults(func=cmd_tripwire)

    baseline = add("baseline", "grandfather a commit (one time)")
    baseline.add_argument("--ref", required=True)
    baseline.add_argument("--write", action="store_true")
    baseline.add_argument("--force", action="store_true")
    baseline.set_defaults(func=cmd_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit 0 ok, 1 a failed check, 2 a usage or validation error."""
    args = build_parser().parse_args(argv)
    args.repo = args.repo.resolve()
    try:
        return int(args.func(args))
    except (UsageError, ValidationError) as error:
        print(f"schema-change: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
