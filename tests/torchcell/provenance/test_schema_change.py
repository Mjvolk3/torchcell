# tests/torchcell/provenance/test_schema_change.py
# [[tests.torchcell.provenance.test_schema_change]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/test_schema_change.py
"""Tests for schema_change: kind rules, the record model, chains, the tripwire, the CLI.

The kind rules are tabled against ``schema_impact``'s exact reason strings. The record
tests build entries with literal fingerprints (``"a" * 64`` and so on); the chain and
enforcement tests use those plus the fixture surfaces of ``_schema_change_repo``. The git
tests run the real index-vs-merge-base diff in a throwaway repository (``main`` = the
fixture, ``feature`` = the edit): adding ``fitness_note`` to ``FitnessPhenotype`` is in
``ServedDataset``'s closure (via ``Experiment``) and not in ``DevOnlyDataset``'s, so it
needs a record whose scope is ``full_kg_rebuild``; a new class nothing imports needs none.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.torchcell.provenance._schema_change_repo import (
    DEV_LOADER,
    PYDANT,
    RELEASE,
    SCHEMA,
    SERVED_LOADER,
    edit_schema,
    git,
    hermetic_git,
    make_repo,
    snapshot_json,
    write,
)
from torchcell.provenance import schema_change as sc
from torchcell.provenance.schema_deps import SchemaSurface, load_surface_from_sources
from torchcell.provenance.schema_impact import ChangeKind
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

K = sc.SchemaChangeKind
FP_A, FP_B, FP_C, FP_D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
COMMIT = "1" * 40
DAY = dt.date(2026, 10, 12)
NOTE_FIELD = "    fitness_note: str | None = Field(default=None)\n"
FITNESS_ANCHOR = "    fitness: float\n"


def _surface(schema: str) -> SchemaSurface:
    return load_surface_from_sources(
        {
            "torchcell/datamodels/schema.py": schema,
            "torchcell/datamodels/pydant.py": PYDANT,
        }
    )


def _entry(
    symbol: str = "FitnessPhenotype",
    before: str | None = FP_A,
    after: str | None = FP_B,
    *,
    kinds: tuple[K, ...] = (K.additive_field,),
    reasons: tuple[str, ...] = ("added optional field 'x'",),
    status: sc.SymbolStatus = "modified",
) -> sc.SymbolChangeEntry:
    return sc.SymbolChangeEntry(
        symbol=symbol,
        status=status,
        kinds=list(kinds),
        fingerprint_before=before,
        fingerprint_after=after,
        impact=ChangeKind.stale,
        reasons=list(reasons),
    )


QUOTE = SourcedValue(
    value="10x Chromium 3' UMI counts",
    provenance=Provenance(source_uri="paper.md", citation_key="k2020", sha256=FP_C),
    quote="Single cell library preparation",
)
SIGNED = sc.OwnerSignOff(signed=True, url="https://github.com/o/r/issues/1")
UNSIGNED = sc.OwnerSignOff(signed=False)
ALTERNATIVE = sc.RejectedAlternative(alternative="keep it", why_rejected="wrong")


def _record_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "record_version": 1,
        "slug": "fitness-note",
        "date": DAY,
        "pr": 7,
        "author_kind": "agent",
        "base_commit": COMMIT,
        "python_version": "3.13.0",
        "symbols": [_entry()],
        "correctness_argument": "the source releases a note",
        "measurement_type_decision": False,
        "source_quotes": [],
        "alternatives_rejected": [],
        "served_release": RELEASE,
        "affected_loaders": [],
        "served_datasets": [],
        "rebuild_scope": sc.RebuildScope.none,
        "owner_sign_off": UNSIGNED,
    }
    fields.update(overrides)
    return fields


def _record(**overrides: object) -> sc.SchemaChangeRecord:
    return sc.SchemaChangeRecord.model_validate(_record_fields(**overrides))


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    hermetic_git(monkeypatch, tmp_path)
    monkeypatch.delenv(sc.ACK_ENV, raising=False)
    yield


# --------------------------------------------------------------------------- #
# Kind rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("new symbol", {K.additive_leaf, K.rename}),
        ("new module-level binding", {K.additive_leaf, K.rename}),
        ("symbol removed", {K.removal, K.rename}),
        ("module-level binding removed", {K.removal, K.rename}),
        (
            "base classes changed: ['A'] -> ['B']",
            {
                K.typed_gap,
                K.contract_tightened,
                K.field_meaning,
                K.field_required_loosened,
                K.rename,
            },
        ),
        (
            "model Config changed",
            {K.contract_tightened, K.field_meaning, K.field_required_loosened},
        ),
        ("added required field 'x'", {K.contract_tightened, K.field_meaning, K.rename}),
        ("added optional field 'x'", {K.additive_field, K.typed_gap, K.rename}),
        ("removed field 'x'", {K.removal, K.rename}),
        ("field 'x' became required", {K.contract_tightened}),
        ("field 'x' became optional", {K.field_required_loosened}),
        (
            "field 'x' default changed",
            {K.contract_tightened, K.field_meaning, K.field_required_loosened},
        ),
        ("added member/const 'c'", {K.enum_member_added, K.rename}),
        ("removed member/const 'c'", {K.enum_member_removed, K.removal, K.rename}),
        ("member/const 'c' value changed", {K.field_meaning, K.rename}),
        ("validator/serializer changed: ['_check']", set(K) - {K.additive_leaf}),
        ("contract fingerprint changed", set(K) - {K.additive_leaf}),
        (
            "module-level '_helper' changed",
            {
                K.equivalent_refactor,
                K.contract_tightened,
                K.field_meaning,
                K.field_required_loosened,
                K.rename,
            },
        ),
        ("lost members ['a']", {K.enum_member_removed, K.removal, K.rename}),
        ("gained members ['c']", {K.enum_member_added}),
        ("members reordered", {K.equivalent_refactor}),
        (
            "field 'n' type changed: dict[str, int] -> dict[str, int] | None",
            {K.typed_gap, K.field_required_loosened, K.field_meaning},
        ),
        (
            "field 'n' type changed: int -> None | int",
            {K.typed_gap, K.field_required_loosened, K.field_meaning},
        ),
        (
            "field 'n' type changed: int -> Optional[int]",
            {K.typed_gap, K.field_required_loosened, K.field_meaning},
        ),
        (
            "field 'n' type changed: int -> float",
            {
                K.field_meaning,
                K.contract_tightened,
                K.field_required_loosened,
                K.rename,
            },
        ),
    ],
)
def test_admissible_kinds_per_schema_impact_reason(
    reason: str, expected: set[K]
) -> None:
    """Each reason string schema_impact emits maps to its exact admissible set."""
    assert sc.admissible_kinds(reason) == frozenset(expected)


def test_an_unknown_reason_is_refused() -> None:
    """A reason the table does not know fails loudly instead of passing as anything."""
    with pytest.raises(ValueError, match="unclassified schema-impact reason 'novel'"):
        sc.admissible_kinds("novel")


@pytest.mark.parametrize(
    ("kinds", "reasons", "expected"),
    [
        (
            [K.additive_field],
            ["added optional field 'x'", "validator/serializer changed: ['_c']"],
            [],
        ),
        (
            [K.additive_field],
            ["removed field 'x'"],
            [
                "no claimed kind covers \"removed field 'x'\" (admissible: removal, "
                "rename)",
                "kind additive_field has no structural witness among the reasons",
            ],
        ),
        ([K.typed_gap], ["field 'n' type changed: int -> int | None"], []),
        (
            [K.typed_gap],
            ["field 'n' type changed: int -> float"],
            [
                "no claimed kind covers \"field 'n' type changed: int -> float\" "
                "(admissible: contract_tightened, field_meaning, "
                "field_required_loosened, rename)",
                "kind typed_gap has no structural witness among the reasons",
            ],
        ),
        (
            [K.enum_member_added],
            ["validator/serializer changed: ['_c']"],
            ["kind enum_member_added has no structural witness among the reasons"],
        ),
        ([K.contract_tightened], ["validator/serializer changed: ['_c']"], []),
        ([K.rename], ["symbol removed"], []),
        ([K.equivalent_refactor], ["members reordered"], []),
        (
            [K.additive_field, K.removal],
            ["added optional field 'y'", "removed field 'x'"],
            [],
        ),
    ],
)
def test_kind_problems_checks_coverage_and_witness(
    kinds: list[K], reasons: list[str], expected: list[str]
) -> None:
    """Every reason covered by a claimed kind; every structural kind witnessed."""
    assert sc.kind_problems(kinds, reasons) == expected


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"status": "added"},
            r"status added needs before/after present = \(False, True\), got "
            r"\(True, True\)",
        ),
        ({"before": "xyz"}, "'xyz' is not a 64-hex fingerprint"),
        ({"after": FP_A}, "before and after fingerprints are equal"),
        (
            {"kinds": (K.additive_field, K.additive_field)},
            "kinds must be non-empty and unique",
        ),
        ({"reasons": ()}, "reasons cannot be empty"),
        ({"kinds": (K.enum_member_added,)}, "FitnessPhenotype: no claimed kind covers"),
        ({"symbol": " "}, "symbol cannot be blank"),
    ],
)
def test_symbol_entry_refusals(overrides: dict[str, object], message: str) -> None:
    """Status, fingerprints, kinds and reasons must agree."""
    with pytest.raises(ValidationError, match=message):
        _entry(**overrides)  # type: ignore[arg-type]


def test_an_added_and_a_removed_entry_validate() -> None:
    """Added: no before; removed: no after."""
    added = _entry(
        "NewLeaf", None, FP_A, kinds=(K.additive_leaf,), reasons=("new symbol",),
        status="added",
    )  # fmt: skip
    removed = _entry(
        "OldLeaf", FP_A, None, kinds=(K.removal,), reasons=("symbol removed",),
        status="removed",
    )  # fmt: skip
    assert (added.fingerprint_before, removed.fingerprint_after) == (None, None)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"signed": True}, "a sign-off needs the owner's quote or a comment URL"),
        (
            {"signed": False, "quote": "fine"},
            "an unsigned sign-off carries no quote or URL",
        ),
        (
            {"signed": True, "url": "http://example.com"},
            "owner_sign_off.url must start with https://github.com/",
        ),
        ({"signed": True, "quote": "  "}, "owner_sign_off.quote cannot be blank"),
    ],
)
def test_owner_sign_off_refusals(fields: dict[str, object], message: str) -> None:
    """A signature carries evidence; no signature carries none."""
    with pytest.raises(ValidationError, match=message):
        sc.OwnerSignOff.model_validate(fields)


def test_owner_sign_off_by_quote() -> None:
    """The owner's words verbatim are evidence enough."""
    sign_off = sc.OwnerSignOff(
        signed=True, quote="why would you get rid of old classes?"
    )
    assert sign_off.model_dump() == {
        "signed": True,
        "quote": "why would you get rid of old classes?",
        "url": None,
    }


def test_rejected_alternative_needs_both_halves() -> None:
    """A blank reason for rejecting is refused."""
    with pytest.raises(ValidationError, match="why_rejected cannot be blank"):
        sc.RejectedAlternative(alternative="two leaves", why_rejected=" ")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"served_classes": ["Other"], "via": ["Media"]},
            r"served \['Other'\] not among its classes",
        ),
        ({"served_classes": [], "via": []}, "via cannot be empty"),
    ],
)
def test_affected_loader_refusals(fields: dict[str, object], message: str) -> None:
    """Served classes are the loader's own; a loader is affected via something."""
    with pytest.raises(ValidationError, match=message):
        sc.AffectedLoader.model_validate(
            {
                "loader": "torchcell/datasets/served.py",
                "dataset_classes": ["ServedDataset"],
                "impact": "stale",
                **fields,
            }
        )


def _loader(served: bool, via: str = "FitnessPhenotype") -> sc.AffectedLoader:
    return sc.AffectedLoader(
        loader="torchcell/datasets/served.py",
        dataset_classes=["ServedDataset"],
        served_classes=["ServedDataset"] if served else [],
        impact=ChangeKind.stale,
        via=[via],
    )


def test_rebuild_scope_follows_the_served_flags() -> None:
    """No loader: none; only unserved: dev stores; any served: full rebuild."""
    assert [
        sc.rebuild_scope_of([]),
        sc.rebuild_scope_of([_loader(False)]),
        sc.rebuild_scope_of([_loader(False), _loader(True)]),
    ] == [
        sc.RebuildScope.none,
        sc.RebuildScope.dev_stores,
        sc.RebuildScope.full_kg_rebuild,
    ]


REMOVAL = _entry(
    "OldLeaf", FP_A, None, kinds=(K.removal,), reasons=("symbol removed",),
    status="removed",
)  # fmt: skip


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"slug": "Fitness_Note"}, "slug 'Fitness_Note' is not kebab-case"),
        ({"base_commit": "abc"}, "base_commit 'abc' is not a 40-hex sha"),
        ({"pr": 0}, "pr must be a positive PR number, got 0"),
        ({"symbols": []}, "a record names at least one symbol"),
        (
            {"symbols": [_entry("Zed"), _entry("Alpha")]},
            "symbols must be unique and sorted by name",
        ),
        ({"correctness_argument": " "}, "correctness_argument cannot be blank"),
        (
            {"measurement_type_decision": True, "alternatives_rejected": [ALTERNATIVE]},
            "a measurement-type decision carries at least one verbatim source quote",
        ),
        (
            {"symbols": [REMOVAL], "alternatives_rejected": [ALTERNATIVE]},
            r"OldLeaf:removal need the owner's sign-off",
        ),
        (
            {"symbols": [REMOVAL], "owner_sign_off": SIGNED},
            "a sign-off change or a measurement-type decision names at least one "
            "rejected alternative",
        ),
        (
            {"affected_loaders": [_loader(False, via="Media")]},
            r"via \['Media'\] not in this record",
        ),
        (
            {"affected_loaders": [_loader(True)], "rebuild_scope": "full_kg_rebuild"},
            r"served_datasets \[\] != the affected loaders' served classes "
            r"\['ServedDataset'\]",
        ),
        (
            {"affected_loaders": [_loader(False)]},
            "rebuild_scope none != dev_stores implied by the affected loaders",
        ),
    ],
)
def test_record_refusals(overrides: dict[str, object], message: str) -> None:
    """Identity, argument, sign-off and rebuild scope must agree with the symbols."""
    with pytest.raises(ValidationError, match=message):
        _record(**overrides)


def test_a_signed_measurement_type_removal_validates() -> None:
    """Sign-off, an alternative and a sourced quote together admit a removal."""
    record = _record(
        symbols=[REMOVAL],
        owner_sign_off=SIGNED,
        alternatives_rejected=[ALTERNATIVE],
        measurement_type_decision=True,
        source_quotes=[QUOTE],
        affected_loaders=[_loader(True, via="OldLeaf")],
        served_datasets=["ServedDataset"],
        rebuild_scope="full_kg_rebuild",
    )
    assert (record.filename, record.schema_breaking) == (
        "2026-10-12-fitness-note.json",
        True,
    )


def test_baseline_refusals() -> None:
    """The baseline names a real commit and well-formed fingerprints."""
    with pytest.raises(ValidationError, match="base_commit 'HEAD' is not a 40-hex"):
        sc.GrandfatheredBaseline(
            base_commit="HEAD", python_version="3.13.0", fingerprints={}
        )
    with pytest.raises(
        ValidationError, match=r"malformed fingerprints for \['Media'\]"
    ):
        sc.GrandfatheredBaseline(
            base_commit=COMMIT, python_version="3.13.0", fingerprints={"Media": "x"}
        )


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def test_write_and_load_round_trip(tmp_path: Path) -> None:
    """A written record loads back equal, keyed by its file name."""
    record = _record()
    path = sc.write_record(record, tmp_path)
    assert path == tmp_path / "database/schema_changes/2026-10-12-fitness-note.json"
    assert sc.load_records(tmp_path) == {"2026-10-12-fitness-note.json": record}


def test_load_records_refuses_stray_files_mismatched_names_and_repeated_slugs(
    tmp_path: Path,
) -> None:
    """The directory holds the baseline and well-named records only."""
    directory = tmp_path / "database/schema_changes"
    directory.mkdir(parents=True)
    (directory / "notes.json").write_text("{}")
    with pytest.raises(ValueError, match="notes.json is not <YYYY-MM-DD>-<slug>.json"):
        sc.load_records(tmp_path)
    (directory / "notes.json").rename(directory / "2026-10-11-fitness-note.json")
    (directory / "2026-10-11-fitness-note.json").write_text(_record().model_dump_json())
    with pytest.raises(
        ValueError,
        match="2026-10-11-fitness-note.json: a record with this date and slug is "
        "2026-10-12-fitness-note.json",
    ):
        sc.load_records(tmp_path)
    (directory / "2026-10-11-fitness-note.json").write_text(
        _record(date=dt.date(2026, 10, 11)).model_dump_json()
    )
    sc.write_record(_record(), tmp_path)
    with pytest.raises(
        ValueError, match=r"used by more than one record: \['fitness-note'\]"
    ):
        sc.load_records(tmp_path)


# --------------------------------------------------------------------------- #
# Chains and enforcement
# --------------------------------------------------------------------------- #
def _named(
    *items: tuple[str, str | None, str | None],
) -> list[tuple[str, sc.SymbolChangeEntry]]:
    out = []
    for name, before, after in items:
        status: sc.SymbolStatus = (
            "added" if before is None else "removed" if after is None else "modified"
        )
        kinds, reasons = {
            "added": ((K.additive_leaf,), ("new symbol",)),
            "removed": ((K.removal,), ("symbol removed",)),
            "modified": ((K.additive_field,), ("added optional field 'x'",)),
        }[status]
        out.append(
            (
                name,
                _entry("S", before, after, kinds=kinds, reasons=reasons, status=status),
            )
        )
    return out


@pytest.mark.parametrize(
    ("items", "start", "end", "anchored", "expected"),
    [
        ([("r1", FP_A, FP_B)], FP_A, FP_B, True, []),
        ([("r2", FP_B, FP_C), ("r1", FP_A, FP_B)], FP_A, FP_C, True, []),
        ([("r1", None, FP_A), ("r2", FP_A, None)], None, None, True, []),
        (
            [("r1", FP_B, FP_C)],
            FP_A,
            FP_C,
            True,
            [
                "S: no record carries it on from aaaaaaaaaaaa; ['r1'] start elsewhere "
                "(stale after a rebase or an edit: rerun schema-change new --refresh)"
            ],
        ),
        (
            [("r1", FP_A, FP_B), ("r2", FP_A, FP_C)],
            FP_A,
            FP_C,
            True,
            [
                "S: ['r1', 'r2'] all start from aaaaaaaaaaaa; refresh the later one "
                "after rebasing (schema-change new --refresh)"
            ],
        ),
        (
            [("r1", FP_A, FP_B)],
            FP_A,
            FP_C,
            True,
            [
                "S: r1 leaves it at bbbbbbbbbbbb but the schema has cccccccccccc: the "
                "record is stale (schema-change new --refresh)"
            ],
        ),
        ([("r2", FP_C, FP_D), ("r1", FP_B, FP_C)], None, FP_D, False, []),
        (
            [("r1", FP_B, FP_C), ("r2", FP_A, FP_D)],
            None,
            FP_D,
            False,
            ["S: records ['r1', 'r2'] do not form one chain"],
        ),
    ],
)
def test_symbol_chain_problems(
    items: list[tuple[str, str | None, str | None]],
    start: str | None,
    end: str | None,
    anchored: bool,
    expected: list[str],
) -> None:
    """Records link before-to-after from the anchor to the current fingerprint."""
    assert (
        sc.symbol_chain_problems("S", _named(*items), start, end, anchored=anchored)
        == expected
    )


NOTED = SCHEMA.replace(FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)


def _baseline(schema: str = SCHEMA) -> sc.GrandfatheredBaseline:
    return sc.GrandfatheredBaseline(
        base_commit=COMMIT,
        python_version="3.13.0",
        fingerprints=_surface(schema).fingerprints,
    )


def _fitness_record(**overrides: object) -> sc.SchemaChangeRecord:
    entry = _entry(
        before=_surface(SCHEMA).fingerprints["FitnessPhenotype"],
        after=_surface(NOTED).fingerprints["FitnessPhenotype"],
        reasons=("added optional field 'fitness_note'",),
    )
    return _record(symbols=[entry], **overrides)


def test_enforcement_on_an_unchanged_schema_is_clean() -> None:
    """At the baseline, no record is needed and none is missing."""
    assert sc.enforcement_problems(_baseline(), {}, _surface(SCHEMA)) == []


def test_enforcement_names_an_unrecorded_change_and_removal() -> None:
    """A moved and a removed baseline symbol without a record are both named."""
    current = _surface(
        NOTED.replace("class Media(ModelStrict)", "class Medium(ModelStrict)")
    )
    assert sc.enforcement_problems(_baseline(), {}, current) == [
        "FitnessPhenotype: changed since the grandfathered baseline 111111111111 and "
        "no record names it",
        "Media: removed since the grandfathered baseline 111111111111 and no record "
        "names it",
    ]


def test_enforcement_accepts_a_chained_record_and_a_new_unrecorded_symbol() -> None:
    """The record carries FitnessPhenotype; NewLeaf (not in the baseline) needs none."""
    current = _surface(NOTED + "\n\nclass NewLeaf(ModelStrict):\n    x: int\n")
    records = {"2026-10-12-fitness-note.json": _fitness_record()}
    assert sc.enforcement_problems(_baseline(), records, current) == []


def test_enforcement_flags_a_stale_record_a_missing_pr_and_another_python() -> None:
    """A record whose after no longer matches is stale; pr and interpreter are checked."""
    records = {
        "2026-10-12-fitness-note.json": _fitness_record(
            pr=None, python_version="3.13.5"
        )
    }
    assert sc.enforcement_problems(_baseline(), records, _surface(SCHEMA)) == [
        "2026-10-12-fitness-note.json: pr is not set (set it once the PR is open)",
        "2026-10-12-fitness-note.json: fingerprints computed under Python 3.13.5, the "
        "baseline under 3.13.0",
        "FitnessPhenotype: 2026-10-12-fitness-note.json leaves it at "
        f"{_surface(NOTED).fingerprints['FitnessPhenotype'][:12]} but the schema has "
        f"{_surface(SCHEMA).fingerprints['FitnessPhenotype'][:12]}: the record is stale "
        "(schema-change new --refresh)",
    ]


# --------------------------------------------------------------------------- #
# The tripwire and build_record, in a git repository
# --------------------------------------------------------------------------- #
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The fixture repository on ``feature``."""
    return make_repo(tmp_path)


def _statement(**overrides: object) -> sc.SchemaChangeStatement:
    fields: dict[str, object] = {
        "slug": "fitness-note",
        "pr": 7,
        "author_kind": "agent",
        "kinds": {"FitnessPhenotype": [K.additive_field]},
        "correctness_argument": "the source releases a note",
        "measurement_type_decision": False,
        "source_quotes": [],
        "alternatives_rejected": [],
        "owner_sign_off": UNSIGNED,
    }
    fields.update(overrides)
    return sc.SchemaChangeStatement.model_validate(fields)


def _stage_record(
    repo: Path, statement: sc.SchemaChangeStatement
) -> sc.SchemaChangeRecord:
    record, _ = sc.build_record(statement, repo=repo, base="main", record_date=DAY)
    path = sc.write_record(record, repo)
    git(repo, "add", str(path.relative_to(repo)))
    return record


def test_tripwire_is_silent_without_a_schema_change(repo: Path) -> None:
    """No change and no staged record: nothing to say."""
    assert sc.branch_problems(repo, "main") == ([], [])


def test_tripwire_names_an_unrecorded_served_change(repo: Path) -> None:
    """A served class's new optional field needs a record, and the hit is named."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    base = git(repo, "rev-parse", "main")[:12]
    assert sc.branch_problems(repo, "main") == (
        [
            "FitnessPhenotype (modified, stale; served via ['ServedDataset']) changed "
            f"since the merge base {base} and no staged record names it "
            "(schema-change template / new)"
        ],
        [],
    )


def test_tripwire_exempts_a_new_symbol_no_served_closure_holds(repo: Path) -> None:
    """A new unused class is a note for the PR body, not a problem."""
    edit_schema(
        repo,
        "class Experiment(",
        "class NewLeaf(ModelStrict):\n    x: int\n\n\nclass Experiment(",
    )
    assert sc.branch_problems(repo, "main") == (
        [],
        [
            "NewLeaf: new and in no served closure, so no record is required; list it "
            "in the PR body"
        ],
    )


def test_tripwire_requires_a_record_for_a_new_symbol_a_served_loader_imports(
    repo: Path,
) -> None:
    """Once a served loader imports the new class, it is in a served closure."""
    edit_schema(
        repo,
        "class Experiment(",
        "class NewLeaf(ModelStrict):\n    x: int\n\n\nclass Experiment(",
    )
    write(
        repo,
        "torchcell/datasets/served.py",
        SERVED_LOADER.replace("Experiment", "Experiment, NewLeaf"),
    )
    problems, notes = sc.branch_problems(repo, "main")
    assert (len(problems), notes) == (1, [])
    assert problems[0].startswith(
        "NewLeaf (added, stale; served via ['ServedDataset']) changed since the merge base"
    )


def test_build_record_measures_the_change_and_the_tripwire_accepts_it(
    repo: Path,
) -> None:
    """The record carries the fingerprints, reasons, loaders and full-rebuild scope."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    record = _stage_record(repo, _statement())
    entry = record.symbols[0]
    assert (entry.fingerprint_before, entry.fingerprint_after) == (
        _surface(SCHEMA).fingerprints["FitnessPhenotype"],
        _surface(NOTED).fingerprints["FitnessPhenotype"],
    )
    assert (entry.reasons, entry.impact, record.base_commit) == (
        ["added optional field 'fitness_note'"],
        ChangeKind.stale,
        git(repo, "rev-parse", "main"),
    )
    assert [loader.model_dump(mode="json") for loader in record.affected_loaders] == [
        {
            "loader": "torchcell/datasets/served.py",
            "dataset_classes": ["ServedDataset"],
            "served_classes": ["ServedDataset"],
            "impact": "stale",
            "via": ["FitnessPhenotype"],
        }
    ]
    assert (record.served_release, record.served_datasets, record.rebuild_scope) == (
        RELEASE,
        ["ServedDataset"],
        sc.RebuildScope.full_kg_rebuild,
    )
    assert sc.branch_problems(repo, "main") == ([], [])


def test_a_dev_only_record_scopes_to_dev_stores(repo: Path) -> None:
    """A new class only the unserved loader imports rebuilds dev stores only."""
    edit_schema(
        repo,
        "class Experiment(",
        "class DevLeaf(ModelStrict):\n    x: int\n\n\nclass Experiment(",
    )
    write(
        repo, "torchcell/datasets/dev.py", DEV_LOADER.replace("Media", "DevLeaf, Media")
    )
    record = _stage_record(
        repo, _statement(slug="dev-leaf", kinds={"DevLeaf": [K.additive_leaf]})
    )
    assert (record.rebuild_scope, record.served_datasets) == (
        sc.RebuildScope.dev_stores,
        [],
    )
    assert sc.branch_problems(repo, "main") == ([], [])


def test_tripwire_flags_a_record_the_schema_moved_past(repo: Path) -> None:
    """Staging a further edit after the record makes the record stale."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    _stage_record(repo, _statement())
    edit_schema(
        repo,
        NOTE_FIELD,
        NOTE_FIELD + "    fitness_flag: bool | None = Field(default=None)\n",
    )
    assert sc.branch_problems(repo, "main") == (
        [
            "2026-10-12-fitness-note.json: stale for FitnessPhenotype (the staged "
            "schema differs from what the record measured; schema-change new --refresh)"
        ],
        [],
    )


def test_tripwire_flags_a_record_naming_an_unchanged_symbol(repo: Path) -> None:
    """Reverting the schema leaves the record naming nothing that changed."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    _stage_record(repo, _statement())
    edit_schema(repo, NOTE_FIELD, "")
    base = git(repo, "rev-parse", "main")[:12]
    assert sc.branch_problems(repo, "main") == (
        [
            "2026-10-12-fitness-note.json: names FitnessPhenotype, which did not change "
            f"since the merge base {base}",
            f"2026-10-12-fitness-note.json: affected loaders or served release ({RELEASE}) "
            "differ from the staged tree (schema-change new --refresh)",
        ],
        [],
    )


def test_tripwire_reads_records_from_the_index_only(repo: Path) -> None:
    """An unstaged record does not count; a staged invalid or misnamed one is reported."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    record, _ = sc.build_record(_statement(), repo=repo, base="main", record_date=DAY)
    path = sc.write_record(record, repo)
    problems, _ = sc.branch_problems(repo, "main")
    assert problems[0].startswith("FitnessPhenotype (modified, stale;")
    data = json.loads(path.read_text())
    data["symbols"][0]["kinds"] = ["removal"]
    path.write_text(json.dumps(data))
    git(repo, "add", str(path.relative_to(repo)))
    problems, _ = sc.branch_problems(repo, "main")
    assert problems[0].startswith(
        "2026-10-12-fitness-note.json: invalid record: 1 validation error for "
        "SchemaChangeRecord"
    )
    misnamed = path.with_name("2026-10-11-fitness-note.json")
    git(repo, "mv", "-f", str(path.relative_to(repo)), str(misnamed.relative_to(repo)))
    misnamed.write_text(record.model_dump_json())
    git(repo, "add", str(misnamed.relative_to(repo)))
    problems, _ = sc.branch_problems(repo, "main")
    assert problems[0] == (
        "2026-10-11-fitness-note.json: a record with this date and slug is "
        "2026-10-12-fitness-note.json"
    )


def test_tripwire_refuses_one_symbol_in_two_records(repo: Path) -> None:
    """One record per symbol per branch."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    _stage_record(repo, _statement())
    _stage_record(repo, _statement(slug="fitness-note-again"))
    problems, _ = sc.branch_problems(repo, "main")
    assert problems == [
        "FitnessPhenotype: named by ['2026-10-12-fitness-note-again.json', "
        "'2026-10-12-fitness-note.json']; one record per symbol per branch"
    ]


def test_build_record_refuses_unchanged_and_empty_statements(repo: Path) -> None:
    """A statement names changed symbols, and at least one."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    with pytest.raises(sc.UsageError, match=r"names \['Media'\], which did not change"):
        sc.build_record(
            _statement(kinds={"Media": [K.additive_field]}),
            repo=repo,
            base="main",
            record_date=DAY,
        )
    with pytest.raises(sc.UsageError, match="the statement names no symbol"):
        sc.build_record(_statement(kinds={}), repo=repo, base="main", record_date=DAY)


def test_served_release_is_the_newest_snapshot(repo: Path) -> None:
    """The latest built_at wins; an empty releases directory is an error."""
    write(
        repo,
        "database/releases/2026.10.20-newer.json",
        snapshot_json(
            "2026.10.20-newer", "2026-10-20T00:00:00-05:00", ["DevOnlyDataset"]
        ),
    )
    assert sc.served_release(repo) == (
        "2026.10.20-newer",
        frozenset({"DevOnlyDataset"}),
    )
    for path in (repo / "database/releases").glob("*.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="no release snapshot under"):
        sc.served_release(repo)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_statement(tmp_path: Path, statement: sc.SchemaChangeStatement) -> str:
    path = tmp_path / "statement.json"
    path.write_text(statement.model_dump_json())
    return str(path)


def test_cli_template_lists_changed_symbols(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The skeleton names every changed symbol; stderr carries the reasons."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD, stage=False)
    assert sc.main(["template", "--repo", str(repo), "--base", "main"]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["kinds"] == {"FitnessPhenotype": ["<kind>"]}
    assert err.splitlines()[1:] == [
        "# FitnessPhenotype (modified, stale)",
        "#   - added optional field 'fitness_note'",
    ]


def test_cli_new_writes_refuses_a_duplicate_and_refreshes(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first new writes; a second needs --refresh and keeps the first date."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD, stage=False)
    statement = _write_statement(tmp_path, _statement())
    args = ["new", "--repo", str(repo), "--base", "main", "--statement", statement]
    assert sc.main([*args, "--date", "2026-10-12"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "wrote database/schema_changes/2026-10-12-fitness-note.json",
        "rebuild scope: full_kg_rebuild",
        "served datasets that rebuild: ServedDataset",
        "label the PR: gh pr edit <N> --add-label schema-breaking",
    ]
    assert sc.main([*args, "--date", "2026-10-13"]) == 2
    assert capsys.readouterr().err == (
        "schema-change: 2026-10-12-fitness-note.json exists; pass --refresh to "
        "rewrite it\n"
    )
    assert sc.main([*args, "--date", "2026-10-13", "--refresh"]) == 0
    assert sorted(
        p.name for p in (repo / "database/schema_changes").glob("*.json")
    ) == ["2026-10-12-fitness-note.json"]


def test_cli_new_reports_a_left_out_symbol_and_an_invalid_claim(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A changed symbol outside the statement is reported; a bad kind exits 2."""
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD, stage=False)
    edit_schema(
        repo,
        "    state: str\n",
        "    state: str\n    ph: float | None = None\n",
        stage=False,
    )
    statement = _write_statement(tmp_path, _statement())
    args = ["new", "--repo", str(repo), "--base", "main", "--statement", statement]
    assert sc.main([*args, "--date", "2026-10-12"]) == 0
    assert (
        capsys.readouterr().out.splitlines()[-1]
        == "not in this record: Media (modified)"
    )
    bad = _write_statement(
        tmp_path, _statement(slug="bad", kinds={"Media": [K.removal]})
    )
    assert (
        sc.main(["new", "--repo", str(repo), "--base", "main", "--statement", bad]) == 2
    )
    assert capsys.readouterr().err.startswith(
        "schema-change: 1 validation error for SymbolChangeEntry"
    )


def test_cli_baseline_check_and_list(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Baseline writes once; check passes, then names an unrecorded change."""
    main_commit = git(repo, "rev-parse", "main")
    assert sc.main(["baseline", "--repo", str(repo), "--ref", "main"]) == 0
    assert json.loads(capsys.readouterr().out)["base_commit"] == main_commit
    assert sc.main(["baseline", "--repo", str(repo), "--ref", "main", "--write"]) == 0
    assert capsys.readouterr().out == (
        "wrote database/schema_changes/grandfathered.json (5 symbols)\n"
    )
    assert sc.main(["baseline", "--repo", str(repo), "--ref", "main", "--write"]) == 2
    assert capsys.readouterr().err.startswith(
        "schema-change: grandfathered.json exists; the baseline is written once"
    )
    assert sc.main(["check", "--repo", str(repo)]) == 0
    assert capsys.readouterr().out == (
        f"schema-change check: 0 record(s) carry every change since {main_commit[:12]}\n"
    )
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD, stage=False)
    assert sc.main(["check", "--repo", str(repo)]) == 1
    assert capsys.readouterr().out == (
        "schema-change check: FitnessPhenotype: changed since the grandfathered "
        f"baseline {main_commit[:12]} and no record names it\n"
    )
    statement = _write_statement(tmp_path, _statement())
    assert (
        sc.main(
            [
                "new",
                "--repo",
                str(repo),
                "--base",
                "main",
                "--statement",
                statement,
                "--date",
                "2026-10-12",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert sc.main(["check", "--repo", str(repo)]) == 0
    assert sc.main(["list", "--repo", str(repo)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "2026-10-12-fitness-note.json  #7  agent  full_kg_rebuild  unsigned  "
        "FitnessPhenotype[additive_field]"
    )


def test_cli_tripwire_exit_codes_and_the_override(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 on an unrecorded change, 0 with the override, 0 when clean."""
    assert sc.main(["tripwire", "--repo", str(repo), "--base", "main"]) == 0
    assert capsys.readouterr().out == (
        "schema-change-records: every schema change vs main is recorded\n"
    )
    edit_schema(repo, FITNESS_ANCHOR, FITNESS_ANCHOR + NOTE_FIELD)
    assert sc.main(["tripwire", "--repo", str(repo), "--base", "main"]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == (
        "schema-change-records: 1 problem(s); see "
        "notes/torchcell.datamodels.schema-change-policy.md"
    )
    monkeypatch.setenv(sc.ACK_ENV, "1")
    assert sc.main(["tripwire", "--repo", str(repo), "--base", "main"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "schema-change-records: TORCHCELL_SCHEMA_RECORD_ACK set -> passing this "
        "commit; CI still requires the record before landing"
    )


def test_cli_tripwire_prints_the_exemption_note(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A new class nothing served imports passes, with a note for the PR body."""
    edit_schema(
        repo,
        "class Experiment(",
        "class NewLeaf(ModelStrict):\n    x: int\n\n\nclass Experiment(",
    )
    assert sc.main(["tripwire", "--repo", str(repo), "--base", "main"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "schema-change-records: note: NewLeaf: new and in no served closure, so no "
        "record is required; list it in the PR body",
        "schema-change-records: every schema change vs main is recorded",
    ]
