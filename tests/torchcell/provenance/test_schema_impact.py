# tests/torchcell/provenance/test_schema_impact.py
# [[tests.torchcell.provenance.test_schema_impact]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/test_schema_impact.py
"""Tests for schema_impact: breaking/stale classification, loader mapping, the CLI gate.

Fixture ``SCHEMA`` is a five-class surface (``ModelStrict`` with a v1 ``Config``, the
``MeasurementType`` enum, ``Media`` <- ``Environment`` <- ``FitnessExperiment``). Each
classification test edits one line and asserts the exact ``(kind, status, reasons)``
the source's ``classify_change`` produces; the reason strings are the f-strings at
``schema_impact.py`` lines 117 to 183, and ``RICH_OLD``/``RICH_NEW`` exercise every one
of them in a single diff so their ORDER (bases, Config, added fields, removed fields,
per-field type/required/default, members, validators) is pinned too.

The end-to-end tests build a throwaway git repository in ``tmp_path`` (``torchcell/
datamodels/schema.py`` committed at ``HEAD``, loaders under ``torchcell/datasets``),
point ``default_surface_modules`` at it, edit the working tree and run
``build_impact_report``/``format_report``/``main``. Loader closures there:
``fit_loader`` imports ``FitnessExperiment`` (closure FitnessExperiment, Environment,
Media, ModelStrict), ``scerevisiae/env_loader`` imports ``Environment`` (Environment,
Media, ModelStrict), ``mt_loader`` imports ``MeasurementType`` only, ``plain`` imports
nothing from ``torchcell.datamodels``, and ``__init__.py``/``old_deprecated.py`` import
``Media`` but are excluded by name. Adding required ``ph`` to ``Media`` (breaking) and
member ``c`` to ``MeasurementType`` (stale) therefore flags fit_loader and env_loader as
breaking via Media and mt_loader as stale via MeasurementType; breaking rows sort first,
then by path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from torchcell.provenance import schema_deps as sd
from torchcell.provenance import schema_impact as si

SCHEMA = """
from enum import Enum
from pydantic import BaseModel, Field


class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"


class MeasurementType(str, Enum):
    a = "a"
    b = "b"


class Media(ModelStrict):
    name: str
    state: str
    base_medium: str | None = Field(default=None, description="label")


class Environment(ModelStrict):
    media: Media
    temperature: float


class FitnessExperiment(ModelStrict):
    env: Environment
    fitness: float
"""


def _surface(src: str) -> sd.SchemaSurface:
    return sd.load_surface_from_sources({"schema": src})


def _change(old_src: str, new_src: str, symbol: str) -> si.SymbolChange | None:
    changes = {
        c.symbol: c for c in si.diff_surfaces(_surface(old_src), _surface(new_src))
    }
    return changes.get(symbol)


def test_no_change_yields_empty_diff() -> None:
    assert si.diff_surfaces(_surface(SCHEMA), _surface(SCHEMA)) == []


def _verdict(
    old_src: str, new_src: str, symbol: str
) -> tuple[si.ChangeKind, str, list[str]]:
    change = _change(old_src, new_src, symbol)
    assert change is not None, f"{symbol} did not change"
    return change.kind, change.status, change.reasons


B, S = si.ChangeKind.breaking, si.ChangeKind.stale


def test_added_required_field_is_breaking() -> None:
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
    assert _verdict(SCHEMA, new, "Media") == (
        B,
        "modified",
        ["added required field 'ph'"],
    )


def test_added_optional_field_is_stale() -> None:
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float = 7.0\n")
    assert _verdict(SCHEMA, new, "Media") == (
        S,
        "modified",
        ["added optional field 'ph'"],
    )


def test_removed_field_is_breaking() -> None:
    new = SCHEMA.replace("    state: str\n", "")
    assert _verdict(SCHEMA, new, "Media") == (B, "modified", ["removed field 'state'"])


def test_type_change_is_breaking() -> None:
    new = SCHEMA.replace("    temperature: float\n", "    temperature: int\n")
    assert _verdict(SCHEMA, new, "Environment") == (
        B,
        "modified",
        ["field 'temperature' type changed: float -> int"],
    )


def test_optional_to_required_is_breaking() -> None:
    new = SCHEMA.replace(
        '    base_medium: str | None = Field(default=None, description="label")\n',
        "    base_medium: str\n",
    )
    assert _verdict(SCHEMA, new, "Media") == (
        B,
        "modified",
        [
            "field 'base_medium' type changed: str | None -> str",
            "field 'base_medium' became required",
        ],
    )


def test_required_to_optional_is_stale() -> None:
    new = SCHEMA.replace("    state: str\n", "    state: str = 'liquid'\n")
    assert _verdict(SCHEMA, new, "Media") == (
        S,
        "modified",
        ["field 'state' became optional"],
    )


def test_default_change_is_stale() -> None:
    new = SCHEMA.replace("default=None", "default='YPD'")
    assert _verdict(SCHEMA, new, "Media") == (
        S,
        "modified",
        ["field 'base_medium' default changed"],
    )


def test_description_only_change_is_not_a_change() -> None:
    """``description`` is a documentation-only Field kwarg, so the fingerprint holds."""
    new = SCHEMA.replace('description="label"', 'description="relabeled"')
    assert si.diff_surfaces(_surface(SCHEMA), _surface(new)) == []


def test_added_enum_member_is_stale() -> None:
    new = SCHEMA.replace('    b = "b"\n', '    b = "b"\n    c = "c"\n')
    assert _verdict(SCHEMA, new, "MeasurementType") == (
        S,
        "modified",
        ["added member/const 'c'"],
    )


def test_removed_enum_member_is_breaking() -> None:
    new = SCHEMA.replace('    b = "b"\n', "")
    assert _verdict(SCHEMA, new, "MeasurementType") == (
        B,
        "modified",
        ["removed member/const 'b'"],
    )


def test_enum_value_change_is_breaking() -> None:
    new = SCHEMA.replace('    b = "b"\n', '    b = "beta"\n')
    assert _verdict(SCHEMA, new, "MeasurementType") == (
        B,
        "modified",
        ["member/const 'b' value changed"],
    )


def test_config_change_is_breaking() -> None:
    new = SCHEMA.replace('extra = "forbid"', 'extra = "allow"')
    assert _verdict(SCHEMA, new, "ModelStrict") == (
        B,
        "modified",
        ["model Config changed"],
    )


def test_config_change_flags_only_the_changed_class() -> None:
    """Fingerprints are per class: subclasses of ModelStrict keep theirs, so the diff
    has exactly one entry; their loaders are reached through the closure instead.
    """
    new = SCHEMA.replace('extra = "forbid"', 'extra = "allow"')
    changes = si.diff_surfaces(_surface(SCHEMA), _surface(new))
    assert [c.symbol for c in changes] == ["ModelStrict"]


def test_added_symbol_is_not_breaking() -> None:
    new = SCHEMA + "\n\nclass NewThing(ModelStrict):\n    x: int\n"
    assert _verdict(SCHEMA, new, "NewThing") == (S, "added", ["new symbol"])


def test_removed_symbol_is_breaking() -> None:
    new = SCHEMA.replace(
        "class FitnessExperiment(ModelStrict):\n    env: Environment\n    fitness: float\n",
        "",
    )
    assert _verdict(SCHEMA, new, "FitnessExperiment") == (
        B,
        "removed",
        ["symbol removed"],
    )


def test_base_class_change_is_breaking() -> None:
    new = SCHEMA.replace("class Environment(ModelStrict):", "class Environment(Media):")
    assert _verdict(SCHEMA, new, "Environment") == (
        B,
        "modified",
        ["base classes changed: ['ModelStrict'] -> ['Media']"],
    )


def test_changed_validator_is_stale() -> None:
    """Adding a ``field_validator`` changes the contract; it is classified stale, even
    though a new validator can reject records built under the old schema.
    """
    new = SCHEMA.replace(
        "    fitness: float\n",
        "    fitness: float\n\n"
        "    @field_validator('fitness')\n"
        "    def positive(cls, v):\n"
        "        return v\n",
    )
    assert _verdict(SCHEMA, new, "FitnessExperiment") == (
        S,
        "modified",
        ["validator/serializer changed: ['positive']"],
    )


def test_reordered_base_classes_are_invisible() -> None:
    """Finding: ``ContractSpec.canonical`` sorts the bases (schema_deps.py line 118), so
    ``class X(A, B)`` -> ``class X(B, A)`` keeps the fingerprint and ``diff_surfaces``
    reports nothing, although the MRO (and so which parent's field wins) changes and
    ``classify_change`` would call the same edit breaking (line 116 compares the
    unsorted tuples). Pinned until the canonical form keeps base order.
    """
    two = SCHEMA + "\n\nclass Both(Media, Environment):\n    x: int\n"
    swapped = two.replace(
        "class Both(Media, Environment):", "class Both(Environment, Media):"
    )
    old, new = _surface(two), _surface(swapped)
    assert si.diff_surfaces(old, new) == []
    assert si.classify_change(old.specs["Both"], new.specs["Both"]) == (
        B,
        ["base classes changed: ['Media', 'Environment'] -> ['Environment', 'Media']"],
    )


def test_constraint_change_is_reported_as_stale_default_change() -> None:
    """Finding: tightening a validation constraint (``Field(..., ge=0)`` -> ``ge=1``) on a
    required field is folded into the default string (schema_deps.py lines 159 to 182),
    so it reaches line 152 and is reported as "default changed", stale, although it can
    invalidate stored records (the module docstring's definition of breaking). Pinned
    until constraints are classified separately.
    """
    old = SCHEMA.replace(
        "    fitness: float\n", "    fitness: float = Field(..., ge=0)\n"
    )
    new = old.replace("ge=0", "ge=1")
    assert _verdict(old, new, "FitnessExperiment") == (
        S,
        "modified",
        ["field 'fitness' default changed"],
    )


RICH_OLD = """
from pydantic import BaseModel, Field, field_validator
class Base(BaseModel):
    pass
class Mixin(BaseModel):
    pass
class Rec(Base):
    class Config:
        extra = "forbid"
    keep: int
    gone: str
    retype: int
    to_req: int | None = None
    to_opt: int
    dflt: int = 1
    both: int = 1
    KIND = "a"
    DROP = 1
    @field_validator("keep")
    def check_keep(cls, v):
        return v
"""

RICH_NEW = """
from pydantic import BaseModel, Field, field_validator
class Base(BaseModel):
    pass
class Mixin(BaseModel):
    pass
class Rec(Mixin):
    class Config:
        extra = "allow"
    keep: int
    retype: str
    to_req: int | None
    to_opt: int = 0
    dflt: int = 2
    both: str = 2
    new_req: float
    new_opt: float = 0.0
    KIND = "b"
    ADD = 2
    @field_validator("keep")
    def check_keep(cls, v):
        return v + 1
"""


def test_every_reason_in_source_order() -> None:
    """One class touching every branch: the reasons come in the order the checks run,
    names sorted within each group; a field that changes type AND default gets both
    reasons; one breaking reason makes the whole symbol breaking.
    """
    assert _verdict(RICH_OLD, RICH_NEW, "Rec") == (
        B,
        "modified",
        [
            "base classes changed: ['Base'] -> ['Mixin']",
            "model Config changed",
            "added optional field 'new_opt'",
            "added required field 'new_req'",
            "removed field 'gone'",
            "field 'both' type changed: int -> str",
            "field 'both' default changed",
            "field 'dflt' default changed",
            "field 'retype' type changed: int -> str",
            "field 'to_opt' became optional",
            "field 'to_req' became required",
            "added member/const 'ADD'",
            "removed member/const 'DROP'",
            "member/const 'KIND' value changed",
            "validator/serializer changed: ['check_keep']",
        ],
    )


def test_stale_only_reasons_give_stale() -> None:
    """``_max_kind`` is breaking only if some reason is breaking."""
    old = RICH_OLD.replace("    gone: str\n", "")
    new = old.replace(
        "    dflt: int = 1\n", "    dflt: int = 5\n    extra_opt: int = 0\n"
    )
    assert _verdict(old, new, "Rec") == (
        S,
        "modified",
        ["added optional field 'extra_opt'", "field 'dflt' default changed"],
    )


def test_classify_equal_specs_falls_back_to_fingerprint_reason() -> None:
    """Line 180: with no structural difference the verdict is conservative stale. Through
    ``diff_surfaces`` this is unreachable (equal specs give equal fingerprints), so the
    fallback is exercised directly.
    """
    spec = _surface(SCHEMA).specs["Media"]
    assert si.classify_change(spec, spec) == (S, ["contract fingerprint changed"])


def _write(path: Path, imports: str) -> Path:
    path.write_text(
        f"from torchcell.datamodels.schema import {imports}\nclass MyDataset:\n    pass\n"
    )
    return path


def test_map_impacts_scopes_by_closure(tmp_path: Path) -> None:
    # a breaking change to Media flags the Environment importer, not a MeasurementType-only importer
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
    old_surface, new_surface = _surface(SCHEMA), _surface(new)
    env_loader = _write(tmp_path / "env_loader.py", "Environment")
    mt_loader = _write(tmp_path / "mt_loader.py", "MeasurementType")
    changes = si.diff_surfaces(old_surface, new_surface)
    impacts = si.map_impacts(
        changes, [env_loader, mt_loader], old_surface, new_surface, tmp_path
    )
    assert impacts == [
        si.LoaderImpact(
            loader="env_loader.py",
            dataset_classes=["MyDataset"],
            changed_symbols=["Media"],
            kind=B,
        )
    ]


def test_map_impacts_flags_importer_of_removed_symbol(tmp_path: Path) -> None:
    # removing a symbol must still flag a loader that imported it (old-closure union)
    new = SCHEMA.replace(
        "class FitnessExperiment(ModelStrict):\n    env: Environment\n    fitness: float\n",
        "",
    )
    old_surface, new_surface = _surface(SCHEMA), _surface(new)
    loader = _write(tmp_path / "fit_loader.py", "FitnessExperiment")
    changes = si.diff_surfaces(old_surface, new_surface)
    impacts = si.map_impacts(changes, [loader], old_surface, new_surface, tmp_path)
    assert impacts == [
        si.LoaderImpact(
            loader="fit_loader.py",
            dataset_classes=["MyDataset"],
            changed_symbols=["FitnessExperiment"],
            kind=B,
        )
    ]


def test_map_impacts_with_no_changes_reads_no_loader(tmp_path: Path) -> None:
    """An empty change list returns before any loader is parsed: a path that does not
    exist would raise ``FileNotFoundError`` if it were read.
    """
    surface = _surface(SCHEMA)
    missing = tmp_path / "does_not_exist.py"
    assert si.map_impacts([], [missing], surface, surface, tmp_path) == []


def test_map_impacts_orders_breaking_first_then_by_path(tmp_path: Path) -> None:
    """Stale ``a_mt.py`` sorts after breaking ``z_env.py`` despite the path order; a
    loader hit by both kinds takes the max (breaking) and lists both symbols sorted.
    """
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n").replace(
        '    b = "b"\n', '    b = "b"\n    c = "c"\n'
    )
    old_surface, new_surface = _surface(SCHEMA), _surface(new)
    a_mt = _write(tmp_path / "a_mt.py", "MeasurementType")
    m_both = _write(tmp_path / "m_both.py", "Media, MeasurementType")
    z_env = _write(tmp_path / "z_env.py", "Environment")
    changes = si.diff_surfaces(old_surface, new_surface)
    impacts = si.map_impacts(
        changes, [a_mt, m_both, z_env], old_surface, new_surface, tmp_path
    )
    assert [(i.loader, i.changed_symbols, i.kind) for i in impacts] == [
        ("m_both.py", ["MeasurementType", "Media"], B),
        ("z_env.py", ["Media"], B),
        ("a_mt.py", ["MeasurementType"], S),
    ]


def test_dataset_classes_detection(tmp_path: Path) -> None:
    """Bare ``@register_dataset``, attribute ``@x.register_dataset`` and a ``*Dataset``
    name count; nested classes and functions do not.

    Finding: the call form ``@register_dataset(...)`` is a ``Call`` node, which the
    Name/Attribute check at lines 227 to 228 does not match, so ``CallForm`` is not
    listed. Every loader in the repo uses the bare form today. Pinned until the call
    form is recognized.
    """
    path = tmp_path / "loader.py"
    path.write_text(
        "import registry\n"
        "@register_dataset\nclass Bare:\n    pass\n"
        "@registry.register_dataset\nclass Dotted:\n    pass\n"
        "@register_dataset('x')\nclass CallForm:\n    pass\n"
        "class Helper:\n    class InnerDataset:\n        pass\n"
        "class PlainDataset:\n    pass\n"
        "def FuncDataset():\n    pass\n"
    )
    assert si._dataset_classes(path) == ["Bare", "Dotted", "PlainDataset"]


def test_loader_paths_skip_dunder_and_deprecated(tmp_path: Path) -> None:
    datasets = tmp_path / "torchcell" / "datasets"
    (datasets / "sub").mkdir(parents=True)
    for rel in [
        "__init__.py",
        "b.py",
        "a_deprecated.py",
        "deprecated_old.py",
        "sub/__init__.py",
        "sub/c.py",
        "notes.txt",
    ]:
        (datasets / rel).write_text("")
    assert si._loader_paths(tmp_path) == [datasets / "b.py", datasets / "sub" / "c.py"]


def test_report_properties_judge_loaders_not_symbols() -> None:
    """``has_breaking`` looks only at impacted loaders: a breaking symbol change with no
    loader impacted is neither breaking nor an impact.
    """
    breaking_symbol = si.SymbolChange(
        symbol="Orphan", kind=B, status="removed", reasons=["symbol removed"]
    )
    orphan = si.ImpactReport(base="HEAD", changed_symbols=[breaking_symbol])
    assert (orphan.has_impact, orphan.has_breaking) == (False, False)
    stale_loader = si.LoaderImpact(
        loader="x.py", dataset_classes=[], changed_symbols=["Media"], kind=S
    )
    stale = si.ImpactReport(base="HEAD", impacted_loaders=[stale_loader])
    assert (stale.has_impact, stale.has_breaking) == (True, False)
    both = si.ImpactReport(
        base="HEAD",
        impacted_loaders=[stale_loader, stale_loader.model_copy(update={"kind": B})],
    )
    assert (both.has_impact, both.has_breaking) == (True, True)


# ---------------------------------------------------------------- end to end on git


_REAL_REPO_ROOT = si._repo_root  # the fixture below patches the module attribute


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()  # fmt: skip


LOADERS = {
    "torchcell/datasets/__init__.py": "from torchcell.datamodels.schema import Media\n",
    "torchcell/datasets/old_deprecated.py": (
        "from torchcell.datamodels.schema import Media\nclass OldDataset:\n    pass\n"
    ),
    "torchcell/datasets/fit_loader.py": (
        "from torchcell.datamodels.schema import FitnessExperiment\n"
        "import torchcell.data as registry\n"
        "@registry.register_dataset\nclass FitLoad:\n    pass\n"
    ),
    "torchcell/datasets/mt_loader.py": (
        "from torchcell.datamodels.schema import MeasurementType\nX = MeasurementType\n"
    ),
    "torchcell/datasets/plain.py": "import os\nclass PlainDataset:\n    pass\n",
    "torchcell/datasets/scerevisiae/env_loader.py": (
        "from torchcell.datamodels.schema import Environment\n"
        "@register_dataset\nclass EnvScreen:\n    pass\n"
        "class EnvHelper:\n    pass\n"
        "class OtherEnvDataset:\n    pass\n"
    ),
}

BREAKING_SCHEMA = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
STALE_SCHEMA = SCHEMA.replace('    b = "b"\n', '    b = "b"\n    c = "c"\n')

REPORT_BREAKING_AND_STALE = """Schema impact vs HEAD

Changed symbols (2):
  [stale] MeasurementType (modified)
      - added member/const 'c'
  [BREAKING] Media (modified)
      - added required field 'ph'

Impacted datasets (3; 2 breaking) -> rebuild:
  [BREAKING] FitLoad  via Media
  [BREAKING] EnvScreen, OtherEnvDataset  via Media
  [stale] torchcell/datasets/mt_loader.py  via MeasurementType"""

BREAKING_TAIL = (
    "\nBREAKING schema change: update + rebuild the datasets above, "
    "then re-run with TORCHCELL_SCHEMA_ACK=1 to acknowledge.\n"
)
ACK_TAIL = "\nTORCHCELL_SCHEMA_ACK set -> impact acknowledged; proceeding.\n"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo whose HEAD holds SCHEMA; the gate is pointed at its surface.

    ``pydant.py`` is written after the commit (with no classes), so it exists in the
    working tree, which ``load_surface`` reads, but not at HEAD, where ``_git_show``
    returns "" for it.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("TORCHCELL_SCHEMA_ACK", raising=False)
    root = tmp_path / "repo"
    datamodels = root / "torchcell" / "datamodels"
    datamodels.mkdir(parents=True)
    (datamodels / "schema.py").write_text(SCHEMA)
    for rel, text in LOADERS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    (datamodels / "pydant.py").write_text("VERSION = 1\n")
    monkeypatch.setattr(
        si,
        "default_surface_modules",
        lambda: [datamodels / "schema.py", datamodels / "pydant.py"],
    )
    monkeypatch.setattr(si, "_repo_root", lambda: root)
    return root


def _edit_schema(repo: Path, text: str) -> None:
    (repo / "torchcell" / "datamodels" / "schema.py").write_text(text)


def test_git_show_reads_a_ref_and_returns_empty_when_absent(repo: Path) -> None:
    """HEAD's committed text, not the working tree; "" for a path missing at the ref
    and for an unknown ref (nonzero git exit).
    """
    _edit_schema(repo, BREAKING_SCHEMA)
    assert si._git_show("HEAD", "torchcell/datamodels/schema.py", repo) == SCHEMA
    assert si._git_show("HEAD", "torchcell/datamodels/pydant.py", repo) == ""
    assert si._git_show("no-such-ref", "torchcell/datamodels/schema.py", repo) == ""


def test_build_impact_report_end_to_end(repo: Path) -> None:
    """The committed schema is the old surface; the working tree is the new one."""
    _edit_schema(
        repo, BREAKING_SCHEMA.replace('    b = "b"\n', '    b = "b"\n    c = "c"\n')
    )
    report = si.build_impact_report("HEAD", repo)
    assert report == si.ImpactReport(
        base="HEAD",
        changed_symbols=[
            si.SymbolChange(
                symbol="MeasurementType",
                kind=S,
                status="modified",
                reasons=["added member/const 'c'"],
            ),
            si.SymbolChange(
                symbol="Media",
                kind=B,
                status="modified",
                reasons=["added required field 'ph'"],
            ),
        ],
        impacted_loaders=[
            si.LoaderImpact(
                loader="torchcell/datasets/fit_loader.py",
                dataset_classes=["FitLoad"],
                changed_symbols=["Media"],
                kind=B,
            ),
            si.LoaderImpact(
                loader="torchcell/datasets/scerevisiae/env_loader.py",
                dataset_classes=["EnvScreen", "OtherEnvDataset"],
                changed_symbols=["Media"],
                kind=B,
            ),
            si.LoaderImpact(
                loader="torchcell/datasets/mt_loader.py",
                dataset_classes=[],
                changed_symbols=["MeasurementType"],
                kind=S,
            ),
        ],
    )
    assert si.format_report(report) == REPORT_BREAKING_AND_STALE


def test_symbol_new_in_working_tree_is_added(repo: Path) -> None:
    """``pydant.py`` does not exist at HEAD, so a class defined there is "added"."""
    (repo / "torchcell" / "datamodels" / "pydant.py").write_text(
        "class Base:\n    x: int\n"
    )
    report = si.build_impact_report("HEAD", repo)
    assert report.changed_symbols == [
        si.SymbolChange(symbol="Base", kind=S, status="added", reasons=["new symbol"])
    ]
    assert report.impacted_loaders == []


def test_format_report_without_changes() -> None:
    assert (
        si.format_report(si.ImpactReport(base="abc123"))
        == "No schema contract changes vs abc123."
    )


def test_main_clean_tree_exits_zero(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert si.main([]) == 0
    assert capsys.readouterr().out == "No schema contract changes vs HEAD.\n"


def test_main_breaking_exits_one(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Staged filenames passed by pre-commit are accepted and ignored."""
    _edit_schema(
        repo, BREAKING_SCHEMA.replace('    b = "b"\n', '    b = "b"\n    c = "c"\n')
    )
    assert si.main(["torchcell/datamodels/schema.py", "README.md"]) == 1
    assert capsys.readouterr().out == REPORT_BREAKING_AND_STALE + "\n" + BREAKING_TAIL


def test_main_ack_env_acknowledges(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``TORCHCELL_SCHEMA_ACK=1`` turns the breaking exit into 0 with the ack line; an
    empty value does not acknowledge.

    Finding: the check is ``os.environ.get(...)`` truthiness (line 372), so ``=0`` also
    acknowledges. Pinned until the variable is parsed as a boolean.
    """
    _edit_schema(repo, BREAKING_SCHEMA)
    monkeypatch.setenv("TORCHCELL_SCHEMA_ACK", "")
    assert si.main([]) == 1
    capsys.readouterr()
    for value in ("1", "0"):
        monkeypatch.setenv("TORCHCELL_SCHEMA_ACK", value)
        assert si.main([]) == 0
        assert capsys.readouterr().out.endswith("\n" + ACK_TAIL)


def test_main_stale_only_passes_unless_strict(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stale-only impact: exit 0 by default, 1 under ``--strict`` (no message), and the
    acknowledgment overrides ``--strict`` too.
    """
    _edit_schema(repo, STALE_SCHEMA)
    report = (
        "Schema impact vs HEAD\n\nChanged symbols (1):\n"
        "  [stale] MeasurementType (modified)\n"
        "      - added member/const 'c'\n\n"
        "Impacted datasets (1; 0 breaking) -> rebuild:\n"
        "  [stale] torchcell/datasets/mt_loader.py  via MeasurementType\n"
    )
    assert si.main([]) == 0
    assert capsys.readouterr().out == report
    assert si.main(["--strict"]) == 1
    assert capsys.readouterr().out == report
    monkeypatch.setenv("TORCHCELL_SCHEMA_ACK", "1")
    assert si.main(["--strict"]) == 0
    assert capsys.readouterr().out == report + ACK_TAIL


def test_main_breaking_symbol_without_loader_exits_zero(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing a class no loader reaches is a breaking SYMBOL change but no impact, so
    the gate passes (the docstring's exit 1 is for breaking impact on a dataset).
    """
    _edit_schema(repo, SCHEMA + "\n\nclass Orphan(ModelStrict):\n    x: int\n")
    _git(repo, "commit", "-q", "-am", "orphan")
    _edit_schema(repo, SCHEMA)
    assert si.main([]) == 0
    assert capsys.readouterr().out == (
        "Schema impact vs HEAD\n\nChanged symbols (1):\n"
        "  [BREAKING] Orphan (removed)\n"
        "      - symbol removed\n\n"
        "Impacted datasets: none (no built loader depends on the changed symbols).\n"
    )


def test_main_base_selects_the_ref(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Commit the breaking schema, tag the first commit: the clean tree has no change
    vs HEAD but the full breaking report vs the tag.
    """
    _git(repo, "tag", "base0")
    _edit_schema(repo, BREAKING_SCHEMA)
    _git(repo, "commit", "-q", "-am", "ph")
    assert si.main([]) == 0
    assert capsys.readouterr().out == "No schema contract changes vs HEAD.\n"
    assert si.main(["--base", "base0"]) == 1
    assert capsys.readouterr().out == (
        "Schema impact vs base0\n\nChanged symbols (1):\n"
        "  [BREAKING] Media (modified)\n"
        "      - added required field 'ph'\n\n"
        "Impacted datasets (2; 2 breaking) -> rebuild:\n"
        "  [BREAKING] FitLoad  via Media\n"
        "  [BREAKING] EnvScreen, OtherEnvDataset  via Media\n" + BREAKING_TAIL
    )


def test_repo_root_inside_and_outside_a_repo(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From a subdirectory: the toplevel. Outside any repo: the current directory."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.chdir(repo / "torchcell" / "datasets")
    assert _REAL_REPO_ROOT() == repo.resolve()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert _REAL_REPO_ROOT() == outside.resolve()


# 2026.10.08, issue #734: changes to module-level vocabularies a class resolves through.
VOCAB = """
import re
from typing import Literal

Namespace = Literal["mg1655", "bw25113"]
PATTERNS = {"mg1655": "^b[0-9]{4}$", "bw25113": "^BW25113_[0-9]{4}$"}


def _validate_tag(value):
    return any(re.match(p, value) for p in PATTERNS.values())


class Leaf(ModelStrict):
    namespace: Namespace
    tag: str

    @field_validator("tag")
    def _tag(cls, value):
        return _validate_tag(value)
"""


def test_literal_and_pattern_widening_is_stale() -> None:
    """The REL606 shape: one namespace and its pattern added, nothing removed."""
    new = VOCAB.replace('"bw25113"]', '"bw25113", "rel606"]').replace(
        '"^BW25113_[0-9]{4}$"}', '"^BW25113_[0-9]{4}$", "rel606": "^ECB_[0-9]{5}$"}'
    )
    assert _verdict(VOCAB, new, "Leaf") == (
        S,
        "modified",
        [
            "module-level 'Namespace' gained members [\"'rel606'\"]",
            "module-level 'PATTERNS' gained members [\"'rel606': '^ECB_[0-9]{5}$'\"]",
        ],
    )


def test_literal_narrowing_is_breaking() -> None:
    new = VOCAB.replace('Literal["mg1655", "bw25113"]', 'Literal["mg1655"]')
    assert _verdict(VOCAB, new, "Leaf") == (
        B,
        "modified",
        ["module-level 'Namespace' lost members [\"'bw25113'\"]"],
    )


def test_changed_pattern_is_breaking() -> None:
    new = VOCAB.replace('"^b[0-9]{4}$"', '"^b[0-9]{5}$"')
    assert _verdict(VOCAB, new, "Leaf") == (
        B,
        "modified",
        [
            "module-level 'PATTERNS' lost members [\"'mg1655': '^b[0-9]{4}$'\"]",
            "module-level 'PATTERNS' gained members [\"'mg1655': '^b[0-9]{5}$'\"]",
        ],
    )


def test_changed_helper_function_is_breaking() -> None:
    new = VOCAB.replace("return any(", "return all(")
    assert _verdict(VOCAB, new, "Leaf") == (
        B,
        "modified",
        ["module-level '_validate_tag' changed"],
    )


def test_reordered_literal_is_stale() -> None:
    new = VOCAB.replace('Literal["mg1655", "bw25113"]', 'Literal["bw25113", "mg1655"]')
    assert _verdict(VOCAB, new, "Leaf") == (
        S,
        "modified",
        ["module-level 'Namespace' reordered"],
    )


def test_newly_reached_and_dropped_bindings_are_reported() -> None:
    """Retargeting a field from the alias to ``str`` drops the binding (and changes the
    type, which is what makes it breaking); the reverse adds it.
    """
    plain = VOCAB.replace("    namespace: Namespace\n", "    namespace: str\n")
    kind, _, reasons = _verdict(VOCAB, plain, "Leaf")
    assert kind == B
    assert "no longer resolves through module-level 'Namespace'" in reasons
    kind, _, reasons = _verdict(plain, VOCAB, "Leaf")
    assert "now resolves through module-level 'Namespace'" in reasons


def test_vocabulary_change_flags_the_loaders_that_reach_the_class(
    tmp_path: Path,
) -> None:
    """A narrowed Literal maps to the loader importing the leaf, as an enum change would."""
    old_surface = _surface(VOCAB)
    new_surface = _surface(
        VOCAB.replace('Literal["mg1655", "bw25113"]', 'Literal["mg1655"]')
    )
    leaf_loader = tmp_path / "leaf_loader.py"
    leaf_loader.write_text("from torchcell.datamodels.schema import Leaf\n")
    other = tmp_path / "other.py"
    other.write_text("from torchcell.datamodels.schema import ModelStrict\n")
    changes = si.diff_surfaces(old_surface, new_surface)
    assert [c.symbol for c in changes] == ["Leaf"]
    impacts = si.map_impacts(
        changes, [leaf_loader, other], old_surface, new_surface, tmp_path
    )
    assert [(i.loader, i.changed_symbols, i.kind) for i in impacts] == [
        ("leaf_loader.py", ["Leaf"], B)
    ]
