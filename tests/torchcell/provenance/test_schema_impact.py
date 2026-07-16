# tests/torchcell/provenance/test_schema_impact.py
"""Tests for schema_impact: breaking/stale classification and loader impact mapping."""

from __future__ import annotations

from pathlib import Path

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


def test_added_required_field_is_breaking() -> None:
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.breaking
    assert any("required field 'ph'" in reason for reason in change.reasons)


def test_added_optional_field_is_stale() -> None:
    new = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float = 7.0\n")
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.stale


def test_removed_field_is_breaking() -> None:
    new = SCHEMA.replace("    state: str\n", "")
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_type_change_is_breaking() -> None:
    new = SCHEMA.replace("    temperature: float\n", "    temperature: int\n")
    change = _change(SCHEMA, new, "Environment")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_optional_to_required_is_breaking() -> None:
    new = SCHEMA.replace(
        '    base_medium: str | None = Field(default=None, description="label")\n',
        "    base_medium: str\n",
    )
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_required_to_optional_is_stale() -> None:
    new = SCHEMA.replace("    state: str\n", "    state: str = 'liquid'\n")
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.stale


def test_default_change_is_stale() -> None:
    new = SCHEMA.replace("default=None", "default='YPD'")
    change = _change(SCHEMA, new, "Media")
    assert change is not None and change.kind == si.ChangeKind.stale


def test_added_enum_member_is_stale() -> None:
    new = SCHEMA.replace('    b = "b"\n', '    b = "b"\n    c = "c"\n')
    change = _change(SCHEMA, new, "MeasurementType")
    assert change is not None and change.kind == si.ChangeKind.stale


def test_removed_enum_member_is_breaking() -> None:
    new = SCHEMA.replace('    b = "b"\n', "")
    change = _change(SCHEMA, new, "MeasurementType")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_enum_value_change_is_breaking() -> None:
    new = SCHEMA.replace('    b = "b"\n', '    b = "beta"\n')
    change = _change(SCHEMA, new, "MeasurementType")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_config_change_is_breaking() -> None:
    new = SCHEMA.replace('extra = "forbid"', 'extra = "allow"')
    change = _change(SCHEMA, new, "ModelStrict")
    assert change is not None and change.kind == si.ChangeKind.breaking


def test_added_symbol_is_not_breaking() -> None:
    new = SCHEMA + "\n\nclass NewThing(ModelStrict):\n    x: int\n"
    change = _change(SCHEMA, new, "NewThing")
    assert change is not None and change.status == "added"
    assert change.kind == si.ChangeKind.stale


def test_removed_symbol_is_breaking() -> None:
    new = SCHEMA.replace(
        "class FitnessExperiment(ModelStrict):\n    env: Environment\n    fitness: float\n",
        "",
    )
    change = _change(SCHEMA, new, "FitnessExperiment")
    assert change is not None and change.status == "removed"
    assert change.kind == si.ChangeKind.breaking


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
    flagged = {impact.loader for impact in impacts}
    assert "env_loader.py" in flagged
    assert "mt_loader.py" not in flagged


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
    assert impacts and impacts[0].kind == si.ChangeKind.breaking
    assert "FitnessExperiment" in impacts[0].changed_symbols
