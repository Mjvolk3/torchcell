# tests/torchcell/provenance/test_schema_deps.py
"""Tests for schema_deps: contract fingerprint stability and dependency closure."""

from __future__ import annotations

from pathlib import Path

from torchcell.provenance import schema_deps as sd

# A miniature schema surface exercising the constructs the real schema uses: a ModelStrict base
# with a nested Config, Field(...) defaults/required, a validator, nested references, an Enum.
SCHEMA = '''
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ModelStrict(BaseModel):
    class Config:
        extra = "forbid"


class MeasurementType(str, Enum):
    a = "a"
    b = "b"


class Media(ModelStrict):
    """Media docstring."""

    name: str
    state: str
    is_synthetic: bool = Field(description="chemically defined by construction")
    base_medium: str | None = Field(default=None, description="canonical base label")

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: str) -> str:
        return value

    def to_label(self) -> str:
        return self.name


class Environment(ModelStrict):
    media: Media
    temperature: float
    measurement: MeasurementType


class FitnessExperiment(ModelStrict):
    env: Environment
    fitness: float
'''


def _surface(src: str) -> sd.SchemaSurface:
    return sd.load_surface_from_sources({"schema": src})


def test_fingerprint_is_deterministic() -> None:
    assert (
        _surface(SCHEMA).fingerprints["Media"] == _surface(SCHEMA).fingerprints["Media"]
    )


def test_docstring_change_does_not_move_fingerprint() -> None:
    changed = SCHEMA.replace("Media docstring.", "Completely different wording here.")
    assert (
        _surface(changed).fingerprints["Media"]
        == _surface(SCHEMA).fingerprints["Media"]
    )


def test_field_description_change_does_not_move_fingerprint() -> None:
    changed = SCHEMA.replace(
        'description="chemically defined by construction"',
        'description="reworded prose"',
    )
    assert (
        _surface(changed).fingerprints["Media"]
        == _surface(SCHEMA).fingerprints["Media"]
    )


def test_adding_a_plain_method_does_not_move_fingerprint() -> None:
    changed = SCHEMA.replace(
        "    def to_label(self) -> str:\n        return self.name\n",
        "    def to_label(self) -> str:\n        return self.name\n\n"
        "    def extra_helper(self) -> int:\n        return 1\n",
    )
    assert (
        _surface(changed).fingerprints["Media"]
        == _surface(SCHEMA).fingerprints["Media"]
    )


def test_field_reorder_does_not_move_fingerprint() -> None:
    changed = SCHEMA.replace(
        "    name: str\n    state: str\n", "    state: str\n    name: str\n"
    )
    assert (
        _surface(changed).fingerprints["Media"]
        == _surface(SCHEMA).fingerprints["Media"]
    )


def test_adding_required_field_moves_fingerprint() -> None:
    changed = SCHEMA.replace("    name: str\n", "    name: str\n    ph: float\n")
    assert (
        _surface(changed).fingerprints["Media"]
        != _surface(SCHEMA).fingerprints["Media"]
    )


def test_default_change_moves_fingerprint() -> None:
    changed = SCHEMA.replace("default=None", "default='YPD'")
    assert (
        _surface(changed).fingerprints["Media"]
        != _surface(SCHEMA).fingerprints["Media"]
    )


def test_validator_body_change_moves_fingerprint() -> None:
    changed = SCHEMA.replace("        return value\n", "        return value.upper()\n")
    assert (
        _surface(changed).fingerprints["Media"]
        != _surface(SCHEMA).fingerprints["Media"]
    )


def test_required_detection_through_field_call() -> None:
    fields = dict(_surface(SCHEMA).specs["Media"].fields)
    # Field(description=...) with no default -> REQUIRED (the is_synthetic case)
    assert fields["is_synthetic"].required is True
    # Field(default=None) -> optional
    assert fields["base_medium"].required is False
    # bare annotation -> required
    assert fields["name"].required is True


def test_forward_closure_scoping() -> None:
    surface = _surface(SCHEMA)
    reach = sd.forward_closure({"FitnessExperiment"}, surface.ref_graph)
    assert {"FitnessExperiment", "Environment", "Media"} <= reach
    # Media is downstream of Environment/FitnessExperiment, not upstream
    assert "FitnessExperiment" not in sd.forward_closure({"Media"}, surface.ref_graph)


def test_base_class_is_a_graph_node() -> None:
    # every record inherits ModelStrict, so it must sit in each class's closure
    surface = _surface(SCHEMA)
    assert "ModelStrict" in sd.forward_closure({"Media"}, surface.ref_graph)


def test_loader_deps_and_closure(tmp_path: Path) -> None:
    loader = tmp_path / "loader.py"
    loader.write_text(
        "from torchcell.datamodels.schema import Environment\n"
        "from torchcell.datasets.dataset_registry import register_dataset\n"
        "class XDataset:\n    pass\n"
    )
    surface = _surface(SCHEMA)
    assert sd.loader_schema_deps(loader, surface) == {"Environment"}
    closure = sd.loader_closure(loader, surface)
    assert {"Environment", "Media", "MeasurementType"} <= closure


def test_media_change_flags_environment_importer_but_not_a_metabolite_only_importer(
    tmp_path: Path,
) -> None:
    # a loader importing only a symbol outside Media's dependents is never in Media's blast radius
    env_loader = tmp_path / "env_loader.py"
    env_loader.write_text("from torchcell.datamodels.schema import Environment\n")
    mt_loader = tmp_path / "mt_loader.py"
    mt_loader.write_text("from torchcell.datamodels.schema import MeasurementType\n")
    surface = _surface(SCHEMA)
    assert "Media" in sd.loader_closure(env_loader, surface)
    assert "Media" not in sd.loader_closure(mt_loader, surface)
