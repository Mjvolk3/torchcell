# tests/torchcell/provenance/test_schema_deps.py
# [[tests.torchcell.provenance.test_schema_deps]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/provenance/test_schema_deps.py
r"""Tests for schema_deps: contract fingerprint stability and dependency closure.

2026.09.30, Phase 16: the canonical text and the field folding pinned exactly on
single-class sources. ``class M(Base): x: int`` canonicalizes to
``"bases::Base\nfield::x::int::<REQUIRED>"``, whose SHA-256 is
``ed8d43cc...f641``; the literal is pinned because a manifest written on one machine is
compared against this value computed on another. ``_FOLDING`` exercises every branch of
``_field_default``: a positional default ``Field(0, ge=0, description=...)`` folds to
``Field(default=0,ge=0)`` (description dropped, kwargs sorted); ``Field(..., ge=1)`` is
``<REQUIRED>|Field(ge=1)``; ``default_factory`` counts as a default; a ``**opts`` splat is
kept verbatim and makes the field required; ``Field(default=..., le=5)`` keeps the literal
``default=...`` inside a required marker; ``pydantic.Field(3)`` folds like ``Field``; a
plain value is its ``ast.unparse`` text (``'x'`` in single quotes). The mini ``SCHEMA``
below has the reference graph Media -> {ModelStrict}, Environment -> {ModelStrict, Media,
MeasurementType}, FitnessExperiment -> {ModelStrict, Environment}, and nothing out of
ModelStrict or MeasurementType (``str``, ``Enum`` and the nested ``Config`` are not
top-level surface classes).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
    assert reach == {
        "FitnessExperiment",
        "Environment",
        "Media",
        "MeasurementType",
        "ModelStrict",
    }
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
    assert closure == {"Environment", "Media", "MeasurementType", "ModelStrict"}


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


_FOLDING = """
class F(pydantic.BaseModel, Mixin):
    a: int = Field(0, ge=0, description="x")
    b: int = Field(..., ge=1)
    c: list[int] = Field(default_factory=list, title="t")
    d: int = Field(**opts)
    e: int = Field(default=..., le=5)
    f: int = pydantic.Field(3)
    g: int = 5
    h: str = "x"
    j: int = Field(le=5, ge=0)
    A = B = "v"
    p, q = 1, 2

    @pydantic.field_validator("a")
    def va(cls, value):
        \"\"\"doc\"\"\"

    @registry["k"]
    def weird(self):
        return 1

    @staticmethod
    def plain():
        return 2

    class Config:
        extra = "forbid"
        a = b = 1
        x.y = 2

        def f(self):
            pass
"""


def _cls(src: str) -> ast.ClassDef:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.ClassDef)
    return node


def test_canonical_text_and_sha256_of_a_one_field_class() -> None:
    """The canonical form is one line per part; the fingerprint is its SHA-256 hex."""
    cls = _cls("class M(Base):\n    x: int\n")
    spec = sd.contract_spec(cls)
    assert spec.canonical() == "bases::Base\nfield::x::int::<REQUIRED>"
    assert (
        sd.fingerprint(cls)
        == "ed8d43cc85679c4e4398592efa1795d614a8e1394a47d568bb2aa2099f65f641"
    )
    assert sd.spec_fingerprint(spec) == sd.fingerprint(cls)


def test_field_defaults_fold_documentation_out_and_keep_semantics() -> None:
    """Each ``Field`` form maps to one exact default string and required flag."""
    fields = dict(sd.contract_spec(_cls(_FOLDING)).fields)
    folded = {name: (spec.default, spec.required) for name, spec in fields.items()}
    assert folded == {
        "a": ("Field(default=0,ge=0)", False),
        "b": ("<REQUIRED>|Field(ge=1)", True),
        "c": ("Field(default_factory=list)", False),
        "d": ("<REQUIRED>|Field(**opts)", True),
        "e": ("<REQUIRED>|Field(default=...,le=5)", True),
        "f": ("Field(default=3)", False),
        "g": ("5", False),
        "h": ("'x'", False),
        "j": ("<REQUIRED>|Field(ge=0,le=5)", True),
    }
    assert fields["c"].annotation == "list[int]"


def test_assigns_contract_methods_and_config_are_extracted_exactly() -> None:
    """Chained assigns give one entry per name, a tuple target is skipped; only the
    validator (attribute-form decorator) is a contract method, docstring replaced by
    ``pass``; the nested Config keeps name assignments only, sorted.
    """
    spec = sd.contract_spec(_cls(_FOLDING))
    assert spec.bases == ("pydantic.BaseModel", "Mixin")
    assert spec.assigns == (("A", "'v'"), ("B", "'v'"))
    assert spec.methods == (
        ("va", "@pydantic.field_validator('a')\ndef va(cls, value):\n    pass"),
    )
    assert spec.config == "a=1,b=1,extra='forbid'"
    assert spec.canonical().splitlines()[0] == "bases::Mixin,pydantic.BaseModel"
    assert spec.canonical().splitlines()[-1] == "config::a=1,b=1,extra='forbid'"


def test_positional_and_keyword_defaults_and_kwarg_order_share_a_fingerprint() -> None:
    """``Field(3)`` == ``Field(default=3)`` and ``Field(le=5, ge=0)`` == ``Field(ge=0, le=5)``."""
    one = _cls("class C(B):\n    x: int = Field(3)\n    y: int = Field(le=5, ge=0)\n")
    two = _cls(
        "class C(B):\n    x: int = Field(default=3)\n    y: int = Field(ge=0, le=5)\n"
    )
    assert sd.fingerprint(one) == sd.fingerprint(two)


def test_required_ellipsis_positional_and_keyword_fingerprint_differently() -> None:
    """Finding: ``Field(..., le=5)`` and ``Field(default=..., le=5)`` mean the same
    required field but fold to ``<REQUIRED>|Field(le=5)`` and
    ``<REQUIRED>|Field(default=...,le=5)`` (``schema_deps.py:163-178``: the positional
    ellipsis is dropped, the keyword one is kept), so rewriting one as the other moves
    the fingerprint. Pinned until the keyword ellipsis is dropped too.
    """
    positional = _cls("class C(B):\n    x: int = Field(..., le=5)\n")
    keyword = _cls("class C(B):\n    x: int = Field(default=..., le=5)\n")
    assert dict(sd.contract_spec(positional).fields)["x"].default == (
        "<REQUIRED>|Field(le=5)"
    )
    assert sd.fingerprint(positional) != sd.fingerprint(keyword)


def test_base_order_does_not_move_the_fingerprint() -> None:
    """Finding: ``canonical()`` sorts the bases, so ``class C(A, B)`` and ``class C(B, A)``
    share a fingerprint although their MRO, and so which base's field wins, differs
    (``schema_deps.py:118``). Pinned until base order is treated as contract.
    """
    assert sd.fingerprint(_cls("class C(A, B):\n    x: int\n")) == sd.fingerprint(
        _cls("class C(B, A):\n    x: int\n")
    )


def test_a_validator_docstring_edit_keeps_and_a_decorator_argument_edit_moves() -> None:
    """The docstring is stripped from the method contract; the decorator is not."""
    base = _cls(_FOLDING)
    reworded = _cls(_FOLDING.replace('"""doc"""', '"""other words"""'))
    retargeted = _cls(_FOLDING.replace('field_validator("a")', 'field_validator("b")'))
    assert sd.fingerprint(reworded) == sd.fingerprint(base)
    assert sd.fingerprint(retargeted) != sd.fingerprint(base)


def test_canonical_keeps_fields_in_the_order_the_spec_holds_them() -> None:
    """``canonical()`` sorts only the bases; fields, assigns and methods are emitted as
    stored, so a hand-built spec with unsorted fields hashes differently from its sorted
    twin. ``contract_spec`` always sorts, which is what makes field order benign.
    """
    x = sd.FieldSpec(annotation="int", default="<REQUIRED>")
    y = sd.FieldSpec(annotation="str", default="'a'")
    unsorted = sd.ContractSpec(
        bases=("Z", "A"),
        fields=(("y", y), ("x", x)),
        assigns=(),
        methods=(),
        config=None,
    )
    assert unsorted.canonical() == (
        "bases::A,Z\nfield::y::str::'a'\nfield::x::int::<REQUIRED>"
    )
    resorted = sd.ContractSpec(
        bases=("A", "Z"),
        fields=(("x", x), ("y", y)),
        assigns=(),
        methods=(),
        config=None,
    )
    assert sd.spec_fingerprint(unsorted) != sd.spec_fingerprint(resorted)


def test_reference_graph_of_the_mini_schema_is_exact() -> None:
    """The edges are the surface names each class body mentions, self excluded."""
    surface = _surface(SCHEMA)
    assert surface.names == {
        "ModelStrict",
        "MeasurementType",
        "Media",
        "Environment",
        "FitnessExperiment",
    }
    assert surface.ref_graph == {
        "ModelStrict": set(),
        "MeasurementType": set(),
        "Media": {"ModelStrict"},
        "Environment": {"ModelStrict", "Media", "MeasurementType"},
        "FitnessExperiment": {"ModelStrict", "Environment"},
    }


def test_a_duplicate_class_name_is_taken_from_the_last_source() -> None:
    """Finding: class names are assumed globally unique; a second definition silently
    replaces the first in ``specs``, ``fingerprints`` and ``module_of``
    (``schema_deps.py:307-308``). Pinned until a duplicate is refused. An empty source
    contributes nothing.
    """
    surface = sd.load_surface_from_sources(
        {
            "first": "class C(B):\n    x: int\n",
            "second": "class C(B):\n    y: str\n",
            "empty": "",
        }
    )
    assert surface.names == {"C"}
    assert surface.module_of == {"C": "second"}
    assert surface.fingerprints["C"] == sd.fingerprint(
        _cls("class C(B):\n    y: str\n")
    )


def test_load_surface_labels_classes_by_path(tmp_path: Path) -> None:
    """``load_surface`` reads each file and labels its classes with the path string."""
    base = tmp_path / "base.py"
    base.write_text("class B:\n    pass\n")
    rec = tmp_path / "rec.py"
    rec.write_text("class R(B):\n    x: int\n")
    surface = sd.load_surface([base, rec])
    assert surface.module_of == {"B": str(base), "R": str(rec)}
    assert surface.ref_graph == {"B": set(), "R": {"B"}}


def test_default_surface_is_schema_plus_pydant() -> None:
    """The live surface is ``datamodels/schema.py`` then ``datamodels/pydant.py``;
    ``ModelStrict`` comes from pydant with the frozen, extra-forbidding Config, and a
    record class like ``Media`` from schema.
    """
    paths = sd.default_surface_modules()
    assert [(p.parent.name, p.name) for p in paths] == [
        ("datamodels", "schema.py"),
        ("datamodels", "pydant.py"),
    ]
    surface = sd.load_default_surface()
    assert surface.module_of["ModelStrict"] == str(paths[1])
    assert surface.module_of["Media"] == str(paths[0])
    assert surface.specs["ModelStrict"] == sd.ContractSpec(
        bases=("BaseModel",),
        fields=(),
        assigns=(),
        methods=(),
        config="extra='forbid',frozen=True",
    )


_LOADER = """
import torchcell.datamodels.schema
from torchcell.datamodels.schema import Environment, NotOnTheSurface
from torchcell.datamodels import Media as M
from torchcell.datasets.dataset_registry import FitnessExperiment
from .schema import MeasurementType


def build():
    from torchcell.datamodels.schema import FitnessExperiment
"""


def test_loader_deps_from_source_keep_datamodels_imports_of_surface_names() -> None:
    """Environment, the aliased Media and the function-local FitnessExperiment count;
    a non-surface name, a plain ``import``, a non-datamodels module and a relative
    import do not.
    """
    surface = _surface(SCHEMA)
    assert sd.loader_schema_deps_from_source(_LOADER, surface) == {
        "Environment",
        "Media",
        "FitnessExperiment",
    }
    assert sd.loader_closure_from_source(_LOADER, surface) == {
        "Environment",
        "Media",
        "FitnessExperiment",
        "MeasurementType",
        "ModelStrict",
    }


def test_loader_deps_from_a_file_match_the_source_form(tmp_path: Path) -> None:
    """The path readers agree with the text readers on the same loader."""
    loader = tmp_path / "loader.py"
    loader.write_text(_LOADER)
    surface = _surface(SCHEMA)
    assert sd.loader_schema_deps(loader, surface) == {
        "Environment",
        "Media",
        "FitnessExperiment",
    }
    assert sd.loader_closure(loader, surface) == sd.loader_closure_from_source(
        _LOADER, surface
    )


def test_symbol_dependents_is_the_reverse_index_of_the_closures() -> None:
    """A Media-only loader and an Environment loader both depend on Media and
    ModelStrict; only the latter on Environment and MeasurementType.
    """
    surface = _surface(SCHEMA)
    dependents = sd.symbol_dependents(
        {
            "media_only": "from torchcell.datamodels.schema import Media\n",
            "env": "from torchcell.datamodels.schema import Environment\n",
            "none": "import os\n",
        },
        surface,
    )
    assert dependents == {
        "Media": {"media_only", "env"},
        "ModelStrict": {"media_only", "env"},
        "Environment": {"env"},
        "MeasurementType": {"env"},
    }


@pytest.mark.parametrize(
    ("seeds", "expected"),
    [
        ({"A"}, {"A", "B", "C"}),
        ({"C"}, {"C", "A", "B"}),
        ({"D"}, {"D"}),
        (set(), set()),
    ],
)
def test_forward_closure_terminates_on_a_cycle_and_keeps_unknown_seeds(
    seeds: set[str], expected: set[str]
) -> None:
    """A -> B -> C -> A is walked once; a seed outside the graph is its own closure."""
    graph = {"A": {"B"}, "B": {"C"}, "C": {"A"}}
    assert sd.forward_closure(seeds, graph) == expected
