# torchcell/provenance/schema_deps.py
# [[torchcell.provenance.schema_deps]]
"""AST-level schema-dependency analysis for torchcell datasets.

This module answers two questions from LOCAL source only -- no git SHAs, no network,
no central database:

    closure(loader)      which schema symbols does a dataset loader transitively depend on?
    fingerprint(symbol)  a content hash of a symbol's *contract* (field shape + validators),
                         deliberately excluding docstrings, comments, plain methods, field
                         descriptions, and field ORDER.

Because a fingerprint hashes the contract and not the source text, it is machine- and
commit-independent: the fingerprint computed on one machine equals the fingerprint computed
on another whenever the contract is the same. That is what lets a build manifest written on
machine A be checked against machine B's *local* schema (see ``build_manifest.py``): staleness
is judged by ``fingerprint_now(local schema) != fingerprint_stored``, computed entirely from
local git state.

The "schema surface" is the set of modules that define record-contract classes. In torchcell
that is ``schema.py`` (the record classes) plus ``pydant.py`` (the ``ModelStrict`` base every
record inherits). ``media.py`` / ``calmorph_labels.py`` export constant INSTANCES, not new
record types, so they are not part of the class-contract surface.

Two orthogonal scoping levers keep the rebuild signal tight (see the module tests):
  1. closure   -- a loader depends only on the symbols reachable from its imports, so a change
                  to a symbol outside its closure never flags it.
  2. fingerprint -- a benign edit (docstring, method, field description, field reorder) leaves
                  the contract fingerprint unchanged, so it flags nobody even for a universal
                  symbol like ``Media`` that sits under every record.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FieldSpec",
    "ContractSpec",
    "contract_spec",
    "fingerprint",
    "spec_fingerprint",
    "SchemaSurface",
    "load_surface",
    "load_surface_from_sources",
    "default_surface_modules",
    "load_default_surface",
    "forward_closure",
    "loader_schema_deps",
    "loader_closure",
]

# Field(...) keyword arguments that are documentation only: changing them must NOT change a
# symbol's contract fingerprint (they do not affect validation or serialization).
_DOC_FIELD_KWARGS: frozenset[str] = frozenset(
    {"description", "title", "examples", "json_schema_extra", "deprecated"}
)

# Decorators that mark a method as part of the record contract (validation/serialization).
_CONTRACT_METHOD_DECORATORS: frozenset[str] = frozenset(
    {
        "field_validator",
        "model_validator",
        "field_serializer",
        "model_serializer",
        "computed_field",
        "validator",
        "root_validator",
    }
)

# Sentinel prefix marking a required field. A plain ``x: T`` yields exactly this; a required
# field written as ``x: T = Field(...)`` (no default/default_factory) yields this + the folded
# semantic kwargs, so a change to e.g. ``ge=0`` still moves the fingerprint while the field
# stays classified as required.
_REQUIRED = "<REQUIRED>"


@dataclass(frozen=True)
class FieldSpec:
    """A single pydantic field's contract-relevant shape."""

    annotation: str
    default: str  # normalized default source, or a ``_REQUIRED``-prefixed marker

    @property
    def required(self) -> bool:
        """True if the field has no default (a required field)."""
        return self.default.startswith(_REQUIRED)


@dataclass(frozen=True)
class ContractSpec:
    """The contract-relevant surface of one schema class.

    Two classes with equal ``ContractSpec`` serialize/validate identically as far as this
    static analysis can tell, and get the same fingerprint. Docstrings, plain (non-validator)
    methods, comments, field descriptions and field ORDER are excluded by construction.
    """

    bases: tuple[str, ...]
    fields: tuple[tuple[str, FieldSpec], ...]  # sorted by field name
    assigns: tuple[tuple[str, str], ...]  # enum members / class vars, sorted by name
    methods: tuple[
        tuple[str, str], ...
    ]  # validator/serializer contracts, sorted by name
    config: str | None  # nested pydantic-v1 ``class Config`` body, if any

    def canonical(self) -> str:
        """Deterministic, order-insensitive serialization used for the fingerprint."""
        parts: list[str] = ["bases::" + ",".join(sorted(self.bases))]
        parts.extend(
            f"field::{n}::{fs.annotation}::{fs.default}" for n, fs in self.fields
        )
        parts.extend(f"assign::{n}::{v}" for n, v in self.assigns)
        parts.extend(f"method::{n}::{b}" for n, b in self.methods)
        if self.config is not None:
            parts.append(f"config::{self.config}")
        return "\n".join(parts)


def _deco_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _is_field_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Field") or (
        isinstance(func, ast.Attribute) and func.attr == "Field"
    )


def _is_ellipsis(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _field_default(node: ast.expr | None) -> str:
    """Normalize a field default, folding out documentation-only ``Field`` kwargs.

    Required-ness is encoded in the returned string via the ``_REQUIRED`` prefix so a field
    written ``x: T = Field(description=...)`` (no default) is correctly seen as required.
    """
    if node is None:
        return _REQUIRED
    if _is_field_call(node):
        assert isinstance(node, ast.Call)
        has_default = False
        semantic: list[str] = []
        if node.args:
            first = node.args[
                0
            ]  # positional first arg to Field == the default; ``...`` == required
            if not _is_ellipsis(first):
                has_default = True
                semantic.append(f"default={ast.unparse(first)}")
        for kw in node.keywords:
            if kw.arg is None:
                semantic.append("**" + ast.unparse(kw.value))
            elif kw.arg in _DOC_FIELD_KWARGS:
                continue
            else:
                semantic.append(f"{kw.arg}={ast.unparse(kw.value)}")
                if kw.arg == "default" and not _is_ellipsis(kw.value):
                    has_default = True
                elif kw.arg == "default_factory":
                    has_default = True
        inner = "Field(" + ",".join(sorted(semantic)) + ")"
        return inner if has_default else f"{_REQUIRED}|{inner}"
    return ast.unparse(node)


def _assign_target_name(target: ast.expr) -> str | None:
    return target.id if isinstance(target, ast.Name) else None


def _is_contract_method(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(_deco_name(d) in _CONTRACT_METHOD_DECORATORS for d in fn.decorator_list)


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _method_contract(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Normalized source of a validator/serializer: decorators + signature + body, no docstring."""
    clone = copy.deepcopy(fn)
    clone.body = _strip_docstring(clone.body) or [ast.Pass()]
    return ast.unparse(clone)


def _config_contract(cls: ast.ClassDef) -> str:
    lines: list[str] = []
    for node in cls.body:
        if isinstance(node, ast.Assign):
            value = ast.unparse(node.value)
            for target in node.targets:
                name = _assign_target_name(target)
                if name is not None:
                    lines.append(f"{name}={value}")
    return ",".join(sorted(lines))


def contract_spec(cls: ast.ClassDef) -> ContractSpec:
    """Extract the contract-relevant surface of a schema class from its AST."""
    fields: dict[str, FieldSpec] = {}
    assigns: dict[str, str] = {}
    methods: dict[str, str] = {}
    config: str | None = None
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields[node.target.id] = FieldSpec(
                annotation=ast.unparse(node.annotation),
                default=_field_default(node.value),
            )
        elif isinstance(node, ast.Assign):
            value = ast.unparse(node.value)
            for target in node.targets:
                name = _assign_target_name(target)
                if name is not None:
                    assigns[name] = value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_contract_method(node):
                methods[node.name] = _method_contract(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Config":
            config = _config_contract(node)
    return ContractSpec(
        bases=tuple(ast.unparse(b) for b in cls.bases),
        fields=tuple(sorted(fields.items())),
        assigns=tuple(sorted(assigns.items())),
        methods=tuple(sorted(methods.items())),
        config=config,
    )


def spec_fingerprint(spec: ContractSpec) -> str:
    """SHA-256 of a contract spec's canonical serialization."""
    return hashlib.sha256(spec.canonical().encode("utf-8")).hexdigest()


def fingerprint(cls: ast.ClassDef) -> str:
    """Contract fingerprint of a schema class AST node."""
    return spec_fingerprint(contract_spec(cls))


@dataclass
class SchemaSurface:
    """All record-contract classes across the schema-surface modules, with their graph.

    ``specs``/``fingerprints`` are keyed by class name (assumed globally unique across the
    surface, which holds in torchcell). ``ref_graph`` maps a class to the surface classes it
    references (by base class, field annotation, or any Name in its body) -- the edges used
    for the transitive closure.
    """

    specs: dict[str, ContractSpec]
    fingerprints: dict[str, str]
    module_of: dict[str, str]
    ref_graph: dict[str, set[str]]

    @property
    def names(self) -> set[str]:
        """The set of all surface class names."""
        return set(self.specs)


def _referenced_names(cls: ast.ClassDef, names: set[str]) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Name) and node.id in names and node.id != cls.name:
            refs.add(node.id)
    return refs


def load_surface_from_sources(sources: dict[str, str]) -> SchemaSurface:
    """Build a :class:`SchemaSurface` from ``label -> source text`` pairs.

    Used to analyze a schema version that is not on disk -- e.g. ``git show HEAD:...`` output
    for the static impact check -- without writing temp files. A missing/empty source simply
    contributes no classes.
    """
    classdefs: dict[str, ast.ClassDef] = {}
    module_of: dict[str, str] = {}
    for label, source in sources.items():
        for node in ast.parse(source).body:  # top-level classes only
            if isinstance(node, ast.ClassDef):
                classdefs[node.name] = node
                module_of[node.name] = label
    names = set(classdefs)
    specs = {name: contract_spec(classdefs[name]) for name in names}
    return SchemaSurface(
        specs=specs,
        fingerprints={name: spec_fingerprint(specs[name]) for name in names},
        module_of=module_of,
        ref_graph={name: _referenced_names(classdefs[name], names) for name in names},
    )


def load_surface(module_paths: list[Path]) -> SchemaSurface:
    """Parse the schema-surface modules on disk into a :class:`SchemaSurface`."""
    return load_surface_from_sources(
        {str(path): path.read_text(encoding="utf-8") for path in module_paths}
    )


def default_surface_modules() -> list[Path]:
    """The torchcell schema surface: ``schema.py`` + the ``pydant.py`` base module."""
    import torchcell.datamodels as datamodels  # local import: avoid import cost/cycles at load

    root = Path(datamodels.__file__).parent
    return [root / "schema.py", root / "pydant.py"]


def load_default_surface() -> SchemaSurface:
    """Load the default torchcell schema surface (``schema.py`` + ``pydant.py``)."""
    return load_surface(default_surface_modules())


def forward_closure(seeds: set[str], ref_graph: dict[str, set[str]]) -> set[str]:
    """All symbols reachable from ``seeds`` (inclusive) along the reference graph."""
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        current = stack.pop()
        for nxt in ref_graph.get(current, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def loader_schema_deps(loader_path: Path, surface: SchemaSurface) -> set[str]:
    """Surface classes a loader imports directly from any ``torchcell.datamodels`` submodule."""
    tree = ast.parse(loader_path.read_text(encoding="utf-8"))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("torchcell.datamodels")
        ):
            for alias in node.names:
                if alias.name in surface.specs:
                    deps.add(alias.name)
    return deps


def loader_closure(loader_path: Path, surface: SchemaSurface) -> set[str]:
    """Transitive closure of a loader's schema imports -- the symbols it actually depends on."""
    return forward_closure(loader_schema_deps(loader_path, surface), surface.ref_graph)
