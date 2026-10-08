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

A class's contract also covers the MODULE-LEVEL names it resolves through (issue #734): a
``Literal`` alias a field is annotated with (``BacterialGeneNamespace``), a pattern map or a
tuple of allowed strings a validator reads (``BACTERIAL_LOCUS_TAG_PATTERNS``), a union alias
(``GenePerturbationType``), and the module-level helper functions a validator calls, followed
transitively (``_validate_bacterial_locus_tag`` -> ``BACTERIAL_LOCUS_TAG_PATTERN`` ->
``BACTERIAL_LOCUS_TAG_PATTERNS``). Their normalized source is folded into the fingerprint of
every class that reaches them, so narrowing a vocabulary or changing a pattern moves the
fingerprint of each class using it and flags every dataset whose closure holds one of those
classes, the same closure semantics an enum gets. A class that reaches no module-level name
keeps the canonical text, and so the fingerprint, it had before this rule existed.

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
import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FieldSpec",
    "ContractSpec",
    "ModuleBinding",
    "contract_spec",
    "fingerprint",
    "spec_fingerprint",
    "collect_module_bindings",
    "class_module_bindings",
    "binding_members",
    "SchemaSurface",
    "load_surface",
    "load_surface_from_sources",
    "default_surface_modules",
    "load_default_surface",
    "forward_closure",
    "loader_schema_deps",
    "loader_closure",
    "loader_schema_deps_from_source",
    "loader_closure_from_source",
    "symbol_dependents",
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
    # Module-level names the class resolves through (Literal aliases, pattern maps, union
    # aliases, helper functions), transitively, as ``name -> normalized source``, sorted by
    # name. Empty for a class that reaches none, which keeps its canonical text unchanged.
    module_bindings: tuple[tuple[str, str], ...] = ()

    def canonical(self) -> str:
        """Deterministic, order-insensitive serialization used for the fingerprint.

        ``module::`` lines are appended last and only when present, so a class with no
        module-level dependency serializes exactly as it did before #734.
        """
        parts: list[str] = ["bases::" + ",".join(sorted(self.bases))]
        parts.extend(
            f"field::{n}::{fs.annotation}::{fs.default}" for n, fs in self.fields
        )
        parts.extend(f"assign::{n}::{v}" for n, v in self.assigns)
        parts.extend(f"method::{n}::{b}" for n, b in self.methods)
        if self.config is not None:
            parts.append(f"config::{self.config}")
        parts.extend(f"module::{n}::{s}" for n, s in self.module_bindings)
        return "\n".join(parts)


@dataclass(frozen=True)
class ModuleBinding:
    """One module-level name of a schema-surface module that is not a class.

    ``source`` is the normalized contract text: ``ast.unparse`` of an assignment's value
    (the annotation of an annotated assignment is a static hint and is left out), or a
    function's decorators + signature + body with its docstring stripped. ``refs`` are the
    other module-level bindings the source mentions, the edges followed transitively.
    """

    source: str
    refs: frozenset[str]


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
    """Contract fingerprint of a schema class AST node, from its body alone.

    A lone ``ClassDef`` carries no module context, so this cannot see the module-level
    names the class resolves through; :func:`load_surface_from_sources` folds those in.
    For a class that reaches none the two agree.
    """
    return spec_fingerprint(contract_spec(cls))


def _binding_nodes(
    tree: ast.Module,
) -> dict[str, ast.expr | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Top-level non-class bindings of a module: name -> the node that defines it.

    A later binding of the same name replaces an earlier one, as it does at runtime.
    Imports, classes, docstrings and ``if __name__ == "__main__"`` blocks bind nothing here.
    """
    nodes: dict[str, ast.expr | ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _assign_target_name(target)
                if name is not None:
                    nodes[name] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            nodes[node.target.id] = node.value
        elif isinstance(node, ast.TypeAlias):
            nodes[node.name.id] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
    return nodes


def collect_module_bindings(
    sources: Mapping[str, str], class_names: set[str]
) -> dict[str, ModuleBinding]:
    """Every top-level non-class binding across the surface sources, with its edges.

    Names are assumed globally unique across the surface, as class names are. A name that
    is also a surface class is left to the class graph.
    """
    nodes: dict[str, ast.expr | ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for source in sources.values():
        nodes.update(_binding_nodes(ast.parse(source)))
    for name in class_names:
        nodes.pop(name, None)
    names = set(nodes)
    bindings: dict[str, ModuleBinding] = {}
    for name, node in nodes.items():
        text = (
            _method_contract(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else ast.unparse(node)
        )
        refs = frozenset(
            sub.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and sub.id in names and sub.id != name
        )
        bindings[name] = ModuleBinding(source=text, refs=refs)
    return bindings


def class_module_bindings(
    cls: ast.ClassDef, bindings: Mapping[str, ModuleBinding]
) -> tuple[tuple[str, str], ...]:
    """The module-level bindings a class reaches, transitively, as sorted ``(name, source)``.

    Seeds are the binding names the class body mentions (annotations, defaults, validator
    bodies); edges are each binding's own references. Classes are not traversed: a class a
    binding names is a graph node with its own fingerprint.
    """
    seen = {
        node.id
        for node in ast.walk(cls)
        if isinstance(node, ast.Name) and node.id in bindings
    }
    stack = list(seen)
    while stack:
        for nxt in bindings[stack.pop()].refs:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return tuple((name, bindings[name].source) for name in sorted(seen))


def _is_literal(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "Literal") or (
        isinstance(node, ast.Attribute) and node.attr == "Literal"
    )


def _union_leaves(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union_leaves(node.left) + _union_leaves(node.right)
    return [node]


def binding_members(source: str) -> frozenset[str] | None:
    """The members of a vocabulary-shaped binding, or ``None`` when it has no member set.

    Members are the normalized element texts of a ``Literal[...]``, a ``A | B`` union, a
    tuple/list/set display (also wrapped in ``frozenset``/``set``/``tuple``), or the
    ``key: value`` items of a dict display. Anything else (a function, a computed value)
    has no member set; the impact check then treats any change to it as breaking.
    """
    statement = ast.parse(source).body[0]
    if not isinstance(statement, ast.Expr):
        return None
    node = statement.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple"}
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if isinstance(node, ast.Subscript) and _is_literal(node.value):
        elements = (
            node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        return frozenset(ast.unparse(element) for element in elements)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return frozenset(ast.unparse(leaf) for leaf in _union_leaves(node))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return frozenset(ast.unparse(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            return None
        return frozenset(
            f"{ast.unparse(key)}: {ast.unparse(value)}"
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        )
    return None


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
    bindings = collect_module_bindings(sources, names)
    specs = {
        name: dataclasses.replace(
            contract_spec(classdefs[name]),
            module_bindings=class_module_bindings(classdefs[name], bindings),
        )
        for name in names
    }
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


def loader_schema_deps_from_source(source: str, surface: SchemaSurface) -> set[str]:
    """Surface classes a loader's SOURCE TEXT imports from ``torchcell.datamodels``.

    The text form lets a loader be analyzed at a git ref (``git show <ref>:<path>``)
    without checking it out, the same way :func:`load_surface_from_sources` does for
    the schema surface.
    """
    tree = ast.parse(source)
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


def loader_closure_from_source(source: str, surface: SchemaSurface) -> set[str]:
    """Transitive closure of a loader's schema imports, from its source text."""
    return forward_closure(
        loader_schema_deps_from_source(source, surface), surface.ref_graph
    )


def symbol_dependents(
    loader_sources: dict[str, str], surface: SchemaSurface
) -> dict[str, set[str]]:
    """Reverse index: schema symbol -> the loaders (by label) whose closure contains it.

    Answers "if this symbol's contract changes, which datasets are disturbed?" for the
    whole fleet at once, which is the question an admission check asks about every
    symbol a NEW dataset shares with the datasets already served.
    """
    dependents: dict[str, set[str]] = {}
    for label, source in loader_sources.items():
        for symbol in loader_closure_from_source(source, surface):
            dependents.setdefault(symbol, set()).add(label)
    return dependents


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
