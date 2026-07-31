"""Introspect the torchcell pydantic schema into a typed, renderable graph.

The schema in :mod:`torchcell.datamodels.schema` is the ontology: an experiment is
a typed ``genotype x environment -> phenotype`` record. This module reads the live
pydantic models (never a hand-maintained copy) and produces an
:class:`OntologyGraph` describing every class, its own fields, its inheritance
parent, and its composition references.

Reading the models at runtime rather than parsing source is what keeps the diagram
honest: a class added to the schema shows up in the figure on the next regeneration,
and a class that is renamed cannot leave a stale box behind.
"""

from __future__ import annotations

import enum
import inspect
import types
import typing
from typing import Any, cast

from pydantic import BaseModel, Field

from torchcell.datamodels import compound_identity as compound_identity_module
from torchcell.datamodels import media as media_module
from torchcell.datamodels import pydant as pydant_module
from torchcell.datamodels import schema as schema_module

# Mixins that carry configuration rather than domain content. They are real bases,
# but drawing them would add a box every class points at and say nothing about the
# biology, so they are recorded as a note in the legend instead.
CONFIG_MIXINS = frozenset({"ModelStrict", "ModelStrictArbitrary"})

# Lane assignment. Each entry is (lane key, roots) where a root's entire inheritance
# subtree joins that lane. Order here is the left-to-right / top-to-bottom reading
# order of the finished figure.
LANE_ROOTS: dict[str, tuple[str, ...]] = {
    "genotype": (
        "Genotype",
        "GenePerturbation",
        "CrisprConstruct",
        "ExpressionRangeMultiplier",
    ),
    "environment": (
        "Environment",
        "EnvironmentPerturbation",
        "Media",
        "MediaComponent",
        "Compound",
        "CompoundIdentityRecord",
        "CompoundIdentityResolution",
        "Concentration",
        "Solvent",
        "Temperature",
    ),
    "phenotype": ("Phenotype",),
    "experiment": ("Experiment", "ExperimentReference"),
    "provenance": ("Publication", "ProvenanceGapMixin", "ReferenceGenome", "SOTerm"),
}

LANE_TITLES: dict[str, str] = {
    "genotype": "Genotype — what was changed in the cell",
    "environment": "Environment — what the cell was grown in",
    "phenotype": "Phenotype — what was measured",
    "experiment": "Experiment — the typed record and its reference",
    "provenance": "Provenance — where the record came from",
    "enum": "Controlled vocabularies",
}

# Block heading + one-line gloss for the printed schematic panel. These two are the
# only editorial strings on that figure; everything inside a block is derived from the
# graph, so a class cannot be named on the panel unless it exists in the schema.
LANE_HEADINGS: dict[str, str] = {
    "genotype": "GENOTYPE",
    "environment": "ENVIRONMENT",
    "phenotype": "PHENOTYPE",
    "experiment": "EXPERIMENT",
    "provenance": "PROVENANCE",
    "enum": "CONTROLLED VOCABULARIES",
}

LANE_SUBTITLES: dict[str, str] = {
    "genotype": "1..n perturbations off a reference genome",
    "environment": "medium × temperature × perturbation",
    "phenotype": "what was measured, with its uncertainty",
    "experiment": "the typed record, paired with its control",
    "provenance": "where the record came from",
    "enum": "closed sets, no free text",
}

# Domain -> palette slot. Six domains map onto the six primary palette colors, so the
# four warm primaries are spent before blue and gray (see CLAUDE.md figure standards).
LANE_PALETTE_INDEX: dict[str, int] = {
    "genotype": 0,  # amber
    "environment": 1,  # brick
    "phenotype": 2,  # lilac
    "experiment": 3,  # wheat
    "provenance": 4,  # steel blue
    "enum": 5,  # gray
}

LANE_ORDER: tuple[str, ...] = (
    "genotype",
    "environment",
    "experiment",
    "phenotype",
    "provenance",
    "enum",
)


class OntologyField(BaseModel):
    """One pydantic field on a class, as it should appear in a diagram row."""

    name: str
    type_label: str = Field(description="rendered type, unions collapsed for width")
    required: bool
    inherited: bool = Field(description="declared on an ancestor, not on this class")
    description: str = ""
    references: list[str] = Field(
        default_factory=list,
        description="schema class/enum names appearing anywhere in the annotation",
    )


class OntologyClass(BaseModel):
    """A pydantic model or enum in the schema, plus its place in the ontology."""

    name: str
    kind: str = Field(description="'model' or 'enum'")
    lane: str
    module: str
    doc: str = ""
    parent: str | None = Field(default=None, description="single schema base class")
    own_fields: list[OntologyField] = Field(default_factory=list)
    inherited_field_count: int = 0
    enum_members: list[str] = Field(default_factory=list)
    is_abstract: bool = False

    @property
    def row_count(self) -> int:
        """Rows the card body needs (fields for a model, members for an enum)."""
        return len(self.enum_members) if self.kind == "enum" else len(self.own_fields)


class CompositionEdge(BaseModel):
    """A ``has-a`` edge: ``source.field`` holds one or more ``target``."""

    source: str
    target: str
    field_name: str
    is_collection: bool = False


class OntologyGraph(BaseModel):
    """The whole schema as classes plus inheritance and composition edges."""

    classes: dict[str, OntologyClass] = Field(default_factory=dict)
    composition_edges: list[CompositionEdge] = Field(default_factory=list)

    def children_of(self, name: str) -> list[str]:
        """Direct inheritance children, in stable alphabetical order."""
        return sorted(c.name for c in self.classes.values() if c.parent == name)

    def lane_members(self, lane: str) -> list[str]:
        """Class names in a lane, in stable alphabetical order."""
        return sorted(c.name for c in self.classes.values() if c.lane == lane)

    def descendants_of(self, name: str) -> list[str]:
        """Every class below ``name`` in the inheritance tree, excluding itself."""
        out: list[str] = []
        frontier = self.children_of(name)
        while frontier:
            child = frontier.pop()
            out.append(child)
            frontier.extend(self.children_of(child))
        return sorted(out)

    def lane_roots(self, lane: str) -> list[str]:
        """Classes in ``lane`` whose inheritance parent sits outside the lane."""
        roots: list[str] = []
        for name in self.lane_members(lane):
            parent = self.classes[name].parent
            if parent is None or self.classes[parent].lane != lane:
                roots.append(name)
        return roots


def _collapse_type(annotation: Any) -> str:
    """Render an annotation compactly enough to fit a card row.

    Wide discriminated unions (``Genotype.perturbations`` is a union of 19
    perturbation classes) would blow the card width out by an order of magnitude,
    so a union of more than two schema classes collapses to a count. The members
    are not lost -- they are still drawn as composition edges.
    """
    text = str(annotation)
    for prefix in (
        "torchcell.datamodels.schema.",
        "torchcell.datamodels.pydant.",
        "torchcell.datamodels.media.",
        "torchcell.datamodels.compound_identity.",
        "torchcell.verification.sourced.",
        "typing.",
    ):
        text = text.replace(prefix, "")
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[8:-2]
    text = text.replace("<enum '", "").replace("'>", "")

    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(args) != len(typing.get_args(annotation))
        if len(args) > 2:
            return f"one of {len(args)}" + (" | None" if optional else "")
    if origin in (list, set, tuple):
        (inner,) = typing.get_args(annotation)[:1] or (None,)
        inner_origin = typing.get_origin(inner)
        if inner_origin in (typing.Union, types.UnionType):
            members = [a for a in typing.get_args(inner) if a is not type(None)]
            if len(members) > 2:
                return f"list[one of {len(members)}]"
    return text if len(text) <= 46 else text[:44] + "…"


def _referenced_names(annotation: Any, known: set[str]) -> list[str]:
    """Every schema class/enum name appearing anywhere inside an annotation."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        name = getattr(node, "__name__", None)
        if name in known:
            found.add(name)
        for arg in typing.get_args(node):
            walk(arg)

    walk(annotation)
    return sorted(found)


def _is_collection(annotation: Any) -> bool:
    if typing.get_origin(annotation) in (list, set, tuple):
        return True
    return any(
        typing.get_origin(a) in (list, set, tuple) for a in typing.get_args(annotation)
    )


def _lane_for(name: str, parent_chain: list[str]) -> str:
    """Lane of a class: its own root assignment, else its nearest assigned ancestor.

    Raises rather than defaulting. A silent catch-all would file every newly added
    top-level class under whichever lane the default named, and the figure would look
    correct while quietly misplacing it -- the exact drift this module exists to stop.
    """
    for lane, roots in LANE_ROOTS.items():
        if name in roots:
            return lane
    for ancestor in parent_chain:
        for lane, roots in LANE_ROOTS.items():
            if ancestor in roots:
                return lane
    raise KeyError(
        f"{name!r} has no lane: it is not in LANE_ROOTS and none of its ancestors "
        f"{parent_chain or '(no schema base)'} are either. Add it to LANE_ROOTS in "
        f"torchcell/paper/ontology_graph.py."
    )


def build_ontology_graph() -> OntologyGraph:
    """Introspect the live schema modules into an :class:`OntologyGraph`."""
    modules = (schema_module, media_module, pydant_module, compound_identity_module)
    module_names = {m.__name__ for m in modules}

    registry: dict[str, tuple[str, type]] = {}
    for module in modules:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or name in registry:
                continue
            if obj.__module__ not in module_names or name in CONFIG_MIXINS:
                continue
            if issubclass(obj, BaseModel):
                registry[name] = ("model", obj)
            elif issubclass(obj, enum.Enum):
                registry[name] = ("enum", obj)

    known = set(registry)
    graph = OntologyGraph()

    for name, (kind, obj) in registry.items():
        parents = [
            b.__name__
            for b in obj.__bases__
            if b.__name__ in registry and registry[b.__name__][0] == "model"
        ]
        parent = parents[0] if parents else None

        chain: list[str] = []
        cursor = parent
        while cursor is not None:
            chain.append(cursor)
            cursor_parents = [
                b.__name__
                for b in registry[cursor][1].__bases__
                if b.__name__ in registry
            ]
            cursor = cursor_parents[0] if cursor_parents else None

        entry = OntologyClass(
            name=name,
            kind=kind,
            lane="enum" if kind == "enum" else _lane_for(name, chain),
            module=obj.__module__,
            doc=(inspect.getdoc(obj) or "").strip().split("\n")[0],
            parent=parent,
        )

        if kind == "enum":
            enum_cls = cast(type[enum.Enum], obj)
            entry.enum_members = [
                m.value if isinstance(m.value, str) else m.name for m in enum_cls
            ]
            graph.classes[name] = entry
            continue

        model_cls = cast(type[BaseModel], obj)
        declared = set(getattr(obj, "__annotations__", {}) or {})
        inherited = 0
        for field_name, info in model_cls.model_fields.items():
            if field_name not in declared:
                inherited += 1
                continue
            refs = _referenced_names(info.annotation, known)
            entry.own_fields.append(
                OntologyField(
                    name=field_name,
                    type_label=_collapse_type(info.annotation),
                    required=info.is_required(),
                    inherited=False,
                    description=info.description or "",
                    references=refs,
                )
            )
            for ref in refs:
                graph.composition_edges.append(
                    CompositionEdge(
                        source=name,
                        target=ref,
                        field_name=field_name,
                        is_collection=_is_collection(info.annotation),
                    )
                )
        entry.inherited_field_count = inherited
        # A base class nobody instantiates directly: it has subclasses and its name
        # is the bare concept ("Phenotype", "GenePerturbation").
        graph.classes[name] = entry

    for name, entry in graph.classes.items():
        if entry.kind == "model":
            entry.is_abstract = any(c.parent == name for c in graph.classes.values())

    return graph
