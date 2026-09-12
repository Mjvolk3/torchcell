# torchcell/datamodels/ontology_checks.py
# [[torchcell.datamodels.ontology-checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/ontology_checks
# Test file: tests/torchcell/datamodels/test_ontology_coherence.py
"""Programmatic coherence checks over the torchcell ontology.

``schema.py`` is the ontology, but it is only half of the contract. A record that
validates in pydantic still fails to reach the knowledge graph unless the BioCypher
graph schema declares its node class, the adapter emits exactly the declared
properties, and the identity keys a cross-dataset join needs (a resolvable
``Compound``, a shared ``Media`` base, a ``Temperature``) are actually carried.

The existing test modules cover ONE artifact each: ``test_ontology_invariants`` and
``test_ontology_all_trees`` the pydantic trees, ``test_adapter_schema_consistency``
the adapter conf against the graph schema. This module covers the SEAMS between
them, and the identity keys the joins stand on. Every function here is pure and
offline: it reads the live pydantic models, the committed
``torchcell_schema_config.yaml``, and the ``cell_adapter.py`` source as an AST.
Nothing opens an LMDB, a database, or the network.

The four properties the checks defend (the review brief's framing):

1. **Faithful representation.** A persistent entity (genotype, medium, compound,
   temperature) and a contingent observation (the phenotype) stay separable, and
   neither can quietly compose the other.
2. **Additive expansion.** A new phenotype family is a new leaf plus a new node
   class plus a new adapter method; the bijection checks fail the moment one of the
   three is missing, which is exactly the failure BioCypher otherwise swallows.
3. **Cross-dataset joins.** Identity keys are present and resolvable: a
   ``Compound`` carries a structure identifier or a typed gap, a derived medium
   names a base that exists, a base's components survive into its derivatives.
4. **Room for a per-experiment protocol.** Nothing here forbids new lanes or new
   leaves; the lane checks constrain DIRECTION (no back-edge into the experiment
   lane), not membership.
"""

from __future__ import annotations

import ast
import enum
import types
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import torchcell
from torchcell.datamodels import schema as s
from torchcell.datamodels.compound_identity import normalize_compound_name
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    GraphSchemaEntry,
    graph_schema_from_yaml,
)
from torchcell.verification.sourced import SourcedValue

__all__ = [
    "REPO_ROOT",
    "IDENTITY_FIELDS",
    "PHENOTYPE_ENVELOPE_PROPERTIES",
    "AdapterNodeSite",
    "CompoundIdentityIssue",
    "EnvironmentCompoundCollision",
    "JoinKeyCensus",
    "MediaBaseIssue",
    "MediaDerivationIssue",
    "OptionalTraversal",
    "PhenotypeLabelMap",
    "PropertyMismatch",
    "adapter_node_sites",
    "adapter_optional_traversals",
    "adapter_property_mismatches",
    "cell_adapter_source",
    "compound_has_identity",
    "compound_has_identity_gap",
    "composition_targets",
    "enum_value_collisions",
    "environment_compound_collisions",
    "experiment_closure",
    "graph_schema",
    "isolated_graph_node_classes",
    "join_key_audit",
    "lane_back_edges",
    "lane_membership",
    "media_base_issues",
    "media_derivation_issues",
    "media_library_compound_issues",
    "orphan_models",
    "phenotype_label_map",
    "schema_enums",
    "schema_models",
    "sourced_value_fields",
]

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent

# A structure identifier is anything that resolves the molecule to a structure. The
# ordering mirrors Compound's own docstring: inchikey is the primary key, the rest
# are cross-references or auxiliary structure strings.
IDENTITY_FIELDS: tuple[str, ...] = (
    "inchikey",
    "chebi_id",
    "pubchem_cid",
    "smiles",
    "inchi",
)

# Properties every phenotype node class carries regardless of family, so they carry
# no information about WHICH phenotype the class models.
PHENOTYPE_ENVELOPE_PROPERTIES: frozenset[str] = frozenset(
    {"graph_level", "label_name", "label_statistic_name", "serialized_data"}
)

# Attribute chains the adapter walks, and the model each is rooted in. These are the
# two entrypoints every node method starts from: the chunked methods take a record
# dict, the reference methods take an experiment-reference-index entry.
ADAPTER_ROOTS: dict[str, type[BaseModel]] = {
    "data['experiment']": s.Experiment,
    "data.reference": s.ExperimentReference,
}


# --------------------------------------------------------------------------- #
# Report models. Pydantic-first: every check returns typed records, never tuples,
# so a caller (a test, a verifier, a future admission gate) reads fields by name.
# --------------------------------------------------------------------------- #
class AdapterNodeSite(BaseModel):
    """One ``BioCypherNode(...)`` construction in ``cell_adapter.py``."""

    model_config = ConfigDict(extra="forbid")

    function: str = Field(description="enclosing CellAdapter method name")
    lineno: int
    node_label: str | None = Field(
        default=None, description="the emitted graph class; None if not a literal"
    )
    property_keys: list[str] | None = Field(
        default=None,
        description="emitted property names; None when the properties argument is "
        "not statically resolvable (a form this reader does not understand)",
    )


class PropertyMismatch(BaseModel):
    """A node class whose emitted properties differ from its declared properties."""

    model_config = ConfigDict(extra="forbid")

    node_label: str
    function: str
    lineno: int
    emitted_not_declared: list[str] = Field(default_factory=list)
    declared_not_emitted: list[str] = Field(default_factory=list)


class OptionalTraversal(BaseModel):
    """An adapter attribute chain that walks THROUGH a field the schema may null."""

    model_config = ConfigDict(extra="forbid")

    lineno: int
    chain: str = Field(description="the full attribute expression, as written")
    optional_hop: str = Field(description="Model.field whose annotation admits None")


class PhenotypeLabelMap(BaseModel):
    """Bijection between concrete phenotype classes and phenotype node classes.

    The mapping is derived from DATA, not from a naming convention: a node class's
    non-envelope properties must be field names of exactly one concrete phenotype.
    That is why ``RNASeqExpressionPhenotype`` <-> ``rnaseq expression phenotype``
    and ``CalMorphPhenotype`` <-> ``calmorph phenotype`` resolve without a hand
    alias table, and why a property drifting off its class is caught.
    """

    model_config = ConfigDict(extra="forbid")

    matched: dict[str, str] = Field(
        default_factory=dict, description="graph node class -> phenotype class name"
    )
    ambiguous: dict[str, list[str]] = Field(
        default_factory=dict, description="node class -> several candidate classes"
    )
    unmatched_labels: list[str] = Field(
        default_factory=list, description="node classes matching no phenotype class"
    )
    unmapped_classes: list[str] = Field(
        default_factory=list, description="phenotype classes with no node class"
    )

    @property
    def is_bijective(self) -> bool:
        """True when every class has exactly one node class and vice versa."""
        return not (self.ambiguous or self.unmatched_labels or self.unmapped_classes)


class CompoundIdentityIssue(BaseModel):
    """A ``Compound`` that is neither identified nor honestly gapped."""

    model_config = ConfigDict(extra="forbid")

    context: str = Field(
        description="where the compound was found, e.g. 'SC.component'"
    )
    compound_name: str
    definition: str = Field(description="ComponentDefinition value, or 'dropout'")


class MediaBaseIssue(BaseModel):
    """A ``Media.base_medium`` that names nothing in ``MEDIA_LIBRARY``."""

    model_config = ConfigDict(extra="forbid")

    media_key: str
    base_medium: str


class MediaDerivationIssue(BaseModel):
    """A derived medium that dropped a base component without declaring a dropout."""

    model_config = ConfigDict(extra="forbid")

    media_key: str
    base_medium: str
    missing_components: list[str]


class EnvironmentCompoundCollision(BaseModel):
    """One compound encoded twice in an environment: as a component AND as an edit."""

    model_config = ConfigDict(extra="forbid")

    identity: str = Field(description="the join key that collided (InChIKey or name)")
    component_name: str
    perturbation_name: str


class JoinKeyCensus(BaseModel):
    """How many records of a built dataset are joinable on each identity key.

    Reads the stored experiment mappings (the JSON dicts an LMDB yields), so a
    verifier can point it at any built dataset without importing its loader.
    """

    model_config = ConfigDict(extra="forbid")

    n_records: int = 0
    n_with_genome: int = 0
    n_with_systematic_gene_names: int = 0
    n_with_media_name: int = 0
    n_with_media_base: int = 0
    n_with_temperature: int = 0
    n_with_temperature_gap: int = 0
    n_with_every_compound_identified: int = 0
    distinct_genomes: list[str] = Field(default_factory=list)
    distinct_media_bases: list[str] = Field(default_factory=list)
    distinct_compound_identities: list[str] = Field(default_factory=list)
    unidentified_compound_names: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Schema introspection.
# --------------------------------------------------------------------------- #
def schema_models() -> frozenset[type[BaseModel]]:
    """Every pydantic model DEFINED in ``schema.py`` (not merely imported)."""
    return frozenset(
        obj
        for obj in vars(s).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == s.__name__
    )


def schema_enums() -> frozenset[type[enum.Enum]]:
    """Every enum DEFINED in ``schema.py``."""
    return frozenset(
        obj
        for obj in vars(s).values()
        if isinstance(obj, type)
        and issubclass(obj, enum.Enum)
        and obj.__module__ == s.__name__
    )


def _models_in(annotation: object) -> set[type[BaseModel]]:
    """Every pydantic model named anywhere inside a (nested) type annotation."""
    found: set[type[BaseModel]] = set()

    def walk(node: object) -> None:
        if isinstance(node, type) and issubclass(node, BaseModel):
            found.add(node)
            return
        for arg in typing.get_args(node):
            walk(arg)

    walk(annotation)
    return found


def composition_targets(model: type[BaseModel]) -> frozenset[type[BaseModel]]:
    """Schema models this model COMPOSES (holds in one of its fields)."""
    known = schema_models()
    out: set[type[BaseModel]] = set()
    for field in model.model_fields.values():
        out |= {
            m for m in _models_in(field.annotation) if m in known and m is not model
        }
    return frozenset(out)


def experiment_closure() -> frozenset[type[BaseModel]]:
    """Models reachable from ``Experiment`` / ``ExperimentReference``.

    Reachability is composition PLUS inheritance in both directions: a union slot
    typed on a leaf makes the leaf's abstract bases part of the ontology, and a slot
    typed on a base admits every subclass. Composition alone would call every
    abstract base an orphan, which is noise rather than a finding.
    """
    known = schema_models()
    seen: set[type[BaseModel]] = {
        m for m in known if issubclass(m, (s.Experiment, s.ExperimentReference))
    }
    stack = list(seen)
    while stack:
        node = stack.pop()
        nxt = set(composition_targets(node))
        nxt |= {b for b in node.__mro__[1:] if b in known}
        nxt |= {c for c in node.__subclasses__() if c in known}
        for item in nxt:
            if item not in seen:
                seen.add(item)
                stack.append(item)
    return frozenset(seen)


def orphan_models() -> list[str]:
    """Schema models no experiment record can ever reach, sorted by name."""
    return sorted(m.__name__ for m in schema_models() - experiment_closure())


def _inheritance_subtree(roots: Iterable[type[BaseModel]]) -> set[type[BaseModel]]:
    known = schema_models()
    out: set[type[BaseModel]] = set()
    stack = [r for r in roots]
    while stack:
        node = stack.pop()
        if node in out:
            continue
        out.add(node)
        stack.extend(c for c in node.__subclasses__() if c in known)
    return out


def lane_membership() -> dict[str, frozenset[type[BaseModel]]]:
    """The three record lanes, each closed under inheritance AND composition.

    A lane is what an experiment's three slots point at: the genotype it applied,
    the environment it applied it in, and the phenotype it observed. Keeping these
    disjoint is what makes a persistent entity separable from a contingent
    observation: a medium that could also be reached from a phenotype would mean the
    observation carries part of the thing observed.
    """
    lanes: dict[str, frozenset[type[BaseModel]]] = {}
    roots: dict[str, tuple[type[BaseModel], ...]] = {
        "genotype": (s.Genotype, s.SegregantGenotype),
        "environment": (s.Environment,),
        "phenotype": (s.Phenotype,),
    }
    for lane, lane_roots in roots.items():
        members = _inheritance_subtree(lane_roots)
        stack = list(members)
        while stack:
            node = stack.pop()
            for target in composition_targets(node):
                for related in _inheritance_subtree([target]):
                    if related not in members:
                        members.add(related)
                        stack.append(related)
        lanes[lane] = frozenset(members)
    return lanes


def lane_back_edges() -> list[str]:
    """Composition edges pointing UP from a lane into the experiment lane.

    The ontology is a DAG with a declared direction: an experiment composes a
    genotype, an environment and a phenotype, never the reverse. A phenotype that
    composed an experiment would make the record self-referential and would break
    every content-addressed node id (the phenotype's hash would depend on the
    experiment's, which depends on the phenotype's).
    """
    experiments = {
        m
        for m in schema_models()
        if issubclass(m, (s.Experiment, s.ExperimentReference))
    }
    out: list[str] = []
    for lane, members in sorted(lane_membership().items()):
        for model in sorted(members, key=lambda m: m.__name__):
            for target in sorted(composition_targets(model), key=lambda m: m.__name__):
                if target in experiments:
                    out.append(f"{lane}: {model.__name__} -> {target.__name__}")
    return out


def sourced_value_fields() -> dict[str, str]:
    """``Model.field -> rendered annotation`` for every field carrying SourcedValue."""
    out: dict[str, str] = {}
    for model in sorted(schema_models(), key=lambda m: m.__name__):
        for name, field in model.model_fields.items():
            if SourcedValue in _models_in(field.annotation):
                out[f"{model.__name__}.{name}"] = str(field.annotation)
    return out


def enum_value_collisions() -> dict[str, list[str]]:
    """Schema enums whose members share a value (a silent Python alias).

    ``a = "x"`` followed by ``b = "x"`` does not error: ``b`` becomes an ALIAS of
    ``a``, iteration yields one member, and two ontology concepts collapse into one
    with no diagnostic anywhere.
    """
    out: dict[str, list[str]] = {}
    for cls in sorted(schema_enums(), key=lambda c: c.__name__):
        by_value: dict[object, list[str]] = {}
        for name, member in cls.__members__.items():
            by_value.setdefault(member.value, []).append(name)
        collided = sorted(
            n for names in by_value.values() if len(names) > 1 for n in names
        )
        if collided:
            out[cls.__name__] = collided
    return out


# --------------------------------------------------------------------------- #
# Graph schema + adapter source.
# --------------------------------------------------------------------------- #
def graph_schema() -> dict[str, GraphSchemaEntry]:
    """The committed BioCypher graph schema, parsed into per-class entries."""
    return graph_schema_from_yaml(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )


def cell_adapter_source() -> str:
    """The ``CellAdapter`` module source (read as text; never imported for this)."""
    return (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")


def isolated_graph_node_classes(schema: dict[str, GraphSchemaEntry]) -> list[str]:
    """Node classes no edge class connects (invisible to any traversal query)."""
    connected: set[str] = set()
    for entry in schema.values():
        if entry.kind == "edge":
            connected |= set(entry.source) | set(entry.target)
    return sorted(
        name
        for name, entry in schema.items()
        if entry.kind == "node" and name not in connected
    )


def phenotype_label_map(schema: dict[str, GraphSchemaEntry]) -> PhenotypeLabelMap:
    """Match each phenotype node class to its concrete pydantic phenotype class."""
    concrete = [m for m in typing.get_args(s.PhenotypeType) if m is not s.Phenotype]
    labels = sorted(
        name
        for name, entry in schema.items()
        if entry.kind == "node" and name.endswith("phenotype")
    )
    result = PhenotypeLabelMap()
    claimed: set[str] = set()
    for label in labels:
        discriminating = set(schema[label].properties) - PHENOTYPE_ENVELOPE_PROPERTIES
        candidates = sorted(
            cls.__name__
            for cls in concrete
            if discriminating and discriminating <= set(cls.model_fields)
        )
        if len(candidates) == 1:
            result.matched[label] = candidates[0]
            claimed.add(candidates[0])
        elif candidates:
            result.ambiguous[label] = candidates
        else:
            result.unmatched_labels.append(label)
    result.unmapped_classes = sorted(
        cls.__name__ for cls in concrete if cls.__name__ not in claimed
    )
    return result


def _dict_keys(node: ast.Dict) -> list[str]:
    """String keys of a dict literal; a non-literal key makes the site unreadable."""
    keys: list[str] = []
    for key in node.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return []
        keys.append(key.value)
    return sorted(keys)


def _local_dict(
    function: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> ast.Dict | None:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    return None


def _returned_dicts(tree: ast.Module) -> dict[str, ast.Dict]:
    """Helper functions whose body returns a dict literal (the shared property builders)."""
    out: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in node.body:
                if isinstance(statement, ast.Return) and isinstance(
                    statement.value, ast.Dict
                ):
                    out[node.name] = statement.value
    return out


def adapter_node_sites(source: str) -> list[AdapterNodeSite]:
    """Every ``BioCypherNode(...)`` the adapter constructs, with what it emits.

    Read from the AST rather than by running the adapter: running one needs a built
    LMDB, and the property set is a static contract the graph schema must match
    whether or not any dataset is currently on disk.
    """
    tree = ast.parse(source)
    helpers = _returned_dicts(tree)
    sites: list[AdapterNodeSite] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "BioCypherNode"
            ):
                continue
            keywords = {kw.arg: kw.value for kw in call.keywords}
            label_node = keywords.get("node_label")
            label = (
                label_node.value
                if isinstance(label_node, ast.Constant)
                and isinstance(label_node.value, str)
                else None
            )
            properties = keywords.get("properties")
            keys: list[str] | None
            if properties is None:
                keys = []
            elif isinstance(properties, ast.Dict):
                keys = _dict_keys(properties)
            elif isinstance(properties, ast.Name):
                local = _local_dict(function, properties.id)
                keys = _dict_keys(local) if local is not None else None
            elif isinstance(properties, ast.Call) and isinstance(
                properties.func, ast.Attribute
            ):
                helper = helpers.get(properties.func.attr)
                keys = _dict_keys(helper) if helper is not None else None
            else:
                keys = None
            sites.append(
                AdapterNodeSite(
                    function=function.name,
                    lineno=call.lineno,
                    node_label=label,
                    property_keys=keys,
                )
            )
    return sorted(sites, key=lambda site: site.lineno)


def adapter_property_mismatches(
    sites: Iterable[AdapterNodeSite], schema: Mapping[str, GraphSchemaEntry]
) -> list[PropertyMismatch]:
    """Node classes where emitted properties != declared properties, either way.

    BioCypher drops a node class whose label the schema does not declare, and drops
    a property the schema does not list, both without an error. A declared property
    nothing emits is the mirror failure: the column exists in the store and is
    always null, so a query written against it silently returns nothing.
    """
    out: list[PropertyMismatch] = []
    for site in sites:
        if site.node_label is None or site.property_keys is None:
            continue
        entry = schema.get(site.node_label)
        declared = set(entry.properties) if entry is not None else set()
        emitted = set(site.property_keys)
        if entry is None or declared != emitted:
            out.append(
                PropertyMismatch(
                    node_label=site.node_label,
                    function=site.function,
                    lineno=site.lineno,
                    emitted_not_declared=sorted(emitted - declared),
                    declared_not_emitted=sorted(declared - emitted),
                )
            )
    return out


def _admits_none(annotation: object) -> bool:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return any(arg is type(None) for arg in typing.get_args(annotation))
    return False


def _model_of(annotation: object) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _attribute_chain(node: ast.Attribute) -> tuple[str, list[str]]:
    """``(root expression text, attribute names)`` for a dotted chain."""
    attributes: list[str] = []
    cursor: ast.expr = node
    while isinstance(cursor, ast.Attribute):
        attributes.append(cursor.attr)
        cursor = cursor.value
    return ast.unparse(cursor), list(reversed(attributes))


def adapter_optional_traversals(source: str) -> list[OptionalTraversal]:
    """Adapter chains that dereference THROUGH a schema field that may be None.

    ``Environment.temperature`` is optional because a secondary curation layer may
    not carry it (a typed gap, not a guess). Any adapter chain that walks past such
    a field unguarded turns a legal record into an AttributeError at KG-build time,
    which is the schema and the adapter disagreeing about what a record may be.
    """
    tree = ast.parse(source)
    found: dict[tuple[str, str], OptionalTraversal] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        root_text, attributes = _attribute_chain(node)
        full = ".".join([root_text, *attributes])
        model: type[BaseModel] | None = None
        remainder: list[str] = []
        for candidate, root_model in ADAPTER_ROOTS.items():
            if full == candidate or full.startswith(candidate + "."):
                model = root_model
                remainder = full[len(candidate) :].lstrip(".").split(".")
                remainder = [part for part in remainder if part]
        if model is None or not remainder:
            continue
        cursor: type[BaseModel] | None = model
        for index, attribute in enumerate(remainder):
            if cursor is None:
                break
            field = cursor.model_fields.get(attribute)
            if field is None:
                break
            if index < len(remainder) - 1 and _admits_none(field.annotation):
                key = (full, f"{cursor.__name__}.{attribute}")
                found.setdefault(
                    key,
                    OptionalTraversal(
                        lineno=node.lineno, chain=full, optional_hop=key[1]
                    ),
                )
                break
            cursor = _model_of(field.annotation)
    return sorted(found.values(), key=lambda t: (t.lineno, t.chain))


# --------------------------------------------------------------------------- #
# Identity keys: compounds, media, temperature, systematic gene names.
# --------------------------------------------------------------------------- #
def compound_has_identity(compound: s.Compound) -> bool:
    """True when the compound carries at least one structure identifier."""
    return any(getattr(compound, field) is not None for field in IDENTITY_FIELDS)


def compound_has_identity_gap(compound: s.Compound) -> bool:
    """True when an identifier's absence is declared as a typed ``ProvenanceGap``."""
    return any(gap.field in IDENTITY_FIELDS for gap in compound.provenance_gaps)


def _compound_key(compound: s.Compound) -> str:
    """The join key for a compound: its InChIKey, else its normalized name."""
    if compound.inchikey is not None:
        return compound.inchikey
    return normalize_compound_name(compound.name)


def media_library_compound_issues() -> list[CompoundIdentityIssue]:
    """Shared-library compounds that are neither identified nor honestly gapped.

    Scoped to compounds that name ONE substance: a ``defined`` component and every
    dropout. ``composition_deferred`` (commercial YNB) and
    ``intrinsically_undefined`` (peptone, yeast extract) are mixtures, so no
    structure identifier exists to demand.
    """
    issues: list[CompoundIdentityIssue] = []
    for key, media in sorted(MEDIA_LIBRARY.items()):
        for component in media.components:
            if component.definition is not s.ComponentDefinition.defined:
                continue
            compound = component.compound
            if compound_has_identity(compound) or compound_has_identity_gap(compound):
                continue
            issues.append(
                CompoundIdentityIssue(
                    context=f"{key}.components",
                    compound_name=compound.name,
                    definition=component.definition.value,
                )
            )
        for dropout in media.dropouts:
            if compound_has_identity(dropout) or compound_has_identity_gap(dropout):
                continue
            issues.append(
                CompoundIdentityIssue(
                    context=f"{key}.dropouts",
                    compound_name=dropout.name,
                    definition="dropout",
                )
            )
    return issues


def media_base_issues() -> list[MediaBaseIssue]:
    """Library media whose ``base_medium`` names no member of ``MEDIA_LIBRARY``.

    A base label that resolves to nothing joins nothing: aggregating "every record
    on an SD base" needs the base to BE an object, with components and provenance,
    not a string two datasets happen to spell the same way.
    """
    return [
        MediaBaseIssue(media_key=key, base_medium=media.base_medium)
        for key, media in sorted(MEDIA_LIBRARY.items())
        if media.base_medium is not None and media.base_medium not in MEDIA_LIBRARY
    ]


def media_derivation_issues() -> list[MediaDerivationIssue]:
    """Derived media that silently lost a base component.

    For a medium naming a base that exists, every (compound name, role) of the base
    must reappear in the derivative or be declared in ``dropouts``. That is what
    makes "SC-Ura" a typed edit of SC rather than a separate recipe that happens to
    look similar.
    """
    issues: list[MediaDerivationIssue] = []
    for key, media in sorted(MEDIA_LIBRARY.items()):
        base_key = media.base_medium
        if base_key is None or base_key == key or base_key not in MEDIA_LIBRARY:
            continue
        base = MEDIA_LIBRARY[base_key]
        own = {(c.compound.name, c.role.value) for c in media.components}
        dropped = {d.name for d in media.dropouts}
        missing = sorted(
            name
            for name, role in {(c.compound.name, c.role.value) for c in base.components}
            - own
            if name not in dropped
        )
        if missing:
            issues.append(
                MediaDerivationIssue(
                    media_key=key, base_medium=base_key, missing_components=missing
                )
            )
    return issues


def environment_compound_collisions(
    environment: s.Environment,
) -> list[EnvironmentCompoundCollision]:
    """Compounds encoded twice in one environment: as a component AND as an edit.

    A chemical stress is an ``EnvironmentPerturbation``; a constant ingredient is a
    ``MediaComponent``. The same species in both roles is double-counted, and any
    aggregate over "records dosed with X" then disagrees with the medium's own
    composition. Identity is the InChIKey when both sides carry one, else the
    normalized name.
    """
    components = {
        _compound_key(component.compound): component.compound.name
        for component in environment.media.components
        if component.definition is s.ComponentDefinition.defined
    }
    out: list[EnvironmentCompoundCollision] = []
    for perturbation in environment.perturbations:
        compound = getattr(perturbation, "compound", None)
        if compound is None:
            compound = getattr(perturbation, "agent", None)
        if not isinstance(compound, s.Compound):
            continue
        key = _compound_key(compound)
        if key in components:
            out.append(
                EnvironmentCompoundCollision(
                    identity=key,
                    component_name=components[key],
                    perturbation_name=compound.name,
                )
            )
    return out


def _compound_identity_from_mapping(compound: Mapping[str, Any]) -> str | None:
    for field in IDENTITY_FIELDS:
        value = compound.get(field)
        if value is not None:
            return f"{field}:{value}"
    return None


def join_key_audit(
    records: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> JoinKeyCensus:
    """Census of the identity keys a built dataset actually carries.

    Takes ``(experiment, reference)`` mapping pairs -- exactly what an LMDB yields
    -- so a verifier can run it over any dataset without importing its loader. The
    counts answer the question a multimodal training set turns on: at which level
    can these records be joined to another dataset's records at all.
    """
    census = JoinKeyCensus()
    genomes: set[str] = set()
    bases: set[str] = set()
    identities: set[str] = set()
    unidentified: set[str] = set()
    for experiment, reference in records:
        census.n_records += 1
        genome = reference.get("genome_reference") or {}
        if genome.get("species") and genome.get("strain"):
            census.n_with_genome += 1
            genomes.add(
                f"{genome['species']}|{genome['strain']}|{genome.get('ploidy')}"
            )
        genotype = experiment.get("genotype") or {}
        perturbations = genotype.get("perturbations") or []
        names = [
            p.get("systematic_gene_name")
            for p in perturbations
            if p.get("systematic_gene_name")
        ]
        if names:
            census.n_with_systematic_gene_names += 1
        environment = experiment.get("environment") or {}
        media = environment.get("media") or {}
        if media.get("name"):
            census.n_with_media_name += 1
        if media.get("base_medium"):
            census.n_with_media_base += 1
            bases.add(str(media["base_medium"]))
        if environment.get("temperature") is not None:
            census.n_with_temperature += 1
        elif any(
            gap.get("field") == "temperature"
            for gap in (environment.get("provenance_gaps") or [])
        ):
            census.n_with_temperature_gap += 1
        compounds: list[Mapping[str, Any]] = []
        for component in media.get("components") or []:
            if component.get("definition") == s.ComponentDefinition.defined.value:
                compounds.append(component.get("compound") or {})
        for perturbation in environment.get("perturbations") or []:
            compound = perturbation.get("compound") or perturbation.get("agent")
            if compound:
                compounds.append(compound)
        resolved = True
        for compound in compounds:
            identity = _compound_identity_from_mapping(compound)
            if identity is None:
                resolved = False
                unidentified.add(str(compound.get("name")))
            else:
                identities.add(identity)
        if resolved:
            census.n_with_every_compound_identified += 1
    census.distinct_genomes = sorted(genomes)
    census.distinct_media_bases = sorted(bases)
    census.distinct_compound_identities = sorted(identities)
    census.unidentified_compound_names = sorted(unidentified)
    return census
