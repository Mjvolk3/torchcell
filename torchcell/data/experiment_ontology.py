# torchcell/data/experiment_ontology.py
# [[torchcell.data.experiment_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/experiment_ontology
"""The experiment's annotation to the schema DAG, and a DCell-style encoder over it.

An experiment record is an instance of the schema: an ``Experiment`` leaf class holding a
``Phenotype`` leaf, an ``Environment`` (``Media`` with components and dropouts, a
``Temperature``), beside an ``ExperimentReference`` (the control the measurement is
relative to). That record IS the experiment's annotation to the ontology; nothing is
looked up elsewhere. This module turns it into a fixed vector over a closed node
vocabulary and propagates it up the DAG to the ``Experiment`` root, so a model is
conditioned on WHAT the experiment is and never on which dataset it came from
(``dataset_name`` and every provenance field are excluded by construction).

Nodes. Class nodes are the schema classes of the experiment, phenotype and environment
lanes as :func:`torchcell.paper.ontology_graph.build_ontology_graph` reads them from
the live pydantic models (inheritance parent per class, composition edges per field).
Value nodes are the closed vocabularies the classes carry: a media component as
``MediaComponent:<compound>:<role>``, a dropout as ``Media.dropout:<compound>``, the
medium's state and base medium, the temperature unit, the environment's aerobicity.
The one numeric node is ``Temperature.value`` (celsius / 100). The reference's
phenotype leaf is annotated under a ``ref:`` namespace so the measurement's phenotype
and its reference phenotype stay distinct nodes with distinct parents. The closed value
vocabulary comes from the media library (:mod:`torchcell.datamodels.media`) plus the
records scanned at build time; a record whose value is not in the vocabulary raises,
because the table is sized to what was seen and an unlisted value is a build error,
not a case to absorb.

Edges run child -> parent: inheritance (leaf class -> its base) and composition (the
value of a field -> the class that holds it; ``Media -> Environment -> Experiment``).
``ExperimentReference`` hangs under the ``Experiment`` root because the record holds
the two side by side. Every node's ``depth`` is its longest path from a leaf; the
encoder computes one depth at a time.

Encoder. :class:`OntologyDagEncoder` keeps one small state per node:
``s_v = tanh(W_v [mean_{c in children(v)} s_c ; x_v ; present_v] + b_v)``, computed
leaves first, and returns the root state projected to the token width. A node that
is neither annotated nor above an annotated node is held at zero, so an absent
subtree contributes nothing and an unseen combination of seen leaves is placed by
the leaves it has. The DAG is three or four levels deep, so what the structure buys is
parameter sharing by subtree (one Environment state for every row of a screen), not
DCell's subsystem semantics.
"""

from __future__ import annotations

import inspect
import json
import math
import types
import typing
from collections.abc import Iterable
from enum import Enum
from typing import Any, cast

import torch
from pydantic import BaseModel, Field, PrivateAttr
from torch import nn

from torchcell.datamodels import media as media_module
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Media,
)

ROOT = "Experiment"
REF_PREFIX = "ref:"
# Sources whose records the composite-fitness converter REWROTE into FitnessExperiment at
# build time (gene_essentiality_to_fitness_conversion, synthetic_lethality_to_fitness_
# conversion). The stored record keeps no trace of the class it came from, and the
# schema has no assay field, so without this the SGD essentiality annotation and the
# curated synthetic-lethality calls annotate identically (both: fitness in bare YEPD at
# 30 C). The origin is declared here per dataset name, the name being the lookup key
# only; what reaches the model is the ORIGIN CLASS as a value node under Experiment.
# Owner 2026-10-22: "converted from good for now"; a build that writes the origin on
# the record replaces this table.
CONVERTED_FROM: dict[str, str] = {
    "GeneEssentialitySgdDataset": "GeneEssentialityExperiment",
    "SynthLethalityYeastSynthLethDbDataset": "SyntheticLethalityExperiment",
}
LANES = frozenset({"experiment", "phenotype", "environment"})
# Base classes that carry configuration, not ontology; a class whose parent is one
# of these is a lane root.
NON_ONTOLOGY_BASES = frozenset(
    {"ModelStrict", "ModelStrictArbitrary", "ProvenanceGapMixin"}
)
TEMPERATURE_SCALE = 100.0


class OntologyNode(BaseModel):
    """One node of the experiment DAG."""

    name: str
    kind: str = Field(description="'class', 'value' or 'numeric'")
    parents: list[str] = Field(default_factory=list)


class ExperimentOntologyVocabulary(BaseModel):
    """The closed node set of the experiment DAG, in a stable order, with its edges.

    A class node's FIRST parent is its inheritance parent when it has one, else the
    class that holds it (composition); value and numeric nodes have one parent, the
    class that carries the field.
    """

    nodes: list[OntologyNode]
    root: str = ROOT
    converted_from: dict[str, str] = Field(
        default_factory=dict,
        description="dataset name -> the experiment class the stored record was "
        "converted from (see CONVERTED_FROM); annotated as "
        "Experiment.converted_from:<class>",
    )
    _index: dict[str, int] = PrivateAttr(default_factory=dict)

    # ---- structure -------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """Node names in vocabulary order."""
        return [n.name for n in self.nodes]

    def __len__(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    def index(self, name: str) -> int:
        """Position of ``name``; raises on a name outside the vocabulary."""
        if not self._index:
            self._index = {n.name: i for i, n in enumerate(self.nodes)}
        if name not in self._index:
            raise ValueError(
                f"{name!r} is not in the experiment ontology vocabulary "
                f"({len(self.nodes)} nodes); rebuild it over the records that carry it"
            )
        return self._index[name]

    def children(self) -> list[list[int]]:
        """Child indices per node (child -> parent edges reversed)."""
        out: list[list[int]] = [[] for _ in self.nodes]
        for i, n in enumerate(self.nodes):
            for p in n.parents:
                out[self.index(p)].append(i)
        return out

    def depths(self) -> list[int]:
        """Longest path from a leaf, per node (leaves 0); the root is the deepest."""
        ch = self.children()
        memo: dict[int, int] = {}

        def d(i: int) -> int:
            if i in memo:
                return memo[i]
            memo[i] = 0 if not ch[i] else 1 + max(d(c) for c in ch[i])
            return memo[i]

        return [d(i) for i in range(len(self.nodes))]

    def numeric_mask(self) -> list[bool]:
        """True where a node carries a value rather than a presence bit."""
        return [n.kind == "numeric" for n in self.nodes]

    # ---- annotation ------------------------------------------------------------
    def annotate(
        self, experiment: Experiment, reference: ExperimentReference | None
    ) -> dict[str, float]:
        """``{node name: value}`` for one record: 1.0 for a present node, the scaled
        value for a numeric node. Raises on a value outside the vocabulary.
        """
        return self.apply(annotation_recipe(experiment, reference, self.converted_from))

    def apply(self, recipe: AnnotationRecipe) -> dict[str, float]:
        """``{node name: value}`` of a recipe (:func:`annotation_recipe`): every class
        it names with its first-parent chain, every value node with its owner's chain.
        Raises on a node outside the vocabulary.
        """
        out: dict[str, float] = {}
        for name in recipe.classes:
            self._chain(out, name)
        for name, value in recipe.values.items():
            out[_checked(self, name)] = value
            self._chain(out, recipe.parents[name])
        return out

    def vector(
        self, experiment: Experiment, reference: ExperimentReference | None
    ) -> torch.Tensor:
        """The annotation as a ``[n_nodes]`` float vector."""
        return self.vector_from_recipe(
            annotation_recipe(experiment, reference, self.converted_from)
        )

    def vector_from_recipe(self, recipe: AnnotationRecipe) -> torch.Tensor:
        """A recipe's annotation as a ``[n_nodes]`` float vector."""
        vec = torch.zeros(len(self.nodes), dtype=torch.float32)
        for name, value in self.apply(recipe).items():
            vec[self.index(name)] = value
        return vec

    def _chain(self, out: dict[str, float], name: str) -> None:
        """A class node and every first parent up to a root (the Experiment root
        included), so the owners of an annotated leaf are marked present.
        """
        node = self.nodes[self.index(name)]
        while True:
            out[node.name] = 1.0
            if not node.parents:
                return
            node = self.nodes[self.index(node.parents[0])]

    # ---- serialization ---------------------------------------------------------
    def to_json(self, path: str) -> None:
        """Write the vocabulary (nodes, kinds, parents) as JSON."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=1)

    @classmethod
    def from_json(cls, path: str) -> ExperimentOntologyVocabulary:
        """Read a vocabulary written by :meth:`to_json`."""
        with open(path) as f:
            return cls.model_validate(json.load(f))


def _checked(vocab: ExperimentOntologyVocabulary, name: str) -> str:
    vocab.index(name)
    return name


# --------------------------------------------------------------------------- #
# The record's annotation, independent of any vocabulary
# --------------------------------------------------------------------------- #
#: Fields that carry provenance, identity or free commentary, never ontology.
SKIP_FIELDS = frozenset(
    {"provenance", "provenance_gaps", "dataset_name", "note", "quote", "description"}
)
#: Environment fields with their own node scheme (030); the generic walk skips them.
ENVIRONMENT_HANDLED = frozenset({"media", "temperature", "perturbations", "aerobicity"})


class AnnotationRecipe(BaseModel):
    """What one record annotates, before it meets a vocabulary.

    ``classes`` are chained to the root through first parents; ``values`` are value
    and numeric nodes (1.0, or the scaled number); ``parents``/``kinds`` hold the
    parent and kind of every value node and of every class node the record introduces
    beyond the schema lanes (the genotype class, the reference genome, a nested model
    such as ``CultureFormat``). Building a vocabulary from recipes and applying a recipe
    to it are the same walk, so a prepare step can annotate records in workers and
    build the vocabulary from the union afterwards.
    """

    classes: list[str] = Field(default_factory=list)
    values: dict[str, float] = Field(default_factory=dict)
    parents: dict[str, str] = Field(default_factory=dict)
    kinds: dict[str, str] = Field(default_factory=dict)

    def add_class(self, name: str, parent: str | None) -> None:
        """Chain ``name``; when ``parent`` is given the node is new to the lanes."""
        if name not in self.classes:
            self.classes.append(name)
        if parent is not None:
            self.parents.setdefault(name, parent)
            self.kinds.setdefault(name, "class")

    def add_value(self, name: str, parent: str, kind: str, value: float) -> None:
        """A value (1.0) or numeric node under the class ``parent``."""
        self.values[name] = value
        self.parents[name] = parent
        self.kinds[name] = kind


def _scale_numeric(x: float) -> float:
    """Signed ``log1p(|x|) / 10``: durations, volumes, concentrations and counts of
    very different magnitudes on one order-one axis. A zero reads as absent to the
    encoder (``present = x != 0``), the same convention as every other numeric node.
    """
    return math.copysign(math.log1p(abs(float(x))), float(x)) / 10.0


def _is_closed_annotation(annotation: Any) -> bool:
    """True for an Enum, a Literal or a bool annotation (optionals unwrapped): a field
    whose values the schema closes, hence ontology rather than free text.
    """
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return all(
            _is_closed_annotation(a)
            for a in typing.get_args(annotation)
            if a is not type(None)
        )
    if origin is typing.Literal:
        return True
    return annotation is bool or (
        isinstance(annotation, type) and issubclass(annotation, Enum)
    )


def _walk_model(
    recipe: AnnotationRecipe,
    model: BaseModel,
    owner: str,
    numeric: bool,
    skip: frozenset[str] = frozenset(),
) -> None:
    """Value nodes of ``model``'s closed fields under the class node ``owner``.

    Per field, by the schema's own typing (no field is named here beyond
    :data:`SKIP_FIELDS`): an Enum, Literal or bool value is ``<owner>.<field>:<value>``;
    a number (when ``numeric``) is the numeric node ``<owner>.<field>``; a compound
    (a model with an ``inchikey``) is ``<owner>.<field>:<name>``, as media components
    are; a quantity (a model with ``value`` and ``unit``) is the numeric node
    ``<owner>.<field>[<unit>]``; any other nested model becomes the class node
    ``<owner>.<field>`` (named by path) under ``owner`` and is walked in turn, except
    artifact pointers (``*Ref``). Free strings
    and lists are not visited: a free string is not a closed vocabulary, and the lists
    of a record are its genes, cells or provenance.
    """
    fields = type(model).model_fields
    for fname, finfo in fields.items():
        if fname in SKIP_FIELDS or fname in skip:
            continue
        v = getattr(model, fname)
        if v is None or isinstance(v, (list, dict, tuple)):
            continue
        if isinstance(v, bool):
            recipe.add_value(f"{owner}.{fname}:{str(v).lower()}", owner, "value", 1.0)
        elif isinstance(v, Enum):
            recipe.add_value(f"{owner}.{fname}:{v.value}", owner, "value", 1.0)
        elif isinstance(v, str):
            if _is_closed_annotation(finfo.annotation):
                recipe.add_value(f"{owner}.{fname}:{v}", owner, "value", 1.0)
        elif isinstance(v, (int, float)):
            if numeric:
                recipe.add_value(
                    f"{owner}.{fname}", owner, "numeric", _scale_numeric(v)
                )
        elif isinstance(v, BaseModel):
            cls = type(v).__name__
            if cls.endswith("Ref"):
                continue
            if hasattr(v, "inchikey") and hasattr(v, "name"):
                recipe.add_value(f"{owner}.{fname}:{v.name}", owner, "value", 1.0)
            elif hasattr(v, "value") and hasattr(v, "unit"):
                if numeric and isinstance(v.value, (int, float)):
                    unit = getattr(v.unit, "value", v.unit)
                    recipe.add_value(
                        f"{owner}.{fname}[{unit}]",
                        owner,
                        "numeric",
                        _scale_numeric(v.value),
                    )
            else:
                # named by path, so a nested Media (a pre-culture's) never lands on
                # the experiment's own Media node
                path = f"{owner}.{fname}"
                recipe.add_class(path, owner)
                _walk_model(recipe, v, path, numeric)


def annotation_recipe(
    experiment: Experiment,
    reference: ExperimentReference | None,
    converted_from: dict[str, str] | None = None,
) -> AnnotationRecipe:
    """The record's annotation (module docstring), without a vocabulary.

    030's nodes (experiment, phenotype and reference classes; media by composition;
    temperature; aerobicity; environment perturbation classes; the converted origin)
    plus a generic walk (:func:`_walk_model`) of the rest of the record's design:
    the environment's remaining fields (durations, growth phase, culture format,
    pre-culture, and the compound and dose of every environment perturbation), the
    closed fields of the phenotype (measurement type, readout, sample unit), the
    genotype's class and closed fields (a segregant's cross parents, a natural
    isolate's zygosity and ploidy; never its genes), and the reference genome's closed
    fields (ploidy, assembly). ``dataset_name`` is read only as the lookup key of
    ``converted_from``.
    """
    origins = CONVERTED_FROM if converted_from is None else converted_from
    r = AnnotationRecipe()
    r.add_class(type(experiment).__name__, None)
    pheno = type(experiment.phenotype).__name__
    r.add_class(pheno, None)
    _walk_model(r, experiment.phenotype, pheno, numeric=False)
    # environment (030 scheme, then the generic walk of the remaining fields)
    env = experiment.environment
    r.add_class("Environment", None)
    for name, parent in _value_nodes_of_environment(env).items():
        r.add_value(name, parent, "value", 1.0)
    r.add_class("Media", None)
    if env.media.components:
        r.add_class("MediaComponent", None)
    if env.temperature is not None:
        r.add_class("Temperature", None)
        r.add_value(
            "Temperature.value",
            "Temperature",
            "numeric",
            float(env.temperature.value) / TEMPERATURE_SCALE,
        )
    for pert in env.perturbations:
        cls = type(pert).__name__
        r.add_class(cls, None)
        _walk_model(r, pert, cls, numeric=True)
    _walk_model(r, env, "Environment", numeric=True, skip=ENVIRONMENT_HANDLED)
    # genotype: its class and closed design fields, never its genes
    gcls = type(experiment.genotype).__name__
    r.add_class(gcls, ROOT)
    _walk_model(r, experiment.genotype, gcls, numeric=True)
    if reference is not None:
        r.add_class(REF_PREFIX + type(reference).__name__, None)
        r.add_class(REF_PREFIX + type(reference.phenotype_reference).__name__, None)
        r.add_class("ReferenceGenome", REF_PREFIX + "ExperimentReference")
        _walk_model(r, reference.genome_reference, "ReferenceGenome", numeric=False)
    origin = origins.get(experiment.dataset_name)
    if origin is not None:
        r.add_value(f"Experiment.converted_from:{origin}", ROOT, "value", 1.0)
    return r


# --------------------------------------------------------------------------- #
# Building the vocabulary
# --------------------------------------------------------------------------- #
def _media_library() -> list[Media]:
    """Every ``Media`` constant the library defines (closed, code-reviewed values)."""
    return [
        obj for _, obj in inspect.getmembers(media_module) if isinstance(obj, Media)
    ]


def _value_nodes_of_environment(env: Environment) -> dict[str, str]:
    """``{node name: parent}`` for the value nodes one environment instantiates."""
    out: dict[str, str] = {f"Environment.aerobicity:{env.aerobicity}": "Environment"}
    out.update(_value_nodes_of_media(env.media))
    if env.temperature is not None:
        out[f"Temperature.unit:{env.temperature.unit.value}"] = "Temperature"
    return out


def _value_nodes_of_media(media: Media) -> dict[str, str]:
    out: dict[str, str] = {}
    out[f"Media.state:{media.state}"] = "Media"
    out[f"Media.is_synthetic:{str(media.is_synthetic).lower()}"] = "Media"
    if media.base_medium is not None:
        out[f"Media.base_medium:{media.base_medium}"] = "Media"
    for comp in media.components:
        out[f"MediaComponent:{comp.compound.name}:{comp.role.value}"] = "MediaComponent"
    for drop in media.dropouts:
        out[f"Media.dropout:{drop.name}"] = "Media"
    return out


def build_experiment_ontology_vocabulary(
    records: Iterable[tuple[Experiment, ExperimentReference | None]] = (),
    converted_from: dict[str, str] | None = None,
    recipes: Iterable[AnnotationRecipe] = (),
) -> ExperimentOntologyVocabulary:
    """The node set from the live schema graph, the media library and ``records``.

    Deterministic for a given schema, library and record set: nodes are sorted by
    (depth order is computed at use, so the order here only has to be stable).
    """
    from torchcell.paper.ontology_graph import build_ontology_graph

    graph = build_ontology_graph()
    parents: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}

    def add(name: str, kind: str, parent: str | None) -> None:
        kinds.setdefault(name, kind)
        lst = parents.setdefault(name, [])
        if parent is not None and parent not in lst:
            lst.append(parent)

    # Class nodes of the three lanes, with their inheritance parent (first parent).
    lane_classes = {
        name: cls
        for name, cls in graph.classes.items()
        if cls.lane in LANES and cls.kind == "model"
    }
    for name, cls in lane_classes.items():
        parent = cls.parent if cls.parent in lane_classes else None
        add(name, "class", parent)

    # The reference namespace: the whole ExperimentReference and Phenotype subtrees
    # (every class whose inheritance chain reaches them, at any depth), mirrored under
    # ``ref:``; ExperimentReference hangs under the root.
    def _reaches(name: str, base: str) -> bool:
        seen: set[str] = set()
        while name in lane_classes and name not in seen:
            if name == base:
                return True
            seen.add(name)
            name = lane_classes[name].parent or ""
        return False

    for name, cls in lane_classes.items():
        for base in ("ExperimentReference", "Phenotype"):
            if _reaches(name, base):
                parent = (
                    REF_PREFIX + cls.parent if name != base and cls.parent else None
                )
                add(REF_PREFIX + name, "class", parent)
    # Composition edges within the lanes: value class -> owner class.
    for e in graph.composition_edges:
        if e.source in lane_classes and e.target in lane_classes:
            if e.source == "ExperimentReference":
                if e.target == "Phenotype":
                    add(
                        REF_PREFIX + "Phenotype",
                        "class",
                        REF_PREFIX + "ExperimentReference",
                    )
                continue  # the reference environment is not annotated (equals the experiment's)
            if e.target in ("Genotype",):
                continue  # genes go through the gene encoder
            add(e.target, "class", e.source)
    # Composition edges land on the lane ROOT of the target (Phenotype, Environment),
    # so every leaf reaches the owner through its inheritance chain; the parent lists
    # of subclasses stay inheritance-only, which _annotate_chain relies on.
    add(REF_PREFIX + "ExperimentReference", "class", ROOT)
    # Numeric and fixed value nodes.
    add("Temperature.value", "numeric", "Temperature")
    origins = dict(CONVERTED_FROM if converted_from is None else converted_from)
    for origin in sorted(set(origins.values())):
        if origin not in lane_classes:
            raise ValueError(f"converted_from origin {origin!r} is not a schema class")
        add(f"Experiment.converted_from:{origin}", "value", ROOT)
    # Value nodes from the media library and the scanned records.
    for media in _media_library():
        for name, parent in _value_nodes_of_media(media).items():
            add(name, "value", parent)
    for aero in ("aerobic", "anaerobic", "microaerobic"):
        add(f"Environment.aerobicity:{aero}", "value", "Environment")
    for unit in ("Celsius", "Kelvin", "Fahrenheit"):
        add(f"Temperature.unit:{unit}", "value", "Temperature")
    # Value nodes and record-introduced class nodes of the scanned records.
    all_recipes = list(recipes) + [annotation_recipe(e, r, origins) for e, r in records]
    for rec in all_recipes:
        for name, parent in rec.parents.items():
            if rec.kinds[name] == "class" and name in kinds:
                # a lane class reached by composition: keep its lane parent first
                add(name, "class", parent)
            else:
                add(name, rec.kinds[name], parent)
        missing_classes = [c for c in rec.classes if c not in kinds]
        if missing_classes:
            raise ValueError(
                f"record classes outside the schema lanes: {missing_classes}"
            )
    # Drop class nodes that are neither in the lanes' experiment subtree nor reachable
    # from a value node: keep everything (the DAG is small) but check every parent exists.
    missing = {p for ps in parents.values() for p in ps if p not in kinds}
    if missing:
        raise ValueError(f"vocabulary parents without nodes: {sorted(missing)}")
    names = sorted(kinds)
    return ExperimentOntologyVocabulary(
        nodes=[OntologyNode(name=n, kind=kinds[n], parents=parents[n]) for n in names],
        converted_from=origins,
    )


# 030's ``scan_store_records`` (a reader of 030's pool-store layout,
# ``torchcell.data.pool_store``) is not part of this copy: that module is absent from
# this tree, and 043 scans its raw store with
# ``experiments/043-expression-proteome-morphology/scripts/ontology_annotation_043.py``.


# --------------------------------------------------------------------------- #
# The encoder
# --------------------------------------------------------------------------- #
class OntologyDagEncoder(nn.Module):
    """DCell-style propagation of an annotation vector up the experiment DAG."""

    def __init__(
        self, vocabulary: ExperimentOntologyVocabulary, state_dim: int, token_dim: int
    ) -> None:
        """Build the per-node parameters over ``vocabulary``'s DAG.

        Args:
            vocabulary: the node set and its child -> parent edges.
            state_dim: width of every node's state.
            token_dim: width of the returned root token.
        """
        super().__init__()
        n = len(vocabulary)
        self.n_nodes = n
        self.state_dim = int(state_dim)
        children = vocabulary.children()
        depths = vocabulary.depths()
        self.max_depth = max(depths)
        self.root = vocabulary.index(vocabulary.root)
        # Child adjacency: any_child[parent, child] = 1.
        adj = torch.zeros(n, n)
        for p, cs in enumerate(children):
            for c in cs:
                adj[p, c] = 1.0
        self.register_buffer("any_child", adj, persistent=False)
        self.register_buffer("depth", torch.tensor(depths), persistent=False)
        # DCell's per-term layer over the CONCATENATED children, written as a sum of
        # per-child projections (so a variable number of children batches): each node
        # projects its own state toward its parent with ``child_proj``, the parent sums
        # them, adds its own annotation through ``own_proj`` and applies tanh. A sum,
        # not a mean: one differing child changes the parent by its full projection.
        self.child_proj = nn.Parameter(torch.empty(n, self.state_dim, self.state_dim))
        self.own_proj = nn.Parameter(torch.empty(n, 2, self.state_dim))
        self.bias = nn.Parameter(torch.zeros(n, self.state_dim))
        nn.init.orthogonal_(self.child_proj)
        nn.init.normal_(self.own_proj, std=1.0)
        self.out = nn.Linear(self.state_dim, int(token_dim))

    def _propagate(self, annotation: torch.Tensor) -> torch.Tensor:
        """Every node's state, ``[E, n_nodes, state_dim]``, leaves first."""
        if annotation.dim() != 2 or annotation.shape[1] != self.n_nodes:
            raise ValueError(
                f"annotation must be [E, {self.n_nodes}], got {tuple(annotation.shape)}"
            )
        x = annotation.to(self.child_proj.dtype)
        present = (x != 0).to(x.dtype)
        any_child = cast(torch.Tensor, self.any_child)
        depth = cast(torch.Tensor, self.depth)
        # A node is active when annotated or above an annotated node.
        active = present.clone()
        for _ in range(self.max_depth):
            active = torch.clamp(active + active @ any_child.T, max=1.0)
        own = torch.einsum(
            "enk,nkd->end", torch.stack([x, present], dim=-1), self.own_proj
        )
        states = x.new_zeros(x.shape[0], self.n_nodes, self.state_dim)
        for d in range(self.max_depth + 1):
            toward_parent = torch.einsum("ecd,cdk->eck", states, self.child_proj)
            agg = torch.einsum("pc,eck->epk", any_child, toward_parent)
            new = torch.tanh(agg + own + self.bias) * active.unsqueeze(-1)
            at = (depth == d).to(x.dtype).view(1, -1, 1)
            states = states * (1 - at) + new * at
        return states

    def forward(self, annotation: torch.Tensor) -> torch.Tensor:
        """``annotation`` ``[E, n_nodes]`` -> token ``[E, token_dim]``."""
        return cast(torch.Tensor, self.out(self._propagate(annotation)[:, self.root]))

    def states(self, annotation: torch.Tensor) -> torch.Tensor:
        """Every node's state, ``[E, n_nodes, state_dim]`` (for inspection)."""
        return self._propagate(annotation)


def describe_annotation(
    vocab: ExperimentOntologyVocabulary, vec: torch.Tensor
) -> list[str]:
    """Active node names (with the numeric value) of one annotation vector."""
    out = []
    for i, v in enumerate(vec.tolist()):
        if v != 0:
            n = vocab.nodes[i]
            out.append(f"{n.name}={v:g}" if n.kind == "numeric" else n.name)
    return out
