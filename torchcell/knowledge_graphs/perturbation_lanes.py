# torchcell/knowledge_graphs/perturbation_lanes.py
# [[torchcell.knowledge_graphs.perturbation_lanes]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/perturbation_lanes
# Test file: tests/torchcell/knowledge_graphs/test_perturbation_lanes.py
# Fleet rule: tests/torchcell/adapters/test_loader_lane_exhaustiveness.py
"""Which perturbation leaves an adapter conf serves, and which it leaves without a node.

A conf (``torchcell/adapters/conf/*.yaml``) enables node methods and edge methods by
name. The edge methods walk EVERY perturbation of a genotype and of an environment, and
since the ontology unification (G, 2026-10-15) ONE node method per axis emits every
leaf, choosing the label by leaf: ``perturbation (chunked)`` serves every
``GenePerturbationType`` member (a ``VariantCallPerturbation`` as ``sequence variant
perturbation``, every other leaf as ``perturbation``) and ``environment perturbation
(chunked)`` every ``EnvironmentPerturbationType`` member (a ``PhagePerturbation`` as
``phage perturbation``, every other leaf as ``environment perturbation``). The retired
lanes (``bacterial perturbation (chunked)`` keyed on a class tuple, ``bacterial sequence
variant perturbation (chunked)``, ``phage perturbation (chunked)``) let a conf leave a
leaf its loader builds with a ``perturbation member of`` row and no start node, which the
full import's ``skip_bad_relationships: true`` dropped silently (the Carruthers
gene-addition gap, audit C 1.3). A gap is now possible only when a conf enables the edge
method and not the one node method of that axis.

This module is the ONE statement of that lane arithmetic. Three guards read it: the
loader-to-lane exhaustiveness test (direct schema imports of each mapped loader), the
rebuild preflight (``perturbation_leaves`` of each store's build manifest, in
``torchcell.database.build_dataset_lmdb.mapped_store_lanes``), and anything that asks
"is this leaf served under this conf". A leaf counts as unserved only when the matching
edge method is enabled: the natural-isolate confs omit ``perturbation to genotype
(chunked)`` by design (their perturbation sets live on the ``genotype`` node's lists),
so they need no allowlist.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from torchcell.provenance.schema_deps import SchemaSurface, loader_schema_deps

__all__ = [
    "GENOTYPE_EDGE_METHOD",
    "ENVIRONMENT_EDGE_METHOD",
    "GENOTYPE_NODE_METHODS",
    "ENVIRONMENT_NODE_METHODS",
    "LaneGaps",
    "LoaderLeafImports",
    "genotype_leaf_names",
    "environment_leaf_names",
    "genotype_lanes",
    "environment_lanes",
    "served_genotype_leaves",
    "served_environment_leaves",
    "unserved_leaves",
    "loader_leaf_imports",
]

GENOTYPE_EDGE_METHOD = "perturbation to genotype (chunked)"
"""The edge method that walks every genotype perturbation (``perturbation member of``)."""
ENVIRONMENT_EDGE_METHOD = "environment perturbation to environment (chunked)"
"""The edge method that walks every environment perturbation."""

GENOTYPE_NODE_METHODS = ("perturbation (chunked)",)
"""The one genotype-axis node method; it emits every ``GenePerturbationType`` member."""
ENVIRONMENT_NODE_METHODS = ("environment perturbation (chunked)",)
"""The one environment-axis node method; it emits every ``EnvironmentPerturbationType``
member."""


class LaneGaps(BaseModel):
    """Leaf class names that an enabled edge method walks and no enabled node lane emits."""

    model_config = ConfigDict(frozen=True)

    genotype: list[str] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)

    @property
    def empty(self) -> bool:
        """True when every walked leaf has a lane."""
        return not self.genotype and not self.environment

    def describe(self) -> str:
        """One line naming the unserved leaves and the edge that walks them."""
        parts: list[str] = []
        if self.genotype:
            parts.append(
                f"genotype leaves {self.genotype} walked by {GENOTYPE_EDGE_METHOD!r}"
            )
        if self.environment:
            parts.append(
                f"environment leaves {self.environment} walked by "
                f"{ENVIRONMENT_EDGE_METHOD!r}"
            )
        return "; ".join(parts) if parts else "every leaf served"


class LoaderLeafImports(BaseModel):
    """The perturbation leaves a loader module imports directly from the schema."""

    model_config = ConfigDict(frozen=True)

    genotype: list[str] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)


def genotype_leaf_names() -> frozenset[str]:
    """Class names of every ``GenePerturbationType`` member (nested unions flattened)."""
    from torchcell.datamodels.schema import GenePerturbationType

    return frozenset(cls.__name__ for cls in get_args(GenePerturbationType))


def environment_leaf_names() -> frozenset[str]:
    """Class names of every ``EnvironmentPerturbationType`` member."""
    from torchcell.datamodels.schema import EnvironmentPerturbationType

    return frozenset(cls.__name__ for cls in get_args(EnvironmentPerturbationType))


def genotype_lanes() -> dict[str, frozenset[str]]:
    """Node method name -> the genotype leaf class names it emits: the whole union."""
    return {GENOTYPE_NODE_METHODS[0]: genotype_leaf_names()}


def environment_lanes() -> dict[str, frozenset[str]]:
    """Node method name -> the environment leaf class names it emits: the whole union."""
    return {ENVIRONMENT_NODE_METHODS[0]: environment_leaf_names()}


def served_genotype_leaves(methods: Iterable[str]) -> frozenset[str]:
    """The genotype leaf class names the enabled node methods emit."""
    enabled = set(methods)
    lanes = genotype_lanes()
    return frozenset().union(*(lanes[m] for m in lanes if m in enabled))


def served_environment_leaves(methods: Iterable[str]) -> frozenset[str]:
    """The environment leaf class names the enabled node methods emit."""
    enabled = set(methods)
    lanes = environment_lanes()
    return frozenset().union(*(lanes[m] for m in lanes if m in enabled))


def unserved_leaves(
    genotype_leaves: Iterable[str],
    environment_leaves: Iterable[str],
    methods: Iterable[str],
) -> LaneGaps:
    """The leaves among the given ones that the conf walks with an edge and emits no node for.

    A leaf is a gap only when the matching edge method is enabled: without the edge no
    row can dangle, and a conf that serves its perturbation set on the genotype node
    alone (the natural isolates) is complete by construction.
    """
    enabled = set(methods)
    genotype = (
        sorted(set(genotype_leaves) - served_genotype_leaves(enabled))
        if GENOTYPE_EDGE_METHOD in enabled
        else []
    )
    environment = (
        sorted(set(environment_leaves) - served_environment_leaves(enabled))
        if ENVIRONMENT_EDGE_METHOD in enabled
        else []
    )
    return LaneGaps(genotype=genotype, environment=environment)


def loader_leaf_imports(loader_path: Path, surface: SchemaSurface) -> LoaderLeafImports:
    """The perturbation leaves ``loader_path`` imports DIRECTLY from the schema.

    Direct imports (:func:`loader_schema_deps`), not the transitive closure: the closure
    reaches every base class a leaf subclasses (``DeletionStrainPerturbation`` is a
    ``DeletionPerturbation``), which would flag 55 of 170 mapped datasets for leaves they
    never build (review 3, 2026-10-15, finding 6). A direct import is the loader saying it
    builds the leaf; an imported-but-unused leaf is a false positive whose fix is to drop
    the import.
    """
    deps = loader_schema_deps(loader_path, surface)
    return LoaderLeafImports(
        genotype=sorted(deps & genotype_leaf_names()),
        environment=sorted(deps & environment_leaf_names()),
    )
