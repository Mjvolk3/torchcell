"""The lane arithmetic of ``torchcell.knowledge_graphs.perturbation_lanes`` on literal inputs.

The rule over every mapped dataset lives in
``tests/torchcell/adapters/test_loader_lane_exhaustiveness.py``; this file pins the
partition the lanes declare and the edge-method gate on synthetic leaf sets.
"""

from __future__ import annotations

from pathlib import Path

from torchcell.knowledge_graphs.perturbation_lanes import (
    ENVIRONMENT_EDGE_METHOD,
    GENOTYPE_EDGE_METHOD,
    LaneGaps,
    environment_lanes,
    environment_leaf_names,
    genotype_lanes,
    genotype_leaf_names,
    loader_leaf_imports,
    unserved_leaves,
)
from torchcell.provenance.schema_deps import load_default_surface

SURFACE = load_default_surface()


def test_one_lane_per_axis_serves_the_whole_union() -> None:
    """Since the ontology unification the lane arithmetic is the union itself."""
    assert genotype_lanes() == {"perturbation (chunked)": genotype_leaf_names()}
    assert environment_lanes() == {
        "environment perturbation (chunked)": environment_leaf_names()
    }
    assert {"GeneAdditionPerturbation", "LocusVariantCallPerturbation"} <= (
        genotype_leaf_names()
    )
    assert {
        "SmallMoleculePerturbation",
        "PhagePerturbation",
    } <= environment_leaf_names()


def test_a_leaf_is_a_gap_only_when_the_edge_that_walks_it_is_enabled() -> None:
    gene_addition = ["GeneAdditionPerturbation"]
    # the edge method with no node lane at all is the one remaining way to dangle
    assert unserved_leaves(gene_addition, [], [GENOTYPE_EDGE_METHOD]) == LaneGaps(
        genotype=["GeneAdditionPerturbation"]
    )
    # no edge method: the natural-isolate shape, complete by construction
    assert unserved_leaves(gene_addition, [], ["genotype (chunked)"]).empty
    # the one lane serves every genotype leaf
    assert unserved_leaves(
        gene_addition, [], ["perturbation (chunked)", GENOTYPE_EDGE_METHOD]
    ).empty
    assert unserved_leaves(
        [],
        ["SmallMoleculePerturbation", "PhagePerturbation"],
        [ENVIRONMENT_EDGE_METHOD],
    ) == LaneGaps(environment=["PhagePerturbation", "SmallMoleculePerturbation"])
    assert unserved_leaves(
        [],
        ["SmallMoleculePerturbation", "PhagePerturbation"],
        ["environment perturbation (chunked)", ENVIRONMENT_EDGE_METHOD],
    ).empty


def test_the_rule_reads_direct_imports_not_the_closure(tmp_path: Path) -> None:
    """A loader importing the deletion-strain leaf does not inherit its base's name."""
    loader = tmp_path / "loader.py"
    loader.write_text(
        "from torchcell.datamodels.schema import (\n"
        "    DeletionStrainPerturbation,\n"
        "    PhagePerturbation,\n"
        "    FitnessExperiment,\n"
        ")\n"
    )
    imports = loader_leaf_imports(loader, SURFACE)
    assert (imports.genotype, imports.environment) == (
        ["DeletionStrainPerturbation"],
        ["PhagePerturbation"],
    )
