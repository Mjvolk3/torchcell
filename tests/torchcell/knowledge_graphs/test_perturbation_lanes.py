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


def test_the_lanes_partition_the_unions_as_the_adapter_declares() -> None:
    lanes = genotype_lanes()
    assert lanes["perturbation (chunked)"] == genotype_leaf_names()
    bacterial = lanes["bacterial perturbation (chunked)"]
    variant = lanes["bacterial sequence variant perturbation (chunked)"]
    assert bacterial.isdisjoint(variant)
    assert bacterial | variant < genotype_leaf_names()
    assert "GeneAdditionPerturbation" not in bacterial | variant
    env = environment_lanes()
    assert env["phage perturbation (chunked)"] == {"PhagePerturbation"}
    assert env["environment perturbation (chunked)"] | {"PhagePerturbation"} == (
        environment_leaf_names()
    )


def test_a_leaf_is_a_gap_only_when_the_edge_that_walks_it_is_enabled() -> None:
    gene_addition = ["GeneAdditionPerturbation"]
    bacterial_only = ["bacterial perturbation (chunked)", GENOTYPE_EDGE_METHOD]
    assert unserved_leaves(gene_addition, [], bacterial_only) == LaneGaps(
        genotype=["GeneAdditionPerturbation"]
    )
    # no edge method: the natural-isolate shape, complete by construction
    assert unserved_leaves(
        gene_addition, [], ["bacterial perturbation (chunked)"]
    ).empty
    # the yeast lane serves every genotype leaf
    assert unserved_leaves(
        gene_addition, [], ["perturbation (chunked)", GENOTYPE_EDGE_METHOD]
    ).empty
    phage_only = ["phage perturbation (chunked)", ENVIRONMENT_EDGE_METHOD]
    assert unserved_leaves(
        [], ["SmallMoleculePerturbation", "PhagePerturbation"], phage_only
    ) == (LaneGaps(environment=["SmallMoleculePerturbation"]))
    assert unserved_leaves([], ["PhagePerturbation"], phage_only).empty


def test_the_rule_reads_direct_imports_not_the_closure(tmp_path: Path) -> None:
    """A loader importing the bacterial deletion leaf does not inherit its base's name."""
    loader = tmp_path / "loader.py"
    loader.write_text(
        "from torchcell.datamodels.schema import (\n"
        "    BacterialDeletionPerturbation,\n"
        "    PhagePerturbation,\n"
        "    FitnessExperiment,\n"
        ")\n"
    )
    imports = loader_leaf_imports(loader, SURFACE)
    assert (imports.genotype, imports.environment) == (
        ["BacterialDeletionPerturbation"],
        ["PhagePerturbation"],
    )
