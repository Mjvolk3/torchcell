"""``Perturbation(released_name_resolver=...)``: released symbols become gene nodes at read time.

A loader stores a deleted gene under the name its source released (#1045); a symbol-named
strain sits on ``ReleasedIdentifierPerturbation`` with the symbol in ``systematic_gene_name``.
Without a resolver the stock processor maps that name to nothing. These tests pin the
resolving path: the strict rule, the SGD standard-name rule for an ambiguous alias, a refusal
listed on the record with the perturbation left unmapped, a non-deletion perturbation left
alone, and the stock output unchanged when no resolver is given.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from sortedcontainers import SortedDict

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import Perturbation
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
)
from torchcell.datasets.gene_alias_resolution import (
    RULE_REFUSED,
    RULE_SGD_STANDARD_NAME,
    RULE_STRICT,
    ReleasedNameResolver,
    is_deletion_kind,
)
from torchcell.datasets.scerevisiae.cachera2023 import _betaxanthin_cassette
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
PHENOTYPES: list[Any] = [FitnessPhenotype]
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


class _Genome:
    """ONE1 has one candidate; AMB is the standard name of YAL002W and an alias of YAL003W;
    TIE is an alias of two ORFs with no standard-name owner; NOPE is not in the genome.
    """

    gene_set = {"YAL001C", "YAL002W", "YAL003W"}
    alias_to_systematic: dict[str, list[str]] = {
        "ONE1": ["YAL001C"],
        "AMB": ["YAL002W", "YAL003W"],
        "TIE": ["YAL001C", "YAL003W"],
    }
    feature_index: dict[str, Any] = {"standard_to_ids": {"AMB": ["YAL002W"]}}


def _released(symbol: str) -> ReleasedIdentifierPerturbation:
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=symbol,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=symbol,
        identifier_kind="gene column entry, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


def _record(perturbations: list[Any]) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=0.5),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _cell_graph() -> Any:
    base = nx.Graph()
    base.add_nodes_from(GENES)
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {"base": GeneGraph(name="base", graph=base, max_gene_set=GENES)}
        )
    )
    return to_cell_data(multigraph)


def test_resolver_rules_and_outcomes() -> None:
    """Strict for a live name or a single candidate, standard-name for AMB, refused for
    TIE (two candidates, no owner) and NOPE (not in the genome); outcomes count names.
    """
    r = ReleasedNameResolver(_Genome())
    assert r.resolve("YAL001C") == ("YAL001C", RULE_STRICT, "live systematic name")
    assert r.resolve("one1") == ("YAL001C", RULE_STRICT, "single candidate ORF")
    assert r.resolve("AMB") == (
        "YAL002W",
        RULE_SGD_STANDARD_NAME,
        "ambiguous among ['YAL002W', 'YAL003W']; SGD standard name of YAL002W",
    )
    assert r.resolve("TIE") == (
        None,
        RULE_REFUSED,
        "ambiguous_alias_unpinned; candidates ['YAL001C', 'YAL003W']",
    )
    assert r.resolve("NOPE") == (None, RULE_REFUSED, "not_in_genome; candidates []")
    assert r.resolve("NOPE") == r.resolve("nope")  # memoized, case-folded
    assert r.outcomes == {RULE_STRICT: 2, RULE_SGD_STANDARD_NAME: 1, RULE_REFUSED: 2}


def test_is_deletion_kind_names_the_two_leaves() -> None:
    assert is_deletion_kind("kanmx_deletion") is True
    assert is_deletion_kind("deletion") is True
    assert is_deletion_kind("released_identifier") is True
    assert is_deletion_kind("gene_addition") is False
    assert is_deletion_kind("crispr_interference") is False


def test_released_symbol_becomes_a_gene_node_and_a_refusal_is_listed() -> None:
    """ONE1 -> YAL001C (index 0), AMB -> YAL002W (index 1); NOPE stays unmapped and listed."""
    data = [
        _record([_released("ONE1")]),
        _record([_released("AMB"), _released("NOPE")]),
    ]
    out = Perturbation(released_name_resolver=ReleasedNameResolver(_Genome())).process(
        _cell_graph(), PHENOTYPES, data
    )
    assert out["gene"].perturbation_indices.tolist() == [0, 1]
    assert out["gene"].pert_mask.tolist() == [True, True, False]
    assert sorted(out["gene"].perturbed_genes) == ["NOPE", "YAL001C", "YAL002W"]
    assert out["gene"].unresolved_perturbations == [
        "NOPE: not_in_genome; candidates []"
    ]
    # The stored records are untouched: the input still carries the released symbol.
    assert (
        data[0]["experiment"].genotype.perturbations[0].systematic_gene_name == "ONE1"
    )


def test_a_gene_addition_is_not_resolved_and_a_node_name_passes_through() -> None:
    """Cachera's heterologous cassette gene CYP76AD1 is left alone (not listed, not mapped);
    a KanMX deletion already named by its ORF maps as before.
    """
    # Cachera's own Btx-cassette additions (CYP76AD1, DOD, ARO4 K229L, ARO7 G141S).
    additions = _betaxanthin_cassette()
    assert [a.perturbation_type for a in additions] == ["gene_addition"] * 4
    addition = next(a for a in additions if a.systematic_gene_name == "CYP76AD1")
    deletion = KanMxDeletionPerturbation(
        systematic_gene_name="YAL003W", perturbed_gene_name="YAL003W"
    )
    out = Perturbation(released_name_resolver=ReleasedNameResolver(_Genome())).process(
        _cell_graph(), PHENOTYPES, [_record([deletion, addition])]
    )
    assert out["gene"].perturbation_indices.tolist() == [2]
    assert out["gene"].unresolved_perturbations == []


def test_without_a_resolver_the_output_is_the_stock_output() -> None:
    """No resolver: a released symbol maps to nothing, and no new attribute appears."""
    out = Perturbation().process(
        _cell_graph(), PHENOTYPES, [_record([_released("ONE1")])]
    )
    assert out["gene"].perturbation_indices.tolist() == []
    assert out["gene"].perturbed_genes == ["ONE1"]
    assert "unresolved_perturbations" not in out["gene"]
