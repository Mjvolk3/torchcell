# tests/torchcell/data/test_segregant_genotype_without_perturbations.py
# [[tests.torchcell.data.test_segregant_genotype_without_perturbations]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_segregant_genotype_without_perturbations.py
"""A ``SegregantGenotype`` has no ``perturbations`` list, and the query path keeps it.

The segregant genotype carries ``cross``, ``segregant_id``, parents and haplotype blocks,
and by design none of the gene-keyed properties of the plain genotype. Every site on the
raw-to-sample path that reads the perturbation list treats a missing list as zero
perturbed genes:

* ``Neo4jQueryRaw.extract_systematic_gene_names`` returns ``[]`` for the dumped genotype
  (it raised ``TypeError: 'NoneType' object is not iterable`` on build job 5851).
* ``RawStageGrouping.prepare`` summarizes the record with perturbation count 0.
* ``Perturbation.process`` keeps the record: no perturbed genes, empty
  ``perturbation_indices``, ``pert_mask`` all False, the phenotype stored.

The record is shaped on a Boocock 2025 record: a ``SegregantExpressionExperiment`` with a
BY x RM ``SegregantGenotype`` and a ``SingleCellExpressionPhenotype`` of UMI counts.
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
from sortedcontainers import SortedDict
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import Perturbation
from torchcell.data.neo4j_cell import RawStageGrouping
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels.schema import (
    ArtifactRef,
    Environment,
    HaplotypeBlock,
    Media,
    ReferenceGenome,
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
    SegregantGenotype,
    SegregantParent,
    SingleCellExpressionPhenotype,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
PHENOTYPES: list[Any] = [SingleCellExpressionPhenotype]  # the processor takes classes
ENVIRONMENT = Environment(media=Media(name="SM", state="liquid", is_synthetic=True))
GENOME = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


def _parent(name: str, peter: str | None) -> SegregantParent:
    return SegregantParent(
        name=name,
        peter_strain_id=peter,
        assembly_ref=ArtifactRef(
            tier="genomes",
            key="peter2018_1011_assemblies",
            path="1011Assemblies.tar.gz",
            member="GENOMES_ASSEMBLED/AAA_6.re.fa",
            sha256="5" * 64,
        )
        if peter
        else ArtifactRef(
            tier="genomes",
            key="sgd_S288C_R64-4-1_20230830",
            path="S288C_reference_sequence_R64-4-1_20230830.fsa",
            sha256="d" * 64,
        ),
        engineered_background=f"{name} MatA (YLK1879)",
    )


def _genotype() -> SegregantGenotype:
    return SegregantGenotype(
        cross="BYxRM",
        segregant_id="0",
        parent_1=_parent("BY", None),
        parent_2=_parent("RM", "AAA"),
        blocks=[
            HaplotypeBlock(
                chromosome="chrI", start=33_040, end=127_853, parent=1, n_markers=358
            )
        ],
        call_method="whole-genome-sequencing hard calls",
        marker_matrix_sha256="2" * 64,
    )


def _phenotype() -> SingleCellExpressionPhenotype:
    """Two cells in one library; 3 + 5 UMIs on two genes out of 10 of 4 features."""
    return SingleCellExpressionPhenotype(
        umi_count={"YAL001C": 3, "YAL003W": 5},
        marker_umi_count={},
        total_umis=10,
        n_features=4,
        feature_universe_sha256="b" * 64,
        n_cells=2,
        cell_cycle_stage_counts={"G1": 2},
        library_cell_counts={"lib1": 2},
        cell_barcodes=["AAAC", "AAAG"],
        cell_libraries=["lib1", "lib1"],
        cell_total_umis=[4, 6],
    )


def _record() -> dict[str, Any]:
    return {
        "experiment": SegregantExpressionExperiment(
            dataset_name="Boocock2025Dataset",
            genotype=_genotype(),
            environment=ENVIRONMENT,
            phenotype=_phenotype(),
        ),
        "experiment_reference": SegregantExpressionExperimentReference(
            dataset_name="Boocock2025Dataset",
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=_phenotype(),
        ),
    }


def _record_dict() -> dict[str, Any]:
    """The stored JSON form of the record, as the raw stage writes and re-reads it."""
    record = _record()
    return {key: json.loads(model.model_dump_json()) for key, model in record.items()}


def _cell_graph() -> HeteroData:
    base = nx.Graph()
    base.add_nodes_from(GENES)
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {"base": GeneGraph(name="base", graph=base, max_gene_set=GENES)}
        )
    )
    return to_cell_data(multigraph)


def test_the_dumped_segregant_genotype_has_no_perturbations_key() -> None:
    """The premise: the stored genotype carries the segregant fields and no list."""
    genotype = _record_dict()["experiment"]["genotype"]
    assert "perturbations" not in genotype
    assert genotype["cross"] == "BYxRM"
    assert genotype["segregant_id"] == "0"


def test_extract_systematic_gene_names_is_empty_for_a_segregant() -> None:
    genotype = _record_dict()["experiment"]["genotype"]
    assert Neo4jQueryRaw.extract_systematic_gene_names(genotype) == []


def test_raw_stage_grouping_summarizes_a_segregant_with_zero_perturbations() -> None:
    """Summary: type, label, dataset, perturbation count 0, no scalar label values."""
    grouping = RawStageGrouping(key_fn=lambda record: "k", summarize=True)
    key, summary = grouping.prepare(_record_dict())
    assert key == "k"
    assert summary == ("segregant_expression", "umi_count", "Boocock2025Dataset", 0, ())


def test_perturbation_processor_keeps_a_segregant_with_no_perturbed_genes() -> None:
    """Empty indices, pert_mask all False, mask all True, umi counts [3, 5] stored."""
    out = Perturbation().process(_cell_graph(), PHENOTYPES, [_record()])
    assert out["gene"].num_nodes == 3
    assert out["gene"].perturbed_genes == []
    assert out["gene"].perturbation_indices.tolist() == []
    assert out["gene"].pert_mask.tolist() == [False, False, False]
    assert out["gene"].mask.tolist() == [True, True, True]
    assert out["gene"].phenotype_types == ["umi_count"]
    assert out["gene"].phenotype_values.tolist() == [3.0, 5.0]
    assert out["gene"].phenotype_sample_indices.tolist() == [0, 0]
