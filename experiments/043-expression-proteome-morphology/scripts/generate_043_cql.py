# experiments/043-expression-proteome-morphology/scripts/generate_043_cql.py
# [[experiments.043-expression-proteome-morphology.scripts.generate_043_cql]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/generate_043_cql
"""Generate the experiment 043 query: expression, proteome and morphology over 15 datasets.

Writes one ``UNION ALL`` Cypher file, byte-identical, to two places:

* ``experiments/043-expression-proteome-morphology/queries/expression_proteome_morphology_043.cql``
  (the copy the store build reads), and
* ``torchcell/knowledge_graphs/queries/expression_proteome_morphology_043.cql`` (the copy
  the supported-query registry points at; ``cql_path`` must be inside that package).

Every block has the property-shaped RETURN of the supported query
``torchcell/knowledge_graphs/queries/expression_proteome_morphology.cql``
(``e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized``) and keeps
the phenotype ``graph_level`` filter. The genotype clause takes one of three shapes:

* ``all`` (deletion and allele panels): at least one perturbation, nothing else. Every
  served record is admitted; the genome never gates a record (owner, 2026-10-12: "the
  genome is separate.. it doesn't gate anything... we do that at modeling time"). Until
  2026-10-23 this clause also required every perturbation to name a gene in
  ``$gene_set``, which dropped 188 served records in the v1 build (job 5856,
  ``results/build_043_store_census.json``): 75 Messner 2023 and 70 Nadal-Ribelles 2025
  records whose deleted gene is a ``ReleasedIdentifierPerturbation`` (the released symbol
  in ``systematic_gene_name``), and 2 Kemmeren 2014, 1 Hu 2007 and 40 Ohya 2005 records
  whose ORFs are outside the gene set. The released name is resolved to an index
  position at read time (``dataset_043``), and a name with no position is a typed gap of
  the representation, never a drop.
* ``isolate`` (Caudal 2024, Muenzner 2024): no genotype clause beyond the ``Genotype``
  label. Natural isolates carry accessory-ORF absences and thousands of
  ``sequence_variant`` perturbations outside S288C, and one Muenzner 2024 proteome strain
  carries none, so no perturbation test applies.
* ``none``: no gene clause and no ``Genotype`` label. Boocock 2025 and Nguessan 2025 have
  ``SegregantGenotype`` genotypes (``segregant genotype`` is_a ``genotype`` in
  ``biocypher/config/torchcell_schema_config.yaml``, reached through
  ``GenotypeMemberOf``), and Vaishnav 2022 carries a
  ``VariablePromoterReporterPerturbation`` with no gene.

``$gene_set`` is no longer read by the query; ``Neo4jCellDataset`` still passes it.

Run from the repo root:
    python experiments/043-expression-proteome-morphology/scripts/generate_043_cql.py
"""

from __future__ import annotations

import os.path as osp
from typing import Literal

from torchcell.data.neo4j_query_raw import (
    PARTITION_MARKER,
    partition_queries,
    single_session_query,
)

GeneClause = Literal["all", "isolate", "none"]

# (dataset id, phenotype class, graph_level, gene clause); order is the UNION order. The
# phenotype class is the one the dataset's recorded closure in
# database/releases/2026.10.17-4295f7ca.closures.json carries (comment only; the query
# filters on graph_level, not on the class).
BLOCKS: list[tuple[str, str, str, GeneClause]] = [
    ("MicroarrayKemmeren2014Dataset", "MicroarrayExpressionPhenotype", "node", "all"),
    ("SmMicroarraySameith2015Dataset", "MicroarrayExpressionPhenotype", "node", "all"),
    ("DmMicroarraySameith2015Dataset", "MicroarrayExpressionPhenotype", "node", "all"),
    (
        "MicroarrayScoreHu2007Dataset",
        "MicroarrayExpressionScorePhenotype",
        "node",
        "all",
    ),
    (
        "TfDeletionPseudobulkJackson2020Dataset",
        "SingleCellExpressionPhenotype",
        "node",
        "all",
    ),
    (
        "NadalRibellesPerturbSeq2025Dataset",
        "PseudobulkExpressionPhenotype",
        "node",
        "all",
    ),
    ("ProteomeMessner2023Dataset", "ProteinAbundancePhenotype", "node", "all"),
    ("ProteomeZelezniak2018Dataset", "ProteinAbundancePhenotype", "node", "all"),
    ("ScmdOhya2005Dataset", "CalMorphPhenotype", "global", "all"),
    (
        "CaudalPanTranscriptome2024Dataset",
        "RNASeqExpressionPhenotype",
        "node",
        "isolate",
    ),
    ("ProteomeMuenzner2024Dataset", "ProteinAbundancePhenotype", "node", "isolate"),
    (
        "ProteinTurnoverMuenzner2024Dataset",
        "ProteinTurnoverPhenotype",
        "node",
        "isolate",
    ),
    ("Boocock2025Dataset", "SingleCellExpressionPhenotype", "node", "none"),
    ("Nguessan2025Dataset", "SingleCellExpressionPhenotype", "node", "none"),
    (
        "PooledPromoterExpressionVaishnav2022Dataset",
        "PooledSortSeqExpressionPhenotype",
        "node",
        "none",
    ),
]

# Served experiment counts in release 2026.10.17-4295f7ca
# (database/releases/2026.10.17-4295f7ca.json, n_experiments per dataset).
EXPECTED_COUNTS: dict[str, int] = {
    "MicroarrayKemmeren2014Dataset": 1484,
    "SmMicroarraySameith2015Dataset": 82,
    "DmMicroarraySameith2015Dataset": 72,
    "MicroarrayScoreHu2007Dataset": 263,
    "TfDeletionPseudobulkJackson2020Dataset": 784,
    "NadalRibellesPerturbSeq2025Dataset": 6188,
    "ProteomeMessner2023Dataset": 4699,
    "ProteomeZelezniak2018Dataset": 97,
    "ScmdOhya2005Dataset": 4718,
    "CaudalPanTranscriptome2024Dataset": 943,
    "ProteomeMuenzner2024Dataset": 795,
    "ProteinTurnoverMuenzner2024Dataset": 55,
    "Boocock2025Dataset": 79516,
    "Nguessan2025Dataset": 2951,
    "PooledPromoterExpressionVaishnav2022Dataset": 71158,
}

QUERY_NAME = "expression_proteome_morphology_043.cql"
#: The partitioned CANDIDATE (2026-10-23): the same text with ``{partition}`` once per
#: block right after the dataset filter (``torchcell.data.neo4j_query_raw.PARTITION_MARKER``),
#: written to the experiment folder only. Removing the markers gives QUERY_NAME byte for
#: byte; ``main`` asserts it.
PARTITIONED_QUERY_NAME = "expression_proteome_morphology_043_partitioned.cql"

HEADER = """\
// expression_proteome_morphology_043.cql
// [[experiments.043-expression-proteome-morphology]]
//
// Experiment 043 query: expression, proteome and CalMorph morphology over 15 datasets of
// KG 4.0 (release 2026.10.17-4295f7ca). GENERATED by
// experiments/043-expression-proteome-morphology/scripts/generate_043_cql.py, which writes
// this text byte-identical to experiments/043-expression-proteome-morphology/queries/ and
// torchcell/knowledge_graphs/queries/. Edit the generator, never this file.
//
// Block shape of torchcell/knowledge_graphs/queries/expression_proteome_morphology.cql
// (property-shaped RETURN, no medium filter). Every served record is admitted: the genome
// never gates a record (no $gene_set test since 2026-10-23, store v2). Clause per block:
//   all     = at least one perturbation (deletion and allele panels)
//   isolate = Genotype label only (natural isolates, one with no perturbation)
//   none    = no gene clause, unlabeled genotype (segregants, promoter library)"""


def _gene_clause(clause: GeneClause) -> str:
    perts = "[(g)<-[:PerturbationMemberOf]-(pert) | pert]"
    if clause == "all":
        return f"\n AND SIZE({perts}) > 0"
    return ""


def block(
    dataset_id: str,
    phenotype: str,
    graph_level: str,
    clause: GeneClause,
    partitioned: bool = False,
) -> str:
    """One dataset's ``MATCH ... RETURN e_serialized, ref_serialized`` block; with
    ``partitioned`` the dataset filter carries the partition marker.
    """
    marker = PARTITION_MARKER if partitioned else ""
    genotype = "(g:Genotype)" if clause != "none" else "(g)"
    return f"""// {dataset_id} ({phenotype}, graph_level '{graph_level}', gene clause {clause})
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = '{dataset_id}'{marker}
MATCH (e)<-[:GenotypeMemberOf]-{genotype}
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = '{graph_level}'{_gene_clause(clause)}
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized"""


def render(partitioned: bool = False) -> str:
    """The full query text (the partitioned candidate with ``partitioned``)."""
    body = "\n\nUNION ALL\n\n".join(block(*b, partitioned=partitioned) for b in BLOCKS)
    return f"{HEADER}\n\n{body}\n"


def main() -> None:
    """Write the query to the experiment folder and the supported-query package."""
    here = osp.dirname(osp.abspath(__file__))
    repo = osp.abspath(osp.join(here, "..", "..", ".."))
    targets = [
        osp.join(repo, "experiments", "043-expression-proteome-morphology", "queries"),
        osp.join(repo, "torchcell", "knowledge_graphs", "queries"),
    ]
    text = render()
    for qdir in targets:
        path = osp.join(qdir, QUERY_NAME)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")
    candidate = render(partitioned=True)
    if single_session_query(candidate) != text:
        raise RuntimeError("the partitioned query without its markers differs")
    n_partitions = len(partition_queries(candidate, 2))
    path = osp.join(targets[0], PARTITIONED_QUERY_NAME)
    with open(path, "w") as f:
        f.write(candidate)
    print(f"wrote {path} ({n_partitions} partitions at prefix length 2)")
    print(f"{len(BLOCKS)} blocks, expected {sum(EXPECTED_COUNTS.values())} records")


if __name__ == "__main__":
    main()
