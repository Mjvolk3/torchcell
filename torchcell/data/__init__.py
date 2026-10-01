"""Datasets, deduplicators, aggregators, and graph processors for torchcell data."""

from .aggregate import Aggregator
from .data import ExperimentReferenceIndex, ReferenceIndex, compute_sha256_hash
from .deduplicate import Deduplicator
from .experiment_dataset import (
    ExperimentDataset,
    RawSha256MismatchError,
    compute_experiment_reference_index_parallel,
    compute_experiment_reference_index_sequential,
    copy_verified,
    file_sha256,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
    write_verified,
)
from .genotype_aggregate import DeletionKeyedGenotypeAggregator, GenotypeAggregator
from .graph_processor import (
    DCellGraphProcessor,
    IncidenceSubgraphRepresentation,
    LazySubgraphRepresentation,
    Perturbation,
    SubgraphRepresentation,
    Unperturbed,
)
from .mean_experiment_deduplicate import MeanExperimentDeduplicator

# from .neo4j_query_raw import Neo4jQueryRaw
from .neo4j_cell import Neo4jCellDataset  # FLAG

__all__ = [
    "ExperimentReferenceIndex",
    "ReferenceIndex",
    "compute_sha256_hash",
    "Deduplicator",
    "MeanExperimentDeduplicator",
    "Aggregator",
    "DeletionKeyedGenotypeAggregator",
    "GenotypeAggregator",
    "ExperimentDataset",
    "Neo4jCellDataset",
    "compute_experiment_reference_index_sequential",
    "compute_experiment_reference_index_parallel",
    "post_process",
    "RawSha256MismatchError",
    "file_sha256",
    "verify_sha256",
    "verify_raw_files",
    "copy_verified",
    "write_verified",
    "link_verified",
    "SubgraphRepresentation",
    "LazySubgraphRepresentation",
    "IncidenceSubgraphRepresentation",
    "Perturbation",
    "DCellGraphProcessor",
    "Unperturbed",
]
