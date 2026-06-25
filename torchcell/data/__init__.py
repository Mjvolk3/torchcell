from .aggregate import Aggregator
from .data import ExperimentReferenceIndex, ReferenceIndex, compute_sha256_hash
from .deduplicate import Deduplicator
from .experiment_dataset import (
    ExperimentDataset,
    compute_experiment_reference_index_parallel,
    compute_experiment_reference_index_sequential,
    post_process,
)
from .genotype_aggregate import GenotypeAggregator
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

data = ["ExperimentReferenceIndex", "ReferenceIndex", "compute_md5_hash"]

# deduplicate
base_deduplicator = ["Deduplicator"]
deduplicators = ["MeanExperimentDeduplicator"]

# aggregate
base_aggregator = ["Aggregator"]
aggregators = ["GenotypeAggregator"]

# "Neo4jCellDataset"
# dataset = ["ExperimentDataset", "Neo4jQueryRaw", "Neo4jCellDataset"]
dataset = ["ExperimentDataset", "Neo4jCellDataset"]

functions = [
    "compute_experiment_reference_index_sequential",
    "compute_experiment_reference_index_parallel",
    "post_process",
]


gene_essentiality_to_fitness = ["GeneEssentialityToFitnessConverter"]

graph_processors = [
    "SubgraphRepresentation",
    "LazySubgraphRepresentation",
    "IncidenceSubgraphRepresentation",
    "Perturbation",
    "DCellGraphProcessor",
    "Unperturbed",
]

__all__ = (
    data
    + base_deduplicator
    + deduplicators
    + base_aggregator
    + aggregators
    + dataset
    + functions
    + graph_processors
)
