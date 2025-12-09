from .data import ExperimentReferenceIndex, ReferenceIndex, compute_sha256_hash

# from .neo4j_query_raw import Neo4jQueryRaw
from .neo4j_cell import Neo4jCellDataset  # FLAG
from .experiment_dataset import ExperimentDataset
from .experiment_dataset import (
    post_process,
    compute_experiment_reference_index_sequential,
    compute_experiment_reference_index_parallel,
)
from .deduplicate import Deduplicator
from .mean_experiment_deduplicate import MeanExperimentDeduplicator
from .aggregate import Aggregator
from .genotype_aggregate import GenotypeAggregator
from .graph_processor import (
    SubgraphRepresentation,
    LazySubgraphRepresentation,
    IncidenceSubgraphRepresentation,
    Perturbation,
    DCellGraphProcessor,
    Unperturbed,
)

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
