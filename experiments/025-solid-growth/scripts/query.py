# experiments/025-solid-growth/scripts/query.py
# [[experiments.025-solid-growth.scripts.query]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/query
"""Build the all-solid-growth dataset from the served knowledge graph.

Queries every solid-medium growth record (fitness smf/dmf/tmf, interactions dmi/tmi,
essentiality, synthetic lethality) from the uncapped 83M-node build, converts
essentiality and synthetic lethality to fitness (CompositeFitnessConverter),
deduplicates replicate experiments (MeanExperimentDeduplicator), and aggregates by
genotype (GenotypeAggregator). The result carries fitness AND gene-interaction labels
per perturbation order, indexed by phenotype_label_index / perturbation_count_index /
dataset_name_index so smf/dmf/tmf/dmi/tmi slices stay queryable.

Connection resolves from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (torchcell.database
.connection), defaulting to the locally served instance. Run from the repo root:

    python experiments/025-solid-growth/scripts/query.py
"""

import json
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


def main() -> None:
    """Query, convert, deduplicate, aggregate, and report the index breakdown."""
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    assert data_root is not None, "DATA_ROOT must be set in .env"

    with open("experiments/025-solid-growth/queries/001_all_solid_growth.cql") as f:
        query = f.read()

    # go_root must be explicit: the relative default resolves against cwd, misses the
    # mirror, and falls into a live GO download that 403s (observed in the smoke run).
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
    )
    print(f"gene_set: {len(genome.gene_set)} genes")

    dataset_root = osp.join(
        data_root, "data/torchcell/experiments/025-solid-growth/001-full-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs=None,
        node_embeddings=None,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(f"dataset length: {len(dataset)}")

    # The index breakdown is the deliverable's contract: smf/dmf/tmf = fitness at
    # perturbation count 1/2/3, dmi/tmi = gene_interaction at count 2/3. Persist it
    # so the slices are queryable without loading the LMDB.
    summary = {
        "length": len(dataset),
        "phenotype_label_index": {
            k: len(v) for k, v in dataset.phenotype_label_index.items()
        },
        "perturbation_count_index": {
            str(k): len(v) for k, v in dataset.perturbation_count_index.items()
        },
        "dataset_name_index": {
            k: len(v) for k, v in dataset.dataset_name_index.items()
        },
    }
    out = "experiments/025-solid-growth/results/dataset_index_summary.json"
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    dataset.close_lmdb()
    print("finished")


if __name__ == "__main__":
    main()
