# torchcell/scratch/neo4j_preprocessed_cell_demo
# [[torchcell.scratch.neo4j_preprocessed_cell_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/neo4j_preprocessed_cell_demo
"""Example: Preprocess a dataset for faster training.

Builds the 006-kuzmin-tmi small-build Neo4jCellDataset and writes its
preprocessed copy (``001-small-build-preprocessed-lazy``) as a
Neo4jPreprocessedCellDataset.

Moved verbatim from torchcell/data/neo4j_preprocessed_cell.py on 2026-10-06 (test
campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/neo4j_preprocessed_cell_demo.py

Needs ``DATA_ROOT`` and ``EXPERIMENT_ROOT`` in ``.env``, the SGD genome, GO,
STRING and TFLink data under ``DATA_ROOT``, and the
``data/torchcell/experiments/006-kuzmin-tmi/001-small-build`` dataset built (or a
reachable Neo4j to build it from the query). No GPU.
"""

import os
from typing import cast

from torchcell.data.neo4j_preprocessed_cell import Neo4jPreprocessedCellDataset


def main_preprocess() -> None:
    """Example: Preprocess a dataset for faster training."""
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import LazySubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.graph import GeneMultiGraph, SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
    EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql")
    ) as f:
        query = f.read()

    # Setup genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Create source dataset
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    source_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=cast(
            GeneMultiGraph,
            {"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        ),
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=None,  # Don't process yet
    )

    print(f"Source dataset length: {len(source_dataset)}")

    # Create preprocessed dataset and run one-time preprocessing
    preprocessed_root = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-lazy",
    )

    preprocessed_dataset = Neo4jPreprocessedCellDataset(
        root=preprocessed_root, source_dataset=source_dataset
    )

    # One-time preprocessing with LazySubgraphRepresentation
    print("\nStarting one-time preprocessing...")
    print("This will take ~50 minutes for 300K samples")
    print("But will save ~280 seconds per epoch during training!")

    preprocessed_dataset.preprocess_from_source(
        source_dataset=source_dataset, graph_processor=LazySubgraphRepresentation()
    )

    # Test loading
    print("\nTesting preprocessed data loading...")
    sample = preprocessed_dataset[0]
    print(f"Sample keys: {sample.keys}")
    print(f"Gene nodes: {sample['gene'].num_nodes}")

    # Cleanup
    source_dataset.close_lmdb()
    preprocessed_dataset.close_lmdb()

    print("\nPreprocessing complete! Use this dataset for training:")
    print(
        f"  preprocessed_dataset = Neo4jPreprocessedCellDataset(root='{preprocessed_root}')"
    )


if __name__ == "__main__":
    main_preprocess()
